import os
import sqlite3
import re
from datetime import datetime
from typing import Annotated, Literal, TypedDict, List, Optional, Union

import pandas as pd
from pydantic import BaseModel, Field, validator
from langchain_openai import AzureChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.tools import tool
from langchain_community.utilities import SQLDatabase
from langgraph.graph import StateGraph, END


# --- 1. Pydantic Schemas (数据校验) ---

class UserIntent(BaseModel):
    """识别用户意图的 Schema"""
    intent_type: Literal['qa', 'calculation', 'summarization', 'general'] = Field(description="用户查询的类型")
    confidence: float = Field(ge=0.0, le=1.0, description="识别置信度")
    reasoning: str = Field(description="识别意图的理由")


class AnswerResponse(BaseModel):
    """最终输出的 Schema"""
    content: str = Field(description="回复内容")
    confidence: float = Field(ge=0.0, le=1.0, description="回答的置信度")
    timestamp: datetime = Field(default_factory=datetime.now, description="响应时间戳")


# --- 2. 状态定义 ---

class AgentState(TypedDict):
    query: str
    intent: Optional[UserIntent]
    sql_results: Optional[str]
    tool_output: Optional[str]
    sql_answer: Optional[pd.DataFrame]
    final_answer: Optional[AnswerResponse]


# --- 3. Tools (工具实现) ---

@tool
def calculator(expression: str) -> str:
    """计算数学表达式。仅允许数字和基础运算符 (+-*/().)。"""
    # 安全验证：只允许数学字符
    if not re.match(r"^[0-9+\-*/().\s]+$", expression):
        return "错误：表达式包含非法字符。为了安全起见，仅支持基础运算。"

    try:
        # 使用 eval 前的二次过滤（实际生产建议使用 simpleeval）
        result = eval(expression, {"__builtins__": None}, {})
        return str(result)
    except Exception as e:
        return f"计算出错: {str(e)}"


# --- 4. 核心类实现 ---

class ReportBuildingAgent:
    def __init__(self):
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.llm = AzureChatOpenAI(
            azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            temperature=0
        )
        self.db = SQLDatabase.from_uri("sqlite:///offer_db.sqlite")
        self.graph = self._build_graph()
        graph_png = self.graph.get_graph().draw_mermaid_png()
        with open("csv_searcher.png", "wb") as f:
            f.write(graph_png)
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)


    def parse_output(self, retrieved_offers, query):
        from langchain_community.vectorstores import FAISS
        """解析retrieve_offers()方法的输出并返回一个数据帧。用FAISS.from_texts(texts, embeddings).similarity_search方法

        参数:
            retrieved_offers (list[str]): 检索到的offer列表。
            query (str): 用于检索offer的查询。

        返回:
            pd.DataFrame: 包含匹配相似度和offer的数据帧。
        """

        # 分割检索到的offer
        top_offers = retrieved_offers.split("#")
        vector_db = FAISS.from_texts(texts=top_offers, embedding=self.embeddings)
        docs_and_scores = vector_db.similarity_search_with_score(query, k=len(top_offers))

        # 构造 DataFrame
        # score 在 FAISS 中通常是 L2 距离，越小越相似；
        # 如果你配置了内积，score 越大越相似。
        df = pd.DataFrame([
            {"距离分数 %": score, "offer": doc.page_content}
            for doc, score in docs_and_scores
        ])

        df.index += 1
        return df


    def _get_chat_prompt_template(self, intent: str) -> ChatPromptTemplate:
        """根据意图动态选择提示词模板"""
        templates = {
            "qa": "你是一个数据分析师。请根据SQL查询结果回答用户关于Offer的问题。结果：{context}",
            "summarization": "你是一个精炼的助手。请总结以下Offer信息，突出重点优惠：{context}",
            "calculation": "你是一个数学专家。请解释计算过程并给出结果：{context}",
            "general": "你是一个友好的助手。请回答用户的问题。"
        }
        sys_msg = templates.get(intent, templates["general"])
        return ChatPromptTemplate.from_messages([
            ("system", sys_msg),
            ("human", "{query}")
        ])

    # --- 节点函数 ---

    def intent_classifier(self, state: AgentState):
        """意图识别节点"""
        structured_llm = self.llm.with_structured_output(UserIntent)
        system_prompt = "分析用户查询并分类：'qa'(查询数据库), 'calculation'(数学计算), 'summarization'(总结信息), 'general'(其他)。"
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{query}")
        ])
        intent = structured_llm.invoke(prompt.format(query=state['query']))
        return {"intent": intent}

    def sql_search_node(self, state: AgentState):
        """数据库检索节点"""
        PROMPT_TEMPLATE = """
                        你会接收到一个查询，你的任务是从`offer_retailer`表中的`OFFER`字段检索相关offer。
                        查询可能是混合大小写的，所以也要搜索大写版本的查询。
                        重要的是，你可能需要使用数据库中其他表的信息，即：`brand_category`, `categories`, `offer_retailer`，来检索正确的offer。
                        不要虚构offer。如果在`offer_retailer`表中找不到offer，返回字符串：`NONE`。
                        如果你能从`offer_retailer`表中检索到offer，用分隔符`#`分隔每个offer。例如，输出应该是这样的：`offer1#offer2#offer3`。
                        如果SQLResult为空，返回`None`。不要生成任何offer。
                        不要返回任何 Markdown 格式，不要以 ``` 开头或结尾。
                        只返回纯文本 SQL 语句。

                        这是查询：`{}`
                        """
        sqlichain_prompt=PROMPT_TEMPLATE.format(state['query'])

        # 这里集成原本的 SQL 生成逻辑
        from langchain_experimental.sql import SQLDatabaseChain
        db_chain = SQLDatabaseChain.from_llm(self.llm, self.db)
        try:
            res = db_chain.run(sqlichain_prompt)
            df=self.parse_output(res, state['query'])
            return {"sql_results": res,"sql_answer":df}
        except Exception as e:
            return {"sql_results": f"查询失败: {str(e)}"}

    def calculator_node(self, state: AgentState):
        """计算器处理节点"""
        # 提取表达式（简单处理，实际可由 LLM 提取）
        expr = state['query']
        result = calculator.invoke(expr)
        return {"tool_output": result}

    def final_generator(self, state: AgentState):
        """最终答案生成节点"""
        intent_type = state['intent'].intent_type
        context = state.get('sql_results') or state.get('tool_output') or ""

        prompt_tmpl = self._get_chat_prompt_template(intent_type)
        chain = prompt_tmpl | self.llm.with_structured_output(AnswerResponse)

        response = chain.invoke({"query": state['query'], "context": context})
        return {"final_answer": response}

    # --- 图构建 ---

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        # 添加节点
        workflow.add_node("classify", self.intent_classifier)
        workflow.add_node("retrieve", self.sql_search_node)
        workflow.add_node("calculate", self.calculator_node)
        workflow.add_node("generate", self.final_generator)

        # 设置入口
        workflow.set_entry_point("classify")

        # 路由逻辑 (Conditional Edges)
        def route_by_intent(state: AgentState):
            it = state['intent'].intent_type
            if it == "calculation": return "calculate"
            if it == "qa" or it == "summarization": return "retrieve"
            return "generate"

        workflow.add_conditional_edges(
            "classify",
            route_by_intent,
            {
                "calculate": "calculate",
                "retrieve": "retrieve",
                "generate": "generate"
            }
        )

        workflow.add_edge("calculate", "generate")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)

        return workflow.compile()


    def run(self, query: str):
        return self.graph.invoke({"query": query})