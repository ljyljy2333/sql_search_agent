import streamlit as st
import pandas as pd
from llm import ReportBuildingAgent
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="SMART OFFER REPORT ASSISTANT", layout="wide")
st.title("SMART_OFFER_REPORT_ASSISTANT🤖")
st.markdown("Support natural language database search, mathematical calculations, and information summarization。")

# 初始化 Agent（使用缓存避免重复加载）
if "agent" not in st.session_state:
    st.session_state.agent = ReportBuildingAgent()

# 侧边栏展示会话状态
with st.sidebar:
    st.header("SYSTEM_STATUS")
    if st.button("ClearHistory"):
        st.rerun()

# 搜索表单
query = st.text_input("Please enter your question (e.g. 'calculate 129*0.85', 'Retrieve all offers containing 'KFC''):")

if query:
    with st.spinner("THINKING..."):
        try:
            # 运行 LangGraph
            result = st.session_state.agent.run(query)

            intent = result['intent']
            answer = result['final_answer']


            # 展示意图识别结果
            st.info(f"IDENTIFY_INTENT: **{intent.intent_type}** (CONFIDENCE: {intent.confidence:.2f})")

            # 展示最终答案
            st.subheader("ANALYZE_THE_RESULTS")
            st.write(answer.content)

            sql_answer = result.get("sql_answer")
            if isinstance(sql_answer, pd.DataFrame) and not sql_answer.empty:
                st.table(sql_answer)


            # 展示元数据
            st.caption(f"ResponseTime: {answer.timestamp} | AnswerReliability: {answer.confidence:.2f}")

        except Exception as e:
            st.error(f"OperationError: {str(e)}")

# 示例展示
with st.expander("Example Queries"):
    st.write("- **Search**: Retrieve all offers containing 'KFC'")
    st.write("- **Math**: (500 - 120) × 0.9")
    st.write("- **Summary**: Provide a summary of the current offer data")
