# sql_search_agent
数据库搜索：用户用自然语言查询数据（如“Retrieve all offers containing 'KFC'”），Agent 自动生成 SQL 并返回表格结果。  数学计算：识别表达式并计算结果（如“calculate 129*0.85”），方便快速得出折扣或统计值。  信息总结：对数据生成自然语言报告或摘要（如“Provide a summary of the current offer data”），便于分析和汇报。  同时，Agent 会识别用户意图（Search / Math / Summary）并给出置信度，还会提供回答时间和可靠性指标。通过 st.session_state 缓存实例，可避免重复加载，提高交互效率，是一个集查询、计算、分析于一体的智能助手。
