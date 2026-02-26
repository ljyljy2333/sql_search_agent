- Here is the professional English version of your **README.md**. It’s clean, precise, and highlights exactly what a reviewer or collaborator would look for.

  ------

  # 🤖 Smart Offer Agent (LangGraph)

  A sophisticated report-building agent powered by **LangGraph**, featuring automated intent classification, secure database retrieval, mathematical reasoning, and data summarization.

  ## 🚀 Key Features

  - **Graph-Based Workflow**: Built on **LangGraph** state machines to manage a non-linear `Classify -> Route -> Execute -> Generate` pipeline.
  - **Strict Data Validation**: Utilizes **Pydantic** to enforce runtime constraints on confidence scores ($0.0$ to $1.0$) and intent categories.
  - **Secure Calculation**: Implements a custom `@tool` calculator with a regex-based whitelist to prevent code injection.
  - **Hybrid Retrieval**: Combines automated **SQL generation** with **FAISS vector re-ranking** for maximum precision.

  ------

  ## 🏗️ Architecture Workflow

  1. **classify**: Analyzes user input to determine intent (`qa`, `calculation`, `summarization`, or `general`).
  2. **retrieve/calculate**: Dynamically routes the request to either the SQLite database or the secure calculation tool based on the classified intent.
  3. **generate**: Synthesizes the gathered `context` to produce a structured, validated response.

  ------

  ## 🛠️ Quick Start

  ### 1. Installation

  Bash

  ```
  pip install langchain langchain-openai langgraph pydantic pandas streamlit faiss-cpu python-dotenv
  ```

  ### 2. Configuration

  Create a `.env` file in the root directory and add your Azure OpenAI credentials:

  代码段

  ```
  AZURE_OPENAI_API_KEY=your_key
  AZURE_OPENAI_ENDPOINT=your_endpoint
  AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=your_deployment
  AZURE_OPENAI_API_VERSION=2024-02-15-preview
  ```

  ### 3. Run the App

  Bash

  ```
  streamlit run csv_search.py
  ```

  ------

  ## 💬 Usage Examples

  | **Scenario** | **User Input**                         | **Identified Intent** |
  | ------------ | -------------------------------------- | --------------------- |
  | **Search**   | "Retrieve all offers containing 'KFC'" | `qa`                  |
  | **Math**     | "What is (100 + 50) * 0.8?"            | `calculation`         |
  | **Summary**  | "Summarize the current offer data"     | `summarization`       |

  ------

  ## 📁 Project Structure

  - `llm.py`: Core logic including the Graph definition, Pydantic schemas, and Agent nodes.
  - `csv_search.py`: Streamlit-based user interface and session management.
  - `offer_db.sqlite`: Local SQLite database containing offer and retailer data.