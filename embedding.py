from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from os import getenv
from dotenv import load_dotenv
# from langchain_classic.agents import AgentType
# from langchain_huggingface import HuggingFaceEndpoint
# from langchain_ollama import ChatOllama
import pandas as pd
# import matplotlib as mt
import streamlit as st
import matplotlib.pyplot as plt


load_dotenv()

openai_key = st.secrets.get("OPENAI_API_KEY", getenv("OPENAI_API_KEY"))

# If using openrouter uncomment this variable
llm = ChatOpenAI(
    model="meta-llama/llama-3.2-3b-instruct:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=openai_key,
    temperature=0,
    verbose=True,
)

# If using huggingface uncomment this variable
# llm = HuggingFaceEndpoint(
#     repo_id="meta-llama/Llama-3.2-1B-Instruct",
#     # provider="featherless-ai",
#     huggingfacehub_api_token=getenv("HUGGINGFACEHUB_API_TOKEN_CSV"),
#     temperature=0,
#     task="text-generational",
# )

# If using ollama uncomment this variable
# llm = ChatOllama(model="llama3.1:8b", temperature=0, verbose=True)

original_df = pd.read_csv("data/train.csv")

df = original_df.copy()

agent = create_pandas_dataframe_agent(
    llm,
    df,
    allow_dangerous_code=True,
    verbose=True,
    # max_iterations=4,
    agent_executor_kwargs={"handle_parsing_errors": True},
    prefix="""
You are a Python data analysis agent with access to a pandas DataFrame named df.

When calling the python tool:
- Output ONLY valid Python code.
- No markdown.
- No explanations.
- Do not reload data.

After the tool runs:
- Provide a clear plain-English answer.
- If a plot was created, describe it.
- Do not include code in the final answer.
""",
)

# Streamlit
st.title("Chatbot")

question = st.chat_input("Ask question")

if question:
    plt.close("all")
    st.write(question)
    response = agent.invoke({"input": question})
    st.write(response["output"])
    if plt.get_fignums():
        fig = plt.gcf()
        st.pyplot(fig)
