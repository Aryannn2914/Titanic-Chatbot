# from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from os import getenv
from dotenv import load_dotenv
# from langchain_classic.agents import AgentType
# from langchain_huggingface import HuggingFaceEndpoint
from langchain_ollama import ChatOllama
import pandas as pd
import matplotlib as mt

mt.use("Agg")

load_dotenv()

# llm = ChatOpenAI(
#     model="meta-llama/llama-3.2-3b-instruct:free",
#     base_url="https://openrouter.ai/api/v1",
#     api_key=getenv("OPENROUTER_API_KEY111"),
#     temperature=0,
#     verbose=True,
# )

# llm = HuggingFaceEndpoint(
#     repo_id="meta-llama/Llama-3.2-1B-Instruct",
#     # provider="featherless-ai",
#     huggingfacehub_api_token=getenv("HUGGINGFACEHUB_API_TOKEN_CSV"),
#     temperature=0,
#     task="text-generational",
# )

llm = ChatOllama(model="llama3.1:8b", temperature=0, verbose=True)

original_df = pd.read_csv("data/train.csv")


def agent_reasoning(question: str):
    df = original_df.copy()

    agent = create_pandas_dataframe_agent(
        llm,
        df,
        allow_dangerous_code=True,
        verbose=True,
        agent_executor_kwargs={"handle_parsing_errors": True},
        # agent_type=AgentType.OPENAI_FUNCTIONS,
        prefix="""
    You are a Python data analysis agent.

    You are given a pandas DataFrame named df.

    When using the python tool:
    - Output ONLY raw executable Python code.
    - Do NOT use markdown.
    - Do NOT wrap code in ``` blocks.
    - Do NOT include explanations.
    - Do NOT reload the CSV file.
    - The dataframe is already available as df.
    - Generate plots when necessary using matplotlib.
    - If generating a plot, ensure it is displayed using plt.show().

    After using the python tool:
    - Explain the results clearly in plain English.
    - If a plot was generated, describe what the plot shows.
    - Do NOT include code in the final answer.
    - Do NOT use markdown in the final answer.
        """,
    )
    response = agent.invoke({"input": question})
    return response["output"]


# agent_reasoning(
#     "How many male were there in the ship, Show me the percentage of the male and female, by plotting it"
# )
