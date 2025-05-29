from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState
from typing import Annotated
from langgraph.graph import START, END, StateGraph
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from tavily import TavilyClient
from langchain_community.tools.tavily_search import TavilySearchResults
import base64
import streamlit as st
import os

import prompts as pt

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_deepseek import ChatDeepSeek
from langchain_anthropic import ChatAnthropic

# Environment setup
os.environ["LANGSMITH_TRACING"] = st.secrets["LANGSMITH_TRACING"]
os.environ["LANGSMITH_API_KEY"] = st.secrets["LANGSMITH_API_KEY"]
os.environ["LANGSMITH_ENDPOINT"] = st.secrets["LANGSMITH_ENDPOINT"]
os.environ["LANGSMITH_PROJECT"] = st.secrets["LANGSMITH_PROJECT"]
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
os.environ["DEEPSEEK_API_KEY"] = st.secrets["DEEPSEEK_API_KEY"]
#os.environ["hf_token"] = st.secrets["hf_token"]

# Constants
MAX_RESULTS = 5

# LLM options
llm_options = {
    "OpenAI-GPT4o": ChatOpenAI(model="GPT-4o-mini", temperature=0),
    "Google-Gemini": ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0),
    "Anthropic-Claude": ChatAnthropic(model="claude-3-haiku-20240307", temperature=0),
    "DeepSeek": ChatDeepSeek(model="deepseek-chat", temperature=0),
}

# Tavily setup
tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
tavily_search = TavilySearchResults(max_results=MAX_RESULTS)

# Bot state
type BotState(MessagesState):
    context: list[str]
    answers: str

# Web search node
def search_web(state: BotState):
    hot_topic = state["messages"][-1].content
    search_docs = tavily_client.search(pt.SEARCH_QUERY.format(topic=hot_topic))
    formatted_docs = "\n\n---\n\n".join([
        f'<Document href="{doc["url"]}"/>{doc["content"]}</Document>'
        for doc in search_docs['results']
    ])
    return {"context": [formatted_docs]}

# First LLM: Bias presence detector
def bias_detector(state: BotState, config: RunnableConfig, llm):
    topic_docs = state["context"]
    prompt = pt.BIAS_DETECTION_PROMPT.format(documents=topic_docs)
    system_message = [SystemMessage(prompt)]
    response = llm.invoke(system_message, config)
    return {"answers": response.content}

# Second LLM: Bias type classifier
def bias_type_classifier(state: BotState, config: RunnableConfig, sub_llm):
    biased_output = state["answers"]
    prompt = pt.BIAS_TYPE_PROMPT.format(biased_documents=biased_output)
    system_message = [SystemMessage(prompt)]
    response = sub_llm.invoke(system_message, config)
    return {"answers": response.content}

# Graph constructor
def build_graph(primary_model_name: str, secondary_model_name: str):
    llm = llm_options[primary_model_name]
    sub_llm = llm_options[secondary_model_name]

    builder = StateGraph(BotState)
    builder.add_node("web_search", search_web)
    builder.add_node("bias_detector", lambda s, c: bias_detector(s, c, llm))
    builder.add_node("bias_classifier", lambda s, c: bias_type_classifier(s, c, sub_llm))
    builder.set_entry_point("web_search")
    builder.set_finish_point("bias_classifier")

    builder.add_edge("web_search", "bias_detector")
    builder.add_edge("bias_detector", "bias_classifier")

    return builder.compile()

# Async streamer
def graph_streamer_factory(primary_model: str, secondary_model: str):
    graph = build_graph(primary_model, secondary_model)

    async def graph_streamer(user_query: str):
        node_to_stream = 'bias_classifier'
        model_config = {"configurable": {"thread_id": "1"}}
        input_message = HumanMessage(content=user_query)

        async for event in graph.astream_events({"messages": [input_message]}, model_config, version="v2"):
            if event["event"] == "on_chat_model_stream":
                if event['metadata'].get('langgraph_node', '') == node_to_stream:
                    data = event["data"]
                    yield data["chunk"].content

    return graph_streamer
