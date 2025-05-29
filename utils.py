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
def setup_environment():
    if "secrets" in dir(st):
        os.environ["LANGSMITH_TRACING"] = st.secrets.get("LANGSMITH_TRACING", "")
        os.environ["LANGSMITH_API_KEY"] = st.secrets.get("LANGSMITH_API_KEY", "")
        os.environ["LANGSMITH_ENDPOINT"] = st.secrets.get("LANGSMITH_ENDPOINT", "")
        os.environ["LANGSMITH_PROJECT"] = st.secrets.get("LANGSMITH_PROJECT", "")
        os.environ["OPENAI_API_KEY"] = st.secrets.get("OPENAI_API_KEY", "")
        os.environ["TAVILY_API_KEY"] = st.secrets.get("TAVILY_API_KEY", "")
        os.environ["GOOGLE_API_KEY"] = st.secrets.get("GOOGLE_API_KEY", "")
        os.environ["ANTHROPIC_API_KEY"] = st.secrets.get("ANTHROPIC_API_KEY", "")
        os.environ["DEEPSEEK_API_KEY"] = st.secrets.get("DEEPSEEK_API_KEY", "")

# Constants
MAX_RESULTS = 5

# LLM options
def get_llm_options():
    try:
        return {
            "OpenAI-GPT4o": ChatOpenAI(model="gpt-4o-mini", temperature=0),
            "Google-Gemini": ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0),
            "Anthropic-Claude": ChatAnthropic(model="claude-3-haiku-20240307", temperature=0),
            "DeepSeek": ChatDeepSeek(model="deepseek-chat", temperature=0),
        }
    except Exception as e:
        st.error(f"Error initializing LLM options: {str(e)}")
        # Fallback to just OpenAI if there are issues
        return {"OpenAI-GPT4o": ChatOpenAI(model="gpt-4o-mini", temperature=0)}

# Tavily setup
def get_tavily_client():
    return TavilyClient(api_key=os.environ.get("TAVILY_API_KEY", ""))

# Bot state
class BotState(MessagesState):
    context: list[str]
    answers: str

# Web search node
def search_web(state: BotState):
    hot_topic = state["messages"][-1].content
    tavily_client = get_tavily_client()
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
    try:
        response = llm.invoke(system_message, config)
        return {"answers": response.content}
    except Exception as e:
        return {"answers": f"Error in bias detection: {str(e)}. Please check API keys and dependencies."}

# Second LLM: Bias type classifier
def bias_type_classifier(state: BotState, config: RunnableConfig, sub_llm):
    biased_output = state["answers"]
    prompt = pt.BIAS_TYPE_PROMPT.format(biased_documents=biased_output)
    system_message = [SystemMessage(prompt)]
    try:
        response = sub_llm.invoke(system_message, config)
        return {"answers": response.content}
    except Exception as e:
        return {"answers": f"Error in bias classification: {str(e)}. Please check API keys and dependencies."}

# Graph constructor
def build_graph(primary_model_name: str, secondary_model_name: str):
    llm_options = get_llm_options()
    
    # Default to OpenAI if the selected models aren't available
    llm = llm_options.get(primary_model_name, llm_options.get("OpenAI-GPT4o"))
    sub_llm = llm_options.get(secondary_model_name, llm_options.get("OpenAI-GPT4o"))

    builder = StateGraph(BotState)
    builder.add_node("web_search", search_web)
    builder.add_node("bias_detector", lambda s, c: bias_detector(s, c, llm))
    builder.add_node("bias_classifier", lambda s, c: bias_type_classifier(s, c, sub_llm))
    
    # Set entry and finish points
    builder.add_edge(START, "web_search")
    builder.add_edge("web_search", "bias_detector")
    builder.add_edge("bias_detector", "bias_classifier")
    builder.add_edge("bias_classifier", END)

    return builder.compile()

# Async streamer
def graph_streamer_factory(primary_model: str, secondary_model: str):
    # Setup environment variables
    setup_environment()
    
    try:
        graph = build_graph(primary_model, secondary_model)

        async def graph_streamer(user_query: str):
            node_to_stream = 'bias_classifier'
            model_config = {"configurable": {"thread_id": "1"}}
            input_message = HumanMessage(content=user_query)
            
            # Initial message
            yield f"Analyzing bias for topic: '{user_query}'\n"
            yield f"Using {primary_model} for detection and {secondary_model} for classification...\n\n"

            try:
                async for event in graph.astream_events({"messages": [input_message]}, model_config, version="v2"):
                    if event["event"] == "on_chat_model_stream":
                        if event['metadata'].get('langgraph_node', '') == node_to_stream:
                            data = event["data"]
                            yield data["chunk"].content
            except Exception as e:
                yield f"\nError during analysis: {str(e)}\n"
                yield "Please check that you have the appropriate API keys set up for the selected models."

        return graph_streamer
    except Exception as e:
        # Return a fallback streamer function if graph creation fails
        async def fallback_streamer(user_query: str):
            yield f"Error initializing the analysis graph: {str(e)}\n"
            yield "Please check that you have installed all required dependencies and set up the API keys correctly."
            yield "\nRequired packages:\n"
            yield "- langchain-openai (for OpenAI)\n"
            yield "- google-generativeai and langchain-google-genai (for Gemini)\n"
            yield "- anthropic and langchain-anthropic (for Claude)\n"
            yield "- langchain-deepseek (for DeepSeek)\n"
        
        return fallback_streamer
