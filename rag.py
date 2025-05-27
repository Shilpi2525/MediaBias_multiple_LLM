from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState
from typing import  Annotated
import operator
from openai import OpenAI
from langgraph.graph import START, END, StateGraph
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from tavily import TavilyClient
from langchain_community.tools.tavily_search import TavilySearchResults
import base64
import streamlit as st
import os

import prompts as pt


os.environ["LANGSMITH_TRACING"] = st.secrets["LANGSMITH_TRACING"]
os.environ["LANGSMITH_API_KEY"] = st.secrets["LANGSMITH_API_KEY"]
os.environ["LANGSMITH_ENDPOINT"] = st.secrets["LANGSMITH_ENDPOINT"]
os.environ["LANGSMITH_PROJECT"] = st.secrets["LANGSMITH_PROJECT"]
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]

MAX_RESULTS = 5

# set the openai model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# create client
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# create travily client
tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

# tavily search
tavily_search = TavilySearchResults(max_results = MAX_RESULTS)


# this is be default has the messages and add_messages reducers
class BotState(MessagesState):
    context: list[str]
    answers: str


def search_web(state: BotState):

    hot_topic = state["messages"][-1].content

    # Search
    search_docs = tavily_client.search(pt.SEARCH_QUERY.format(topic = hot_topic))
    

     # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_docs['results']
        ]
    )


    return {"context": [formatted_search_docs]}


def bias_identifier(state: BotState, config: RunnableConfig):
    topic_docs = state["context"]

    # create the system message
    system_message = [SystemMessage(pt.BIAS_QUERY.format(documents = topic_docs))]
    print(system_message)

    # invoke llm
    invoke_llm = llm.invoke(system_message, config)

    return {"answers": invoke_llm.content}


# add nodes and edges
helper_builder = StateGraph(BotState)
helper_builder.add_node("web_search", search_web)
helper_builder.add_node("bias_analyser", bias_identifier)

# build graph
helper_builder.add_edge(START, "web_search")
helper_builder.add_edge("web_search", "bias_analyser")
helper_builder.add_edge("bias_analyser", END)

# compile the graph
helper_graph = helper_builder.compile()


async def graph_streamer(user_query: str):
    node_to_stream = 'bias_analyser'
    model_config = {"configurable": {"thread_id": "1"}}

    input_message = HumanMessage(content = user_query)

    async for event in helper_graph.astream_events({"messages": [input_message]}, model_config, version="v2"):
        # Get chat model tokens from a particular node
        #print(event)
        if event["event"] == "on_chat_model_stream":
            if event['metadata'].get('langgraph_node','') == node_to_stream:
                data = event["data"]
                yield data["chunk"].content
