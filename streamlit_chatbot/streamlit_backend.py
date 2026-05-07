from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# loading the api key
load_dotenv()

MODEL_NAME = 'gpt-4o-mini'
llm_model = ChatOpenAI(model=MODEL_NAME)

THREAD_ID = 1

# Define the state 
class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# Define the graph
graph = StateGraph(ChatState)

# Define the chatbot node
def chat_node(state: ChatState):
    
    # Fetch the messages from the state
    messages = state['messages']

    # Collect the response from llm model
    response = llm_model.invoke(messages)

    # update the state
    return {'messages': [response]}

# Create a memory checkpoint
memory_checkpoint = MemorySaver()

# Add nodes to the graph
graph.add_node('chat_node', chat_node)

# Define the edges
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

# compile the graph and add the checkpoint 
chat_bot = graph.compile(checkpointer=memory_checkpoint)

