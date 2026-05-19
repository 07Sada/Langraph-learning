from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, List, Annotated
from langchain_core.messages import HumanMessage, BaseMessage

# load the environment variables
load_dotenv()

MODEL_NAME = 'gpt-4o-mini'
llm_model = ChatOpenAI(model=MODEL_NAME)

# Define the state
class ChatState(TypedDict):
    ''''
    store the messages
    '''
    messages: Annotated[List[BaseMessage], add_messages]

def chat_node(state: ChatState):
    # retive the message
    messages = state['messages']

    # make the llm call and collect the response
    response = llm_model.invoke(messages)
    
    # return the response
    return {'messages': [response]}

# Create a memory checkpoint
memory_checkpoint = MemorySaver()

# Define the graph
graph = StateGraph(ChatState)

# define the node
graph.add_node('chat_node', chat_node)

# Define the edges
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

# compile the graph
chat_bot = graph.compile(checkpointer=memory_checkpoint)

# for chunk, metadata in chat_bot.stream(
#     input={'messages': [HumanMessage(content='Write 100 word blog on India')]},
#     config={'configurable': {'thread_id': '1'}},
#     stream_mode="messages"):

#     if chunk.content:
#         print(chunk.content, end="", flush=True)