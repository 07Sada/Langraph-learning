import streamlit as st
from streamlit_backend import chat_bot
from langchain_core.messages import HumanMessage, AIMessage
import uuid

# Set up the main header for the web application
st.title("Streamlit Basic Chatbot")

# Utility function
def get_session_id()-> str:
    """Create unique session id"""
    # Generate a cryptographically secure random UUID string for tracking threads
    session_id = str(uuid.uuid4())
    return session_id

# Add thread
def add_thread(thread_id):
    # Register the thread ID into the historical tracking list if it isn't already there
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

# Reset chatting / starting new chat
def reset_chat():
    # Generate a fresh thread ID for the new conversation session
    thread_id = get_session_id()
    st.session_state['thread_id'] = thread_id
    # Register the new thread ID in the conversation list
    add_thread(st.session_state['thread_id'])
    # Clear the active message history on the screen
    st.session_state['messages'] = []

# Load the conversation
def load_conversation(thread_id):
    # Retrieve the state history of a specific thread from the LangGraph backend
    state = chat_bot.get_state(config={'configurable': {'thread_id': thread_id}})
    # Safely extract the list of messages; return an empty list if no history exists
    return state.values.get('messages', [])

# maintain message history
# Initialize the UI message list in Streamlit session state if it does not exist
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Initialize the tracking list for all active chat threads if it does not exist
if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []

# Thread id
# Initialize the current session's thread ID if it doesn't exist yet
if 'thread_id' not in st.session_state:
    st.session_state['thread_id']= get_session_id()

# Ensure the current active thread ID is added to the historical tracking list
add_thread(st.session_state['thread_id'])

# **************************************** Sidebar UI *********************************
st.sidebar.title('Langgraph Chatbot')

# Provide a button to trigger the chat reset/new session function
if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header('Conversation History')

# Iterate through past thread IDs in reverse chronological order (newest first)
for thread_id in st.session_state['chat_threads'][::-1]:
    # Render a button for each thread ID; clicking it loads that conversation
    if st.sidebar.button(str(thread_id)):
        # Switch the active session to the selected thread ID
        st.session_state['thread_id'] = thread_id
        # Fetch LangChain message objects from the backend for this thread
        messages = load_conversation(thread_id)

        # Convert LangChain message classes into standard Streamlit UI dictionaries
        temp_messages = []
        for message in messages:
            if isinstance(message, HumanMessage):
                role = 'user'
            else:
                role='assistant'
            temp_messages.append({'role': role, 'content': message.content})
        # Override current message history with the selected thread's messages
        st.session_state['messages'] = temp_messages

# loading the conversation history
# Render all messages from the current conversation history onto the main screen
for message in st.session_state['messages']:
    with st.chat_message(message['role']):
        st.text(message['content'])

# **************************************** Main Input UI ******************************
# Capture input text from the user chat box
user_input = st.chat_input("Type here...")

if user_input:
    # first add the user message to message history
    st.session_state['messages'].append({'role': 'user', 'content': user_input})
    # Immediately render the user's input message to the UI
    with st.chat_message('user'):
        st.text(user_input)

    # Set up the configuration payload required by LangGraph to identify the thread
    config = {'configurable': {'thread_id': st.session_state.thread_id}}

    # Add the assistant response to message history
    with st.chat_message('assistant'):
        # Helper generator function to filter and yield tokens coming from the LLM
        def ai_only_stream():
            # Invoke the LangGraph bot stream with the user input and configuration
            for message_chunk, metadata in chat_bot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config=config,
                stream_mode='messages'
            ):
                # Filter out tool calls or metadata; yield only the actual AI message content
                if isinstance(message_chunk, AIMessage):
                    # yield only assistant tokens
                    yield message_chunk.content

        # Feed the stream generator into Streamlit to print tokens in real time
        ai_message = st.write_stream(ai_only_stream())

    # Commit the full generated AI response to the session state history
    st.session_state['messages'].append({'role': 'assistant', 'content': ai_message})