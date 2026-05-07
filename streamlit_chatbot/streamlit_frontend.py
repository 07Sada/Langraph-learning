import streamlit as st
from streamlit_backend import chat_bot
from langchain_core.messages import HumanMessage

THREAD_ID = 1
CONFIG = {'configurable': {'thread_id': THREAD_ID}}

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []


# Loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

# Taking user input
user_input = st.chat_input("Type Here")

if user_input:

    # add message to meesage_history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})

    # printing the user message in chat ui
    with st.chat_message('user'):
        st.text(user_input)

    # collect the response from LLM
    llm_response = chat_bot.invoke({'messages': [HumanMessage(content=user_input)]}, config=CONFIG)
    ai_message = llm_response['messages'][-1].content

    # add the message to message_history
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
    
    # print the llm response
    with st.chat_message('assistant'):
        st.text(ai_message)    
        
        
