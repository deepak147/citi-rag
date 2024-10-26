import streamlit as st
from streamlit_chat import message
from backend.retrieve import retrieve

# Set page config
st.set_page_config(page_title="Citi Bank Chatbot", page_icon=":bank:", layout="wide")

st.markdown("""
<style>
    .stApp {
        background-color: #FFFFFF;
        color: #003B70;
    }
    .stTextInput > div > div > input {
        background-color: #FFFFFF;
        color: #003B70;
        border: 1px solid #003B70;
        border-radius: 0.5rem;
        padding: 0.5rem;
    }
    .stButton > button {
        background-color: #004685;
        color: #FFFFFF;
        border-radius: 0.5rem;
    }
    .stSidebar {
        background-color: #E9EAEC;
    }
    .chat-message {
        font-family: Arial;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        color: #003B70;
    }
    .chat-message.user {
        background-color: #E9EAEC;
    }
    .chat-message.bot {
        background-color: #F0F8FF;
    }
    .chat-message .avatar {
        width: 20%;
    }
    .chat-message .message {
        width: 80%;
        padding: 0 1.5rem;
    }
    .chat-input {
        position: fixed;
        bottom: 0;
        background-color: white;
        width: 100%;
        padding: 1rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.image("https://www.citi.com/CBOL/IA/Angular/assets/citiredesign.svg", width=200)
    st.title("Citi Bank Assistant")

# Main content
st.header("Citi Bank Interactive Chatbot")

# Display chat messages
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input
with st.container():
    st.markdown('<div class="chat-input">', unsafe_allow_html=True)
    user_input = st.text_input("You:", key="user_input", placeholder="Ask me anything about Citi Bank services...")
    send_button = st.button("Send")
    st.markdown('</div>', unsafe_allow_html=True)

    if send_button and user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.spinner("Generating response..."):
            # Get chatbot response
            response = retrieve(query=user_input)
            bot_response = response['answer']

        # Add bot response to chat history
        st.session_state.messages.append({"role": "assistant", "content": bot_response})

        # Rerun the app to display the new messages
        st.rerun()

# Add some space at the bottom to prevent overlap with input box
st.markdown("<br><br><br>", unsafe_allow_html=True)
