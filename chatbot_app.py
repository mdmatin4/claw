import streamlit as st
import time
import random

st.set_page_config(page_title="SnarkyBot", page_icon="🤖")

st.title("🤖 SnarkyBot Interface")
st.markdown("A simple conversational interface built with Streamlit.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am SnarkyBot. Say something, and I'll reply."}
    ]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Snarky responses to pick from
snarky_responses = [
    "That's exactly what I expected you to say.",
    "Fascinating. Tell me more about things I don't care about.",
    "I'm a bot, not a miracle worker.",
    "Did you really just type that?",
    "I'll pretend I didn't see that.",
    "Wow. Just... wow.",
    "I've computed 14 million possible futures, and you say this in all of them."
]

# React to user input
if prompt := st.chat_input("Type your message here..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Decide what the bot will say
        if "?" in prompt:
            assistant_response = "I'm not answering that. Google it!"
        else:
            assistant_response = f"You said: '{prompt}'. {random.choice(snarky_responses)}"
            
        # Simulate a stream of response with a slight delay (typing effect)
        for chunk in assistant_response.split():
            full_response += chunk + " "
            time.sleep(0.08)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
            
        message_placeholder.markdown(full_response)
        
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
