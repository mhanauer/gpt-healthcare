from openai import OpenAI
import streamlit as st

# Load environment variables
api_key = st.secrets["OPENAI_API_KEY"]

# Initialize the OpenAI client
client = OpenAI(api_key=api_key)

def generate_response(user_input):
    # Define the additional context and prompt engineering
    messages = [
        {"role": "system", "content": "You are a helpful healthcare support chatbot."},
        {"role": "user", "content": user_input}
    ]

    # Call the OpenAI API to generate a response
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=1,
        max_tokens=2425,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    return response.choices[0].message.content.strip()

# Streamlit application
st.title("Healthcare Support Chatbot")
user_input = st.text_input("Ask me anything about healthcare:")
if user_input:
    response = generate_response(user_input)
    st.write(response)