import streamlit as st
import openai

# Load environment variables
api_key = st.secrets["OPENAI_API_KEY"]

# Set your OpenAI API key
openai.api_key = api_key

def generate_response(user_input):
    # Define the additional context and prompt engineering
    system_content = (
        "You should answer all questions in the context of payors and providers. "
        "Additionally, when someone mentions denials, you should also mention that they check out the Mede Denial Dashboards."
    )

    # Combine the user input with the context
    messages = [
        {
            "role": "system",
            "content": system_content
        },
        {
            "role": "user",
            "content": user_input
        }
    ]

    # Call the OpenAI API to generate a response
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=1,
        max_tokens=2425,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    return response['choices'][0]['message']['content'].strip()

# Streamlit application
st.title("Healthcare Support Chatbot")

user_input = st.text_area("Ask a question about payors and providers:")

if st.button("Get Response"):
    if user_input:
        response = generate_response(user_input)
        st.write("### Response")
        st.write(response)
    else:
        st.write("Please enter a question to get a response.")
