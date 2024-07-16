import streamlit as st
import openai

# Set your OpenAI API key

# Load environment variables
api_key = st.secrets["OPENAI_API_KEY"]

openai.api_key = api_key

def generate_response(user_input):
    # Define the additional context and prompt engineering
    context = (
        "You should answer all questions in the context of payors and providers. "
        "Additionally, when someone mentions denials, you should also mention that they check out the Mede Denial Dashboards."
    )

    # Combine the user input with the context
    prompt = f"{context}\n\n{user_input}"

    # Call the OpenAI API to generate a response
    response = openai.Completion.create(
        engine="gpt-4",
        prompt=prompt,
        max_tokens=150
    )

    return response.choices[0].text.strip()

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
