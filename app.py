from openai import OpenAI
import streamlit as st


import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Generate synthetic data
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', periods=24, freq='M')
denials = np.cumsum(np.random.normal(loc=5, scale=2, size=len(dates)))
cash_collections = np.cumsum(np.random.normal(loc=-5, scale=2, size=len(dates)))

# Create a DataFrame
data = pd.DataFrame({'Date': dates, 'Denials': denials, 'Cash Collections': cash_collections})

# Plotting with Plotly
fig = go.Figure()

# Add Denials trace
fig.add_trace(go.Scatter(x=data['Date'], y=data['Denials'], mode='lines+markers', name='Denials', line=dict(color='red')))

# Add Cash Collections trace
fig.add_trace(go.Scatter(x=data['Date'], y=data['Cash Collections'], mode='lines+markers', name='Cash Collections', line=dict(color='blue')))

# Update layout
fig.update_layout(
    title='Trends Over Time',
    xaxis_title='Date',
    yaxis_title='Values',
    legend_title='Metrics',
    template='plotly_white'
)

# Streamlit application
st.title("Trends in KPIs over time")
st.plotly_chart(fig)


# Load environment variables
api_key = st.secrets["OPENAI_API_KEY"]

# Initialize the OpenAI client
client = OpenAI(api_key=api_key)

def generate_response(user_input):
    # Define the additional context and prompt engineering
    messages = [
        {"role": "system", "content": "You support revenue cycle management customers in making data based decisions."},
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
st.title("Mede Decision Support Assitant")
user_input = st.text_input("Ask me how I can help improve RCM")
if user_input:
    response = generate_response(user_input)
    st.write(response)