import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from openai import OpenAI

# Generate synthetic data with noise
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', periods=24, freq='M')
denials = np.cumsum(np.random.normal(loc=5, scale=2, size=len(dates))) + np.random.normal(scale=5, size=len(dates))
cash_collections = np.cumsum(np.random.normal(loc=-5, scale=2, size=len(dates))) + np.random.normal(scale=5, size=len(dates))

# Convert to percentages
denials = (denials - denials.min()) / (denials.max() - denials.min()) * 100
cash_collections = (cash_collections - cash_collections.min()) / (cash_collections.max() - cash_collections.min()) * 100

# Create a DataFrame
data = pd.DataFrame({'Date': dates, 'Denials (%)': denials, 'Cash Collections (%)': cash_collections})

# Plotting with Plotly
fig = go.Figure()

# Add Denials trace
fig.add_trace(go.Scatter(x=data['Date'], y=data['Denials (%)'], mode='lines+markers', name='Denials (%)', line=dict(color='red')))

# Add Cash Collections trace
fig.add_trace(go.Scatter(x=data['Date'], y=data['Cash Collections (%)'], mode='lines+markers', name='Cash Collections (%)', line=dict(color='blue')))

# Update layout
fig.update_layout(
    title='Trends Over Time',
    xaxis_title='Date',
    yaxis_title='Percentage (%)',
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
    data_summary = (
        f"The data shows the following trends: Denials have varied between {denials.min():.2f}% and {denials.max():.2f}% over the period, "
        f"showing a general trend of {'increase' if denials[-1] > denials[0] else 'decrease'}. Cash collections have varied between "
        f"{cash_collections.min():.2f}% and {cash_collections.max():.2f}%, showing a general trend of {'increase' if cash_collections[-1] > cash_collections[0] else 'decrease'}."
    )
    
    messages = [
        {"role": "system", "content": "You support revenue cycle management customers in making data-based decisions. Before answering the prompt, provide a written summary of the data based on the summary data provided. Then give specific tips and limit the tips to five."},
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": data_summary}
    ]

    # Call the OpenAI API to generate a response
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.7,
        max_tokens=300,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    return response.choices[0].message.content.strip()

# Streamlit application
st.title("Mede Decision Support Assistant")
user_input = st.text_input("Ask me how I can help improve RCM")
if user_input:
    response = generate_response(user_input)
    st.write(response)
