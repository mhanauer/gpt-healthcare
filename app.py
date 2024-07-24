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

# Normalize data to specified ranges
def normalize(data, start_value, lower, upper):
    normalized_data = lower + (upper - lower) * (data - data.min()) / (data.max() - data.min())
    normalized_data[0] = start_value  # Set the start value
    return np.round(normalized_data)

denials = normalize(denials, 40, 40, 70)
cash_collections = normalize(cash_collections, 50, 40, 70)

# Create a DataFrame
data = pd.DataFrame({'Date': dates, 'Denials (%)': denials, 'Cash Collections (%)': cash_collections})

# Find dates of min and max values
denials_min_date = data['Date'][data['Denials (%)'].idxmin()].strftime('%Y-%m-%d')
denials_max_date = data['Date'][data['Denials (%)'].idxmax()].strftime('%Y-%m-%d')
cash_collections_min_date = data['Date'][data['Cash Collections (%)'].idxmin()].strftime('%Y-%m-%d')
cash_collections_max_date = data['Date'][data['Cash Collections (%)'].idxmax()].strftime('%Y-%m-%d')

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

# Generate a summary of the data
data_summary = (
    f"The data shows the following trends: Denials have varied between {denials.min():.0f}% and {denials.max():.0f}% over the period, "
    f"showing a general trend of {'increase' if denials[-1] > denials[0] else 'decrease'}. The minimum denial percentage occurred on {denials_min_date}, and the maximum denial percentage occurred on {denials_max_date}. "
    f"Cash collections have varied between {cash_collections.min():.0f}% and {cash_collections.max():.0f}%, showing a general trend of {'increase' if cash_collections[-1] > cash_collections[0] else 'decrease'}. "
    f"The minimum cash collections percentage occurred on {cash_collections_min_date}, and the maximum cash collections percentage occurred on {cash_collections_max_date}."
)

st.write("**Data Summary:**")
st.write(data_summary)

# Load environment variables
api_key = st.secrets["OPENAI_API_KEY"]

# Initialize the OpenAI client
client = OpenAI(api_key=api_key)

def generate_response(user_input):
    messages = [
        {"role": "system", "content": "You support revenue cycle management customers in making data-based decisions. Before answering the prompt, provide a written summary of the data based on the summary data provided. Then give specific tips and limit the tips to five."},
        {"role": "user", "content": user_input}
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
