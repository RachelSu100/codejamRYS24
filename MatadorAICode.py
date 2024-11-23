# requirements
# Make sure to install the necessary packages before running the script
# pip install pandas flask requests ollama streamlit streamlit-chat langchain faiss-cpu transformers ctransformers langchain-community

# imports
import pandas as pd
from flask import Flask, request, jsonify
import json
import requests
import re
import tempfile
import streamlit as st
from langchain.document_loaders import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

print("Starting imports...")
print("Successfully imported!")

# Load LLM (dummy function for now)
def load_llm():
    print("Loading LLM - placeholder function")
    return None

# Extract preferences from user's message
def extract_preferences(message, carsDF):
    preferences = {}

    if "new" in message.lower() or "used" in message.lower():
        preferences['car_type'] = 'New' if 'new' in message.lower() else 'Used'

    if 'year' in message.lower():
        year = [int(word) for word in message.split() if word.isdigit() and len(word) == 4]
        if year:
            preferences['year'] = year[0]

    if any(make in message.lower() for make in carsDF['Make'].str.lower().unique()):
        preferences['make'] = next(make.title() for make in carsDF['Make'].str.lower().unique() if make in message.lower())

    # Extract price range (e.g., "under $20,000", "between $20,000 and $30,000")
    price_under = re.search(r'under\s*\$?(\d{1,3}(?:,\d{3})*)', message, re.IGNORECASE)
    price_between = re.search(r'between\s*\$?(\d{1,3}(?:,\d{3})*)\s*and\s*\$?(\d{1,3}(?:,\d{3})*)', message, re.IGNORECASE)

    if price_under:
        max_price = int(price_under.group(1).replace(',', ''))
        preferences['max_price'] = max_price
    elif price_between:
        min_price = int(price_between.group(1).replace(',', ''))
        max_price = int(price_between.group(2).replace(',', ''))
        preferences['min_price'] = min_price
        preferences['max_price'] = max_price

    # Extract mileage
    if "low mileage" in message.lower():
        preferences['max_miles'] = 50000  # Assume "low mileage" means under 50,000 miles
    mileage = re.search(r'(\d{1,3}(?:,\d{3})*)\s*miles', message, re.IGNORECASE)
    if mileage:
        preferences['max_miles'] = int(mileage.group(1).replace(',', ''))

    # Extract exterior color preference
    colors = carsDF['Ext_Color_Generic'].dropna().apply(str).str.lower().unique()
    for color in colors:
        if color in message.lower():
            preferences['exterior_color'] = color.title()
            break

    # Extract drivetrain preference
    drivetrains = ['AWD', 'FWD', 'RWD', '4WD']
    for drivetrain in drivetrains:
        if drivetrain.lower() in message.lower():
            preferences['drivetrain'] = drivetrain.upper()
            break

    return preferences

# Function to get matching cars based on preferences
def get_matching_cars(preferences, carsDF):
    filtered_cars = carsDF

    # Apply each preference as a filter
    if 'car_type' in preferences:
        filtered_cars = filtered_cars[filtered_cars['Type'].str.lower() == preferences['car_type'].lower()]

    if 'year' in preferences:
        filtered_cars = filtered_cars[filtered_cars['Year'] == preferences['year']]

    if 'make' in preferences:
        filtered_cars = filtered_cars[filtered_cars['Make'].str.upper() == preferences['make'].upper()]

    if 'min_price' in preferences and 'max_price' in preferences:
        filtered_cars = filtered_cars[
            (filtered_cars['SellingPrice'] >= preferences['min_price']) & 
            (filtered_cars['SellingPrice'] <= preferences['max_price'])
        ]
    elif 'max_price' in preferences:
        filtered_cars = filtered_cars[filtered_cars['SellingPrice'] <= preferences['max_price']]

    if 'max_miles' in preferences:
        filtered_cars = filtered_cars[filtered_cars['Miles'] <= preferences['max_miles']]

    if 'exterior_color' in preferences:
        filtered_cars = filtered_cars[filtered_cars['Ext_Color_Generic'].str.lower() == preferences['exterior_color'].lower()]

    if 'drivetrain' in preferences:
        filtered_cars = filtered_cars[filtered_cars['Drivetrain'].str.upper() == preferences['drivetrain'].upper()]

    # Return the top 3 matching cars
    if not filtered_cars.empty:
        return filtered_cars[['Make', 'Model', 'Year', 'VIN', 'SellingPrice']].head(3)
    else:
        return None

# Streamlit UI
st.title("Matador Car Match Chatbot")

# Sidebar for CSV file upload
csv_data = st.sidebar.file_uploader("Upload vehicles.csv", type="csv")

if csv_data is not None:
    # Handle uploaded CSV file with temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
        tmp_file.write(csv_data.getvalue())
        tmp_file_path = tmp_file.name

    # Load CSV data using pandas directly
    carsDF = pd.read_csv(tmp_file_path)

    # Load LLM
    llm = load_llm()

    # Chat UI
    if 'user' not in st.session_state:
        st.session_state['user'] = ["Hey there"]

    if 'assistant' not in st.session_state:
        st.session_state['assistant'] = ["Hello, I am Matador Chatbot! How can I help you find your car today?"]

    container = st.container()

    with container:
        with st.form(key='car_form', clear_on_submit=True):
            user_input = st.text_input("", placeholder="Enter your car preferences...", key='input')
            submit = st.form_submit_button(label='Submit')

        if submit:
            # Append user message
            st.session_state['user'].append(user_input)

            # Extract preferences and find matching cars
            preferences = extract_preferences(user_input, carsDF)
            matching_cars = get_matching_cars(preferences, carsDF)

            # LLM response
            if matching_cars is not None:
                llm_response = f"Here are the top 3 cars that match your preferences:\n\n{matching_cars.to_string(index=False)}"
            else:
                llm_response = "I couldn't find any cars matching your preferences. Could you refine your criteria?"

            st.session_state['assistant'].append(llm_response)

    # Display chat messages
    if st.session_state['assistant']:
        for i in range(len(st.session_state['assistant'])):
            st.write(st.session_state["user"][i])
            st.write(st.session_state["assistant"][i])
