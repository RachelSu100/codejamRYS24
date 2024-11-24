# imports
import pandas as pd
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import tempfile

print("Starting imports...")
print("Successfully imported!")

# Load the tokenizer and model
model_id = "gpt2"  # GPT-2 is smaller, fully open, and good for prototyping
print("Loading GPT-2 model...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
print("GPT-2 model loaded successfully!")

# Extract preferences from user's message
def extract_preferences(message, carsDF):
    preferences = {}

    # Check if the user is asking for available models or car types
    if "model" in message.lower() or "type" in message.lower() or "offer" in message.lower():
        preferences['ask_for_models'] = True

    # New or Used
    if "new" in message.lower() or "used" in message.lower():
        preferences['car_type'] = 'New' if 'new' in message.lower() else 'Used'

    # Year
    year = [int(word) for word in message.split() if word.isdigit() and len(word) == 4]
    if year:
        preferences['year'] = year[0]

    # Make
    makes = carsDF['Make'].str.lower().unique()
    if any(make in message.lower() for make in makes):
        preferences['make'] = next(make.title() for make in makes if make in message.lower())

    # Model
    models = carsDF['Model'].str.lower().unique()
    if any(model in message.lower() for model in models):
        preferences['model'] = next(model.title() for model in models if model in message.lower())

    # Fast car (determine based on engine displacement or high-performance description)
    if "fast" in message.lower() or "high performance" in message.lower():
        preferences['fast_car'] = True

    return preferences


# Function to get matching cars based on preferences
def get_matching_cars(preferences, carsDF):
    if 'ask_for_models' in preferences:
        return list_available_models(carsDF)

    filtered_cars = carsDF

    # Apply each preference as a filter
    if 'car_type' in preferences:
        filtered_cars = filtered_cars[filtered_cars['Type'].str.lower() == preferences['car_type'].lower()]

    if 'year' in preferences:
        filtered_cars = filtered_cars[filtered_cars['Year'] == preferences['year']]

    if 'make' in preferences:
        filtered_cars = filtered_cars[filtered_cars['Make'].str.lower() == preferences['make'].lower()]

    if 'model' in preferences:
        filtered_cars = filtered_cars[filtered_cars['Model'].str.lower() == preferences['model'].lower()]

    if 'fast_car' in preferences:
        # Assume fast cars have large engine displacement or high horsepower descriptions
        filtered_cars = filtered_cars[filtered_cars['EngineDisplacement'].str.replace(" L", "").astype(float) > 3.0]

    # Return the top 3 matching cars
    if not filtered_cars.empty:
        return filtered_cars[['Make', 'Model', 'Year', 'VIN', 'SellingPrice', 'Miles', 'Fuel_Type', 'PassengerCapacity']].head(3)
    else:
        return None
    


# Function to generate LLM response
def generate_llm_response(context):
    inputs = tokenizer(context, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=50)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


# Function to list available models
def list_available_models(carsDF):
    makes_and_models = carsDF[['Make', 'Model']].drop_duplicates()
    model_list = ""
    for index, row in makes_and_models.iterrows():
        model_list += f"- {row['Make']} {row['Model']}\n"
    return model_list


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

    # Chat UI
    if 'user' not in st.session_state:
        st.session_state['user'] = ["Hi there! Please start describing which type of car you are looking for."]

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
            if 'ask_for_models' in preferences:
                # Provide a list of available models if user is asking for types/models of cars
                llm_response = "Here are some of the car models we offer:\n\n" + list_available_models(carsDF)
            else:
                matching_cars = get_matching_cars(preferences, carsDF)

                # If preferences are not detailed enough, ask follow-up questions
                if not preferences:
                    llm_response = "Could you please provide more details? For example, do you have a budget or a specific brand in mind?"
                elif matching_cars is None:
                    llm_response = "I couldn't find any cars matching your preferences. Could you refine your criteria?"
                else:
                    # Create a prettier output for the car results
                    llm_response = "Here are the top 3 cars that match your preferences:"
                    for index, car in matching_cars.iterrows():
                        llm_response += f"\n\n- **{car['Year']} {car['Make']} {car['Model']}**\n"
                        llm_response += f"  - VIN: {car['VIN']}\n"
                        llm_response += f"  - Selling Price: ${car['SellingPrice']:,.2f}\n"
                        llm_response += f"  - Mileage: {car['Miles']} miles\n"
                        llm_response += f"  - Fuel Type: {car['Fuel_Type']}\n"
                        llm_response += f"  - Passenger Capacity: {car['PassengerCapacity']}"

            # Generate LLM-based response for more dynamic interaction
            context = f"User: {user_input}\nAssistant: {llm_response}"
            llm_generated_response = generate_llm_response(context)

            # Append both responses
            st.session_state['assistant'].append(llm_generated_response)

    # Display chat messages
    if st.session_state['assistant']:
        for i in range(len(st.session_state['assistant'])):
            # Display user message
            st.write(f"**User:** {st.session_state['user'][i]}")
            # Display assistant message
            st.markdown(f"**Assistant:** {st.session_state['assistant'][i]}")



