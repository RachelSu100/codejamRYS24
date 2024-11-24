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









# # Load LLM (dummy function for now)
# def load_llm():
#     print("Loading LLM - placeholder function")
#     return None

# def extract_preferences(message, carsDF):
#     preferences = {}

#     # New or Used
#     if "new" in message.lower() or "used" in message.lower():
#         preferences['car_type'] = 'New' if 'new' in message.lower() else 'Used'

#     # Year
#     year = [int(word) for word in message.split() if word.isdigit() and len(word) == 4]
#     if year:
#         preferences['year'] = year[0]

#     # Make
#     makes = carsDF['Make'].str.lower().unique()
#     if any(make in message.lower() for make in makes):
#         preferences['make'] = next(make.title() for make in makes if make in message.lower())

#     # Model
#     models = carsDF['Model'].str.lower().unique()
#     if any(model in message.lower() for model in models):
#         preferences['model'] = next(model.title() for model in models if model in message.lower())

#     # Body
#     body_styles = carsDF['Body'].str.lower().unique()
#     for body in body_styles:
#         if body in message.lower():
#             preferences['body'] = body.title()
#             break

#     # Engine Cylinders
#     engine_cylinders = re.search(r'(\d+)\s*cylinders', message, re.IGNORECASE)
#     if engine_cylinders:
#         preferences['engine_cylinders'] = int(engine_cylinders.group(1))

#     # Drivetrain
#     drivetrains = ['AWD', 'FWD', 'RWD', '4WD']
#     for drivetrain in drivetrains:
#         if drivetrain.lower() in message.lower():
#             preferences['drivetrain'] = drivetrain.upper()
#             break

#     # Fuel Type
#     fuel_types = carsDF['Fuel_Type'].str.lower().unique()
#     for fuel in fuel_types:
#         if fuel in message.lower():
#             preferences['fuel_type'] = fuel.title()
#             break

#     # Transmission
#     if "automatic" in message.lower():
#         preferences['transmission'] = "Automatic"
#     elif "manual" in message.lower():
#         preferences['transmission'] = "Manual"

#     # Exterior Color
#     colors = carsDF['Ext_Color_Generic'].dropna().apply(str).str.lower().unique()
#     for color in colors:
#         if color in message.lower():
#             preferences['exterior_color'] = color.title()
#             break

#     # Interior Color
#     interior_colors = carsDF['Int_Color_Generic'].dropna().apply(str).str.lower().unique()
#     for color in interior_colors:
#         if color in message.lower():
#             preferences['interior_color'] = color.title()
#             break

#     # Price Range
#     price_under = re.search(r'under\s*\$?(\d{1,3}(?:,\d{3})*)', message, re.IGNORECASE)
#     price_between = re.search(r'between\s*\$?(\d{1,3}(?:,\d{3})*)\s*and\s*\$?(\d{1,3}(?:,\d{3})*)', message, re.IGNORECASE)

#     if price_under:
#         max_price = int(price_under.group(1).replace(',', ''))
#         preferences['max_price'] = max_price
#     elif price_between:
#         min_price = int(price_between.group(1).replace(',', ''))
#         max_price = int(price_between.group(2).replace(',', ''))
#         preferences['min_price'] = min_price
#         preferences['max_price'] = max_price

#     # Mileage
#     if "low mileage" in message.lower():
#         preferences['max_miles'] = 50000
#     mileage = re.search(r'(\d{1,3}(?:,\d{3})*)\s*miles', message, re.IGNORECASE)
#     if mileage:
#         preferences['max_miles'] = int(mileage.group(1).replace(',', ''))

#     # Passenger Capacity
#     passenger_capacity = re.search(r'(\d+)\s*passenger', message, re.IGNORECASE)
#     if passenger_capacity:
#         preferences['passenger_capacity'] = int(passenger_capacity.group(1))

#     # Certified
#     if "certified" in message.lower():
#         preferences['certified'] = True

#     return preferences


# # Function to get matching cars based on preferences
# def get_matching_cars(preferences, carsDF):
#     filtered_cars = carsDF

#     # Apply each preference as a filter
#     if 'car_type' in preferences:
#         filtered_cars = filtered_cars[filtered_cars['Type'].str.lower() == preferences['car_type'].lower()]

#     if 'year' in preferences:
#         filtered_cars = filtered_cars[filtered_cars['Year'] == preferences['year']]

#     if 'make' in preferences:
#         filtered_cars = filtered_cars[filtered_cars['Make'].str.lower() == preferences['make'].lower()]

#     if 'model' in preferences:
#         filtered_cars = filtered_cars[filtered_cars['Model'].str.lower() == preferences['model'].lower()]

#     if 'body' in preferences:
#         filtered_cars = filtered_cars[filtered_cars['Body'].str.lower() == preferences['body'].lower()]

#     if 'engine_cylinders' in preferences:
#         filtered_cars = filtered_cars[filtered_cars['EngineCylinders'] == preferences['engine_cylinders']]

#     if 'drivetrain' in preferences:
#         filtered_cars = filtered_cars[filtered_cars['Drivetrain'].str.upper() == preferences['drivetrain']]

#     if 'fuel_type' in preferences:
#         filtered_cars = filtered_cars[filtered_cars['Fuel_Type'].str.lower() == preferences['fuel_type'].lower()]

#     if 'transmission' in preferences:
#         filtered_cars = filtered_cars[filtered_cars['Transmission'].str.lower() == preferences['transmission'].lower()]

#     if 'exterior_color' in preferences:
#         filtered_cars = filtered_cars[filtered_cars['Ext_Color_Generic'].str.lower() == preferences['exterior_color'].lower()]

#     if 'interior_color' in preferences:
#         filtered_cars = filtered_cars[filtered_cars['Int_Color_Generic'].str.lower() == preferences['interior_color'].lower()]

#     if 'min_price' in preferences and 'max_price' in preferences:
#         filtered_cars = filtered_cars[
#             (filtered_cars['SellingPrice'] >= preferences['min_price']) & 
#             (filtered_cars['SellingPrice'] <= preferences['max_price'])
#         ]
#     elif 'max_price' in preferences:
#         filtered_cars = filtered_cars[filtered_cars['SellingPrice'] <= preferences['max_price']]

#     if 'max_miles' in preferences:
#         filtered_cars = filtered_cars[filtered_cars['Miles'] <= preferences['max_miles']]

#     if 'passenger_capacity' in preferences:
#         filtered_cars = filtered_cars[filtered_cars['PassengerCapacity'] == preferences['passenger_capacity']]

#     if 'certified' in preferences:
#         filtered_cars = filtered_cars[filtered_cars['Certified'] == preferences['certified']]

#     # Return the top 3 matching cars
#     if not filtered_cars.empty:
#         return filtered_cars[['Make', 'Model', 'Year', 'VIN', 'SellingPrice', 'Miles', 'Fuel_Type', 'PassengerCapacity']].head(3)
#     else:
#         return None
    

# # Streamlit UI
# st.title("Matador Car Match Chatbot")

# # Sidebar for CSV file upload
# csv_data = st.sidebar.file_uploader("Upload vehicles.csv", type="csv")

# if csv_data is not None:
#     # Handle uploaded CSV file with temporary file
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
#         tmp_file.write(csv_data.getvalue())
#         tmp_file_path = tmp_file.name

#     # Load CSV data using pandas directly
#     carsDF = pd.read_csv(tmp_file_path)

#     # Chat UI
#     if 'user' not in st.session_state:
#         st.session_state['user'] = ["Hi there! Please start describing which type of car you are looking for."]

#     if 'assistant' not in st.session_state:
#         st.session_state['assistant'] = ["Hello, I am Matador Chatbot! How can I help you find your car today?"]

#     container = st.container()

#     with container:
#         with st.form(key='car_form', clear_on_submit=True):
#             user_input = st.text_input("", placeholder="Enter your car preferences...", key='input')
#             submit = st.form_submit_button(label='Submit')

#         if submit:
#             # Append user message
#             st.session_state['user'].append(user_input)

#             # Extract preferences and find matching cars
#             preferences = extract_preferences(user_input, carsDF)
#             matching_cars = get_matching_cars(preferences, carsDF)

#             # If preferences are not detailed enough, ask follow-up questions
#             if not preferences:
#                 llm_response = "Could you please provide more details? For example, do you have a budget or a specific brand in mind?"
#             elif matching_cars is None:
#                 llm_response = "I couldn't find any cars matching your preferences. Could you refine your criteria?"
#             else:
#                 # Create a prettier output for the car results
#                 llm_response = "Here are the top 3 cars that match your preferences:"
#                 for index, car in matching_cars.iterrows():
#                     llm_response += f"\n\n- **{car['Year']} {car['Make']} {car['Model']}**\n"
#                     llm_response += f"  - VIN: {car['VIN']}\n"
#                     llm_response += f"  - Selling Price: ${car['SellingPrice']:,.2f}\n"
#                     llm_response += f"  - Mileage: {car['Miles']} miles\n"
#                     llm_response += f"  - Fuel Type: {car['Fuel_Type']}\n"
#                     llm_response += f"  - Passenger Capacity: {car['PassengerCapacity']}"

#             st.session_state['assistant'].append(llm_response)

#     # Display chat messages
#     if st.session_state['assistant']:
#         for i in range(len(st.session_state['assistant'])):
#             # Display user message
#             st.write(f"**User:** {st.session_state['user'][i]}")
#             # Display assistant message
#             st.markdown(f"**Assistant:** {st.session_state['assistant'][i]}")

