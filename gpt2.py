# imports
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the tokenizer and model
print("Loading GPT-2 model...")
model_id = "gpt2"  # GPT-2 is smaller, fully open, and good for prototyping
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
print("GPT-2 model loaded successfully!")

# Load the dataset into a dataframe (replace 'vehicles.csv' with your file path)
csv_data = 'vehicles.csv'
carsDF = pd.read_csv(csv_data)

# Initialize conversation state
conversation_state = {
    "preferences": {},
    "available_cars": carsDF,
    "ask_for_more_details": True,
    "conversation_history": []  # Keep a history of the conversation
}

# Function to extract preferences from user's message
def extract_preferences(message):
    preferences = {}

    # Extract details based on user's message
    if "new" in message.lower() or "used" in message.lower():
        preferences['car_type'] = 'New' if 'new' in message.lower() else 'Used'

    year = [int(word) for word in message.split() if word.isdigit() and len(word) == 4]
    if year:
        preferences['year'] = year[0]

    makes = carsDF['Make'].str.lower().unique()
    if any(make in message.lower() for make in makes):
        preferences['make'] = next(make.title() for make in makes if make in message.lower())

    models = carsDF['Model'].str.lower().unique()
    if any(model in message.lower() for model in models):
        preferences['model'] = next(model.title() for model in models if model in message.lower())

    if "fast" in message.lower() or "high performance" in message.lower():
        preferences['fast_car'] = True

    if "cheap" in message.lower():
        preferences['cheap_car'] = True

    return preferences

# Function to filter cars based on preferences
def filter_cars(preferences, carsDF):
    filtered_cars = carsDF

    # Apply filters
    if 'car_type' in preferences:
        filtered_cars = filtered_cars[filtered_cars['Type'].str.lower() == preferences['car_type'].lower()]

    if 'year' in preferences:
        filtered_cars = filtered_cars[filtered_cars['Year'] == preferences['year']]

    if 'make' in preferences:
        filtered_cars = filtered_cars[filtered_cars['Make'].str.lower() == preferences['make'].lower()]

    if 'model' in preferences:
        filtered_cars = filtered_cars[filtered_cars['Model'].str.lower() == preferences['model'].lower()]

    if 'fast_car' in preferences:
        filtered_cars = filtered_cars[filtered_cars['EngineDisplacement'].str.replace(" L", "").astype(float) > 3.0]

    if 'cheap_car' in preferences:
        filtered_cars = filtered_cars[filtered_cars['SellingPrice'] < 20000]  # Assume "cheap" means under $20,000

    return filtered_cars

# Function to format car matches as text
def format_car_matches(matching_cars):
    if matching_cars.empty:
        return "I couldn't find any cars matching your preferences. Could you refine your criteria?"

    response = "Here are the top cars that match your preferences:\n"
    for index, car in matching_cars.iterrows():
        response += (
            f"- {car['Year']} {car['Make']} {car['Model']}\n"
            f"  - VIN: {car['VIN']}\n"
            f"  - Selling Price: ${car['SellingPrice']:,.2f}\n"
            f"  - Mileage: {car['Miles']} miles\n"
            f"  - Fuel Type: {car['Fuel_Type']}\n"
            f"  - Passenger Capacity: {car['PassengerCapacity']}\n\n"
        )
    return response

# Function to generate LLM response
def generate_llm_response(context):
    # Truncate context if it exceeds the maximum length of 1024 tokens
    max_length = 1024
    input_ids = tokenizer.encode(context, return_tensors="pt")
    if input_ids.shape[-1] > max_length:
        input_ids = input_ids[:, -max_length:]  # Keep only the last 1024 tokens

    outputs = model.generate(input_ids, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Terminal-based interactive loop
def run_chat():
    global conversation_state
    print("Welcome to the Matador Car Match Chatbot!")
    print("Hello, I am Matador Chatbot! How can I help you find your car today?")

    while True:
        user_message = input("\nYou: ")

        # Exit condition
        if user_message.lower() in ['exit', 'quit']:
            print("Goodbye! Thanks for using Matador Car Match Chatbot.")
            break

        # Extract preferences from user input
        new_preferences = extract_preferences(user_message)
        conversation_state["preferences"].update(new_preferences)

        # Update available cars based on new preferences
        conversation_state["available_cars"] = filter_cars(conversation_state["preferences"], conversation_state["available_cars"])

        # If no cars found, ask for more details
        if conversation_state["available_cars"].empty:
            assistant_response = "I couldn't find any cars matching your preferences. Could you refine your criteria?"
            conversation_state["ask_for_more_details"] = True
        else:
            # Format car matches and respond
            assistant_response = format_car_matches(conversation_state["available_cars"])

            # If more details are needed, ask follow-up questions
            if conversation_state["ask_for_more_details"]:
                assistant_response += "\nCould you please provide more details? For example, do you have a specific brand or year in mind?"
                conversation_state["ask_for_more_details"] = False

        # Limit conversation history to the most recent messages
        max_context_length = 500
        conversation_state["conversation_history"].append(f"User: {user_message}\nAssistant: {assistant_response}")
        if len(conversation_state["conversation_history"]) > max_context_length:
            conversation_state["conversation_history"] = conversation_state["conversation_history"][-max_context_length:]

        # Generate LLM-based response for more interactive experience
        context = "\n".join(conversation_state["conversation_history"])
        llm_generated_response = generate_llm_response(context)

        # Display assistant's response
        final_response = llm_generated_response.split("Assistant:")[-1].strip()  # Extract only the Assistant's reply
        print(f"\nAssistant: {final_response}")

# Run the terminal chat
if __name__ == "__main__":
    run_chat()
