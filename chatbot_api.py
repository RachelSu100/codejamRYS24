from flask import Flask, request, jsonify
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

# Load the model and tokenizer
model_id = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Load vehicle data
carsDF = pd.read_csv('vehicles.csv')  # Assuming vehicles.csv is already present

# Function to generate a response
def generate_llm_response(context):
    inputs = tokenizer(context, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=50)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get("message", "")

    # Extract preferences, get matching cars, generate response, etc.
    # (You can include the functions extract_preferences and get_matching_cars as needed)
    
    # Example simple response generation:
    response = generate_llm_response(user_message)
    
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)