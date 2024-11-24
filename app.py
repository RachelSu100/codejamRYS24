from flask import Flask, request, jsonify
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
from qdrant_client import QdrantClient
from flask_cors import CORS  # Import CORS

app = Flask(__name__)
CORS(app)  

# Load the necessary models
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_generator = pipeline("text-generation", model=gpt2_model, tokenizer=gpt2_tokenizer)

# Connect to Qdrant
qdrant_client = QdrantClient(host="localhost", port=6333)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_query = request.json.get('query')
        print(user_query)
        if not user_query:
            return jsonify({"error": "No query provided"}), 400

        # Generate embedding for the user query
        query_embedding = sentence_model.encode([user_query])[0]

        # Search for car options in Qdrant
        results = qdrant_client.search(
            collection_name="car_inventory",
            query_vector=query_embedding,
            limit=20
        )

        # Get the payloads from the results and choose the top 5
        candidate_data = [result.payload for result in results][:5]

        # Generate descriptions for the top 5 cars
        candidate_descriptions = [
            f"{idx+1}. {car['Year']} {car['Make']} {car['Model']}, {car['Miles']} miles, priced at ${car['SellingPrice']}."
            for idx, car in enumerate(candidate_data)
        ]

        # Construct the prompt for GPT-2
        prompt = (
            "Here are some great car options I found for you:\n"
            + "\n".join(candidate_descriptions)
            + "\n\nDo any of these cars sound interesting to you? Or do you have more preferences you'd like me to consider?"
        )

        # Generate a conversational response using GPT-2
        response = gpt2_generator(
            prompt,
            max_length=150,
            num_return_sequences=1,
            pad_token_id=gpt2_tokenizer.eos_token_id
        )

        # Output the response
        return jsonify({"response": response[0]['generated_text']})

    except Exception as e:
        # Log the error to the console
        print(f"Error: {e}")
        return jsonify({"error": "An unexpected error occurred.", "details": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
