import pandas as pd
data = pd.read_csv('vehicles.csv')

# Specify the columns you want to keep
columns_to_keep = ['Year', 'Make', 'Model', 'Type', 'SellingPrice', 'Miles']

# Create a new DataFrame with only the specified columns
filtered_data = data[columns_to_keep]

# Fill missing values for 'Year', 'SellingPrice', 'Miles' with the average value and assign back
filtered_data['Year'] = filtered_data['Year'].fillna(filtered_data['Year'].mean())
filtered_data['SellingPrice'] = filtered_data['SellingPrice'].fillna(filtered_data['SellingPrice'].mean())
filtered_data['Miles'] = filtered_data['Miles'].fillna(filtered_data['Miles'].mean())

# Step 2: Drop rows where any of ['Make', 'Model', 'Type'] is missing
filtered_data.dropna(subset=['Make', 'Model', 'Type'], inplace=True)

# Create descriptions using the specified columns
filtered_data['Description'] = filtered_data.apply(
    lambda x: f"This {x['Type'].lower()} {x['Year']} {x['Make']} {x['Model']} is available with {x['Miles']} miles on it, priced at ${x['SellingPrice']}.",
    axis=1
)

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
car_descriptions = filtered_data['Description'].tolist()
embeddings = model.encode(car_descriptions)  # Tokenization happens inside this method

# %pip install qdrant-client --quiet

from qdrant_client import QdrantClient

# Connect to Qdrant
qdrant_client = QdrantClient(host="localhost", port=6333)

# Recreate collection in Qdrant
qdrant_client.recreate_collection(
    collection_name="car_inventory",
    vectors_config={
        "size": embeddings.shape[1],  # Set the size of the embeddings
        "distance": "Cosine"
    }
)

# Insert embeddings and payload data (the car metadata)
payloads = filtered_data.to_dict(orient='records')

for idx, embedding in enumerate(embeddings):
    qdrant_client.upsert(
        collection_name="car_inventory",
        points=[{
            "id": idx,
            "vector": embedding,
            "payload": payloads[idx]  # Metadata like make, model, year, etc.
        }]
    )
 
user_query = "I want a car made in 2020"
query_embedding = model.encode([user_query])[0]



results = qdrant_client.search(
    collection_name="car_inventory",
    query_vector=query_embedding,
    limit=20  
)

# Print results
for result in results:
    car_details = result.payload
    print(f"Match: {car_details['Year']} {car_details['Make']} {car_details['Model']}, Price: ${car_details['SellingPrice']}")


from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

# Load GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Create text generation pipeline for GPT-2
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Assuming we have retrieved the top 20 candidates
results = qdrant_client.search(
    collection_name="car_inventory",
    query_vector=query_embedding,
    limit=20  
)


# Get the payloads from the results
candidate_data = [result.payload for result in results]

candidate_data = candidate_data[:5]

# Create a description of the top 5 cars
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
response = generator(prompt, max_length=150, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

# Output the response
print(response[0]['generated_text'])