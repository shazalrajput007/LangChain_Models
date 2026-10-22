from langchain_cohere import CohereEmbeddings
from dotenv import load_dotenv

load_dotenv()


embeddings = CohereEmbeddings(model="embed-english-light-v3.0")

document = [
    "I need AI/ML Internship",
    "What you think? when i will get?",
    "Can yuo help me in this?"
]

response = embeddings.embed_documents(document)

print(str(response))