from langchain_cohere import CohereEmbeddings
from dotenv import load_dotenv

load_dotenv()


embeddings = CohereEmbeddings(model="embed-english-light-v3.0")

response = embeddings.embed_query("what is this world?")

print(str(response))