from langchain_cohere import CohereEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()


sentences = [
    "Machine learning enables computers to learn from data.",
    "Deep learning uses neural networks with multiple layers.",
    "Python is a popular programming language.",
    "Natural language processing helps computers understand text.",
    "Computer vision enables machines to understand images.",
    "Semantic search finds results based on meaning."
]


embeddings = CohereEmbeddings(
    model="embed-english-light-v3.0"
)


sentence_vectors = embeddings.embed_documents(sentences)


query = input("Enter your query: ")


query_vector = embeddings.embed_query(query)


scores = cosine_similarity(
    [query_vector],
    sentence_vectors
)[0]


top_indices = scores.argsort()[::-1][:3]

for index in top_indices:
    print("Sentence:", sentences[index])
    print("Similarity:", round(float(scores[index]), 4))
    print("-" * 40)
