import os
import numpy as np
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from umap import UMAP
import matplotlib.pyplot as plt
from tqdm import tqdm
import networkx as nx
import nltk
from pyvis.network import Network
nltk.download('punkt', quiet=True)

# Load the model
model = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')

def split_into_chunks(text, chunk_size=10):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0
    for sentence in sentences:
        words = sentence.split()
        if current_length + len(words) <= chunk_size:
            current_chunk.extend(words)
            current_length += len(words)
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = words
            current_length = len(words)
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

def get_relevant_chunks(document, topic_embedding, model, top_n=2):
    chunks = split_into_chunks(document)
    chunk_embeddings = model.encode(chunks)
    similarities = cosine_similarity(chunk_embeddings, topic_embedding.reshape(1, -1)).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    return [chunks[i] for i in top_indices]
    
def create_topic_graph(topic, doc_embeddings, topic_embedding, topic_influences, documents, model, top_n=10, chunks_per_doc=2):
    
    
    G = nx.Graph()
    
    # Add topic node
    G.add_node(topic, size=30, color='#FF5733', label=topic, group=1)
    
    # Get top N influential documents
    top_indices = topic_influences.argsort()[-top_n:][::-1]
    
    # Add document nodes and edges
    for i, idx in enumerate(top_indices):
        doc_label = f"Doc {idx}"
        influence = topic_influences[idx]
        relevant_chunks = get_relevant_chunks(documents[idx], topic_embedding, model, chunks_per_doc)
        chunk_text = "\n\n".join(relevant_chunks)
        G.add_node(doc_label, size=10 + influence * 10, color='#33B5FF', 
                   label=doc_label, title=chunk_text, group=2)
        G.add_edge(topic, doc_label, weight=influence*10)
    
    # Add edges between documents based on similarity
    doc_embeddings_subset = doc_embeddings[top_indices]
    doc_similarities = cosine_similarity(doc_embeddings_subset)
    for i in range(len(top_indices)):
        for j in range(i+1, len(top_indices)):
            similarity = doc_similarities[i][j]
            if similarity > 0.5:  # Threshold for similarity
                G.add_edge(f"Doc {top_indices[i]}", f"Doc {top_indices[j]}", weight=similarity*5)
    
    return G

def visualize_topic_graph(G, topic, output_file):
    nt = Network(height="500px", width="100%", bgcolor="#222222", font_color="white")
    nt.from_nx(G)
    nt.show_buttons(filter_=['physics'])
    nt.save_graph(output_file)
    
def load_documents(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                text = file.read()
                documents.append(text)
    return documents

def calculate_topic_influence(doc_embedding, topic_embeddings):
    similarities = cosine_similarity(doc_embedding.reshape(1, -1), topic_embeddings)[0]
    # We use softmax to normalize the similarities
    exp_similarities = np.exp(similarities)
    return exp_similarities / exp_similarities.sum()

def main():
    folder_path = r'V:\Prague_Biking\Data\Interview recordings+Transcripts\Street Interviews'
    documents = load_documents(folder_path)
    print(f"Loaded {len(documents)} documents")

    # Pre-specified topics
    topics = ["safety", "infrastructure", "biking"]
    topic_embeddings = model.encode(topics)

    # Generate embeddings for documents
    doc_embeddings = model.encode(documents, show_progress_bar=True)

    # Calculate topic influence for each document
    topic_influences = np.array([calculate_topic_influence(doc_emb, topic_embeddings) for doc_emb in doc_embeddings])

    # Reduce dimensionality for visualization
    umap_model = UMAP(n_components=2, random_state=42)
    doc_embedding_2d = umap_model.fit_transform(doc_embeddings)

    # Plotting
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(doc_embedding_2d[:, 0], doc_embedding_2d[:, 1], 
                          c=topic_influences, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter)
    plt.title("Document Semantic Space with Topic Influences")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")

    # Add topic labels
    for i, topic in enumerate(topics):
        centroid = np.average(doc_embedding_2d, axis=0, weights=topic_influences[:, i])
        plt.annotate(topic, centroid, fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig("semantic_space_visualization.png")
    plt.close()
    for i, topic in enumerate(topics):
        # Create graph for each topic
        G = create_topic_graph(topic, doc_embeddings, topic_embeddings[i], topic_influences[:, i], documents, model, top_n=10, chunks_per_doc=2)
        
        # Visualize the graph
        output_file = f"semantic_graph_{topic.lower()}.html"
        visualize_topic_graph(G, topic, output_file)
        print(f"Semantic graph for '{topic}' has been saved as '{output_file}'")

        # Print top influenced documents
        print(f"\nTop 5 documents most influenced by '{topic}':")
        top_indices = topic_influences[:, i].argsort()[-5:][::-1]
        for idx in top_indices:
            print(f"Document {idx}: Influence score = {topic_influences[idx, i]:.4f}")
            print(documents[idx][:200] + "...\n")



if __name__ == "__main__":
    main()