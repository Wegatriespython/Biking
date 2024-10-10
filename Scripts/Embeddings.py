import os
import nltk
import pickle
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Download NLTK data
nltk.download('punkt', quiet=True)

def load_documents(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                text = file.read()
                documents.append(text)
    return documents

def split_into_sentences(documents):
    sentences = []
    for doc in documents:
        sentences.extend(sent_tokenize(doc))
    return sentences

def fine_tune_model(model, sentences, epochs=1, batch_size=32):
    train_examples = [InputExample(texts=[sent, sent]) for sent in sentences]
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=100,
        show_progress_bar=True
    )
    
    return model

def create_informed_embeddings(model, words):
    return model.encode(words, convert_to_tensor=True)

def main():
    folder_path = r'V:\Prague_Biking\Data\Interview recordings+Transcripts\Street Interviews'
    documents = load_documents(folder_path)
    print(f"Loaded {len(documents)} documents")

    sentences = split_into_sentences(documents)
    print(f"Split into {len(sentences)} sentences")

    model = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')

    print("Fine-tuning the model...")
    fine_tuned_model = fine_tune_model(model, sentences)

    topics_and_words = {
        "safety": ["safety", "secure", "protection", "hazard", "risk"],
        "infrastructure": ["infrastructure", "road", "path", "lane", "facility"],
        "cost": ["cost", "price", "expense", "affordable", "budget"],
        "perception": ["perception", "view", "opinion", "attitude", "perspective"],
        "health": ["health", "fitness", "exercise", "wellness", "active"],
        "traffic": ["traffic", "congestion", "flow", "volume", "vehicle"],
        "weather": ["weather", "climate", "temperature", "rain", "wind"],
        "topography": ["topography", "terrain", "hill", "slope", "elevation"],
        "culture": ["culture", "lifestyle", "habit", "tradition", "custom"]
    }

    all_words = [word for words in topics_and_words.values() for word in words]

    print("Creating informed embeddings...")
    informed_embeddings = create_informed_embeddings(fine_tuned_model, all_words)

    embeddings_dict = {word: embedding for word, embedding in zip(all_words, informed_embeddings)}
    with open('informed_embeddings.pkl', 'wb') as f:
        pickle.dump(embeddings_dict, f)

    print("Informed embeddings created and saved.")

if __name__ == "__main__":
    main()
    