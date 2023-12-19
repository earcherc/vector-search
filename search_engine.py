import pandas as pd
import os
from pathlib import Path
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import argparse
import joblib

# Initialize NLP model
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
nlp.add_pipe("sentencizer")

model = SentenceTransformer("all-MiniLM-L6-v2")


def load_data(data_dir):
    """
    Load data from text files and return a DataFrame.
    """
    data = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            print(f"Processing {filename}...")
            file_path = data_dir / filename
            with open(file_path, "r", encoding="utf-8") as file:
                line_number = 1
                for line in file:
                    doc = nlp(line.strip())
                    for sent in doc.sents:
                        sentence = sent.text.strip()
                        data.append(
                            {
                                "filename": filename,
                                "sentence": sentence,
                                "line_number": line_number,
                            }
                        )
                    line_number += 1
    return pd.DataFrame(data)


def generate_embeddings(df):
    """
    Generate embeddings for the sentences in the DataFrame.
    """
    print("Generating embeddings for the dataset...")
    embeddings = model.encode(df["sentence"].tolist(), show_progress_bar=True)
    return embeddings


def search(query, embeddings, df):
    """
    Perform a search against the embeddings with the given query.
    """
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(similarities)[-5:]
    top_docs = df.iloc[top_indices]
    top_scores = similarities[top_indices]
    return top_docs, top_scores


def main():
    parser = argparse.ArgumentParser(description="Text Search Engine")
    parser.add_argument("query", help="Search query")
    args = parser.parse_args()

    data_dir = Path.cwd() / "data"
    df_dir = Path.cwd() / "saved_dfs"
    df_path = df_dir / "dataframe.csv"
    embeddings_path = Path.cwd() / "embeddings" / "sentence_embeddings_v1.joblib"

    df_dir.mkdir(exist_ok=True)
    Path.cwd().joinpath("embeddings").mkdir(exist_ok=True)

    # Load or create DataFrame
    if df_path.exists():
        print("Loading dataset from CSV...")
        df = pd.read_csv(df_path)
    else:
        df = load_data(data_dir)
        df.to_csv(df_path, index=False)

    # Load or generate embeddings
    if embeddings_path.exists():
        print("Loading embeddings from file...")
        embeddings = joblib.load(embeddings_path)
    else:
        embeddings = generate_embeddings(df)
        joblib.dump(embeddings, embeddings_path)

    # Perform search
    query = args.query
    top_docs, top_scores = search(query, embeddings, df)

    # Print search results
    for i, (index, row) in enumerate(top_docs.iterrows()):
        print(f"Result {i+1}:")
        print(f"  Sentence: {row['sentence']}")
        print(f"  File: {row['filename']}, Line: {row['line_number']}")


if __name__ == "__main__":
    main()
