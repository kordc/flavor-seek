import argparse
import gc
import json
import os
import string
from collections import defaultdict

import faiss
import numpy as np
import pandas as pd
import torch
from joblib import dump, load
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from transformers import AutoModel


class RecipeData:
    def __init__(self, raw_data_path, preprocessed_data_path):
        self.raw_data_path = raw_data_path
        self.preprocessed_data_path = preprocessed_data_path

    def load_data(self):
        try:
            self.df = pd.read_csv(self.preprocessed_data_path)
            print("Loaded preprocessed data.")
        except FileNotFoundError:
            self.df = pd.read_csv(self.raw_data_path)
            self.df.fillna("", inplace=True)
            self.preprocess()
            self.df.to_csv(self.preprocessed_data_path, index=False)
            print("Preprocessed and saved data.")

    def preprocess(self):
        self.df["combined"] = (
            self.df["name"]
            + ": "
            + self.df["description"]
            + " "
            + self.df["steps"]
            + " "
            + self.df["ingredients"]
        )
        self.df["combined"] = self.df["combined"].apply(self.clean_text)

    @staticmethod
    def clean_text(text):
        # Convert text to lowercase
        text = text.lower()

        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))

        # Tokenize the text
        tokens = text.split()

        # Remove stopwords
        stop_words = set(stopwords.words("english"))
        tokens = [word for word in tokens if word not in stop_words]

        # Lemmatize the tokens
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]

        # Join the tokens back into a single string
        cleaned_text = " ".join(tokens)

        return cleaned_text


class TfIdfSearchEngine:
    def __init__(self, recipe_data, max_features=512, save_vectors=False):
        self.recipe_data = recipe_data
        self.max_features = max_features
        self.save_vectors = save_vectors
        self.vectorizer = None
        self.index = None

    def load_or_create_vectorizer(self):
        try:
            self.vectorizer = load("data/vectorizer.joblib")
        except FileNotFoundError:
            self.vectorizer = TfidfVectorizer(
                stop_words="english", max_features=self.max_features
            )
            self.vectorizer.fit(self.recipe_data.df["combined"])
            dump(self.vectorizer, "data/vectorizer.joblib")

    def load_or_create_tfidf_matrix(self):
        try:
            return load("data/tfidf_matrix.joblib")
        except FileNotFoundError:
            tfidf_matrix = self.vectorizer.transform(self.recipe_data.df["combined"])
            dump(tfidf_matrix, "data/tfidf_matrix.joblib")
            return tfidf_matrix

    def save_vectors_to_csv(self, matrix, filename):
        if self.save_vectors:
            pd.DataFrame(matrix).to_csv(filename, index=False)

    def save_vectors_to_parquet_file(self, matrix, filename):
        if self.save_vectors:
            pd.DataFrame(matrix).to_parquet(filename, index=False)

    def prepare_index(self, tfidf_matrix):
        d = tfidf_matrix.shape[1]
        index = faiss.IndexFlatL2(d)
        if faiss.get_num_gpus() > 0:
            self.index = faiss.index_cpu_to_all_gpus(index)
        else:
            self.index = index
        self.index.add(tfidf_matrix.toarray().astype("float32"))

    def prepare(self):
        self.load_or_create_vectorizer()
        self.tfidf_matrix = self.load_or_create_tfidf_matrix()
        df = self.recipe_data.df
        df["embedding"] = self.tfidf_matrix.toarray().tolist()
        df.rename(columns={"id": "recipe_id"}, inplace=True)
        self.save_vectors_to_parquet_file(df, "data/tfidf_recipes_matrix.parquet")
        self.prepare_index(self.tfidf_matrix)

    def get_queries_vectors(self, file_path):
        df = pd.read_parquet(file_path)
        vectors = (
            self.vectorizer.transform(df["query"].tolist()).toarray().astype("float32")
        )
        for i in range(vectors.shape[0]):
            df.at[i, "query_embeddings"] = vectors[i]
        self.save_vectors_to_parquet_file(df, "data/tfidf_queries_vectors.parquet")

    def search(self, query, top_n=10):
        query_vector = self.vectorizer.transform([query]).toarray().astype("float32")
        _, indices = self.index.search(query_vector, top_n)
        return self.recipe_data.df.iloc[indices[0]]


class BM25SearchEngine:
    def __init__(self, recipe_data, max_features=1000):
        self.recipe_data = recipe_data
        self.max_features = max_features

    def prepare(self):
        tokenized_corpus = [
            word_tokenize(doc) for doc in self.recipe_data.df["combined"]
        ]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query, top_n=10):
        print("Searching for:", query)
        query_tokens = word_tokenize(query)
        doc_scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(doc_scores)[::-1][:top_n]
        return self.recipe_data.df.iloc[top_indices]


class TextEmbedderSearchEngine:
    def __init__(self, model_name, qdrant_path, collection_name):
        self.model_name = model_name
        self.qdrant_path = qdrant_path
        self.collection_name = collection_name
        self.client = QdrantClient(path=self.qdrant_path)
        self.model = None
        self.prepare_collection()

    def prepare_collection(self):
        self.client.delete_collection(collection_name=self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=512, distance=Distance.COSINE),
        )

    def prepare_model(self):
        if self.model is None:
            self.model = AutoModel.from_pretrained(
                "jinaai/jina-embeddings-v2-small-en", trust_remote_code=True
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(device)

    def preprocess_and_upload(self, df, batch_size=512):
        df = self.preprocess(df)
        self.upload_data_to_qdrant(df, batch_size)

    def preprocess(self, df):
        df["content"] = (
            "Name:\n"
            + df["name"]
            + "\nDescription:\n"
            + df["description"]
            + "\nSteps:\n"
            + df["steps"]
        )
        df["id"] = df.index
        df.dropna(subset=["content"], inplace=True)
        return df

    def upload_data_to_qdrant(self, df, batch_size):
        self.prepare_model()
        for batch_df in tqdm(
            self.batchify(df, batch_size), total=len(df) // batch_size
        ):
            with torch.no_grad():
                texts = batch_df["content"].tolist()
                embeddings = self.model.encode(texts).tolist()
                payloads = batch_df.to_dict(orient="records")
                points = [
                    PointStruct(id=payload["id"], vector=emb, payload=payload)
                    for emb, payload in zip(embeddings, payloads)
                ]
                self.client.upsert(
                    collection_name=self.collection_name, wait=True, points=points
                )
                del texts, embeddings, payloads, points
                gc.collect()
                torch.cuda.empty_cache()
                gc.collect()

    def batchify(self, to_batch, batch_size):
        for i in range(0, len(to_batch), batch_size):
            yield to_batch[i : i + batch_size]

    def search(self, query, top_n=5):
        self.prepare_model()
        query_vector = self.model.encode(query)
        search_result = self.client.search(
            collection_name=self.collection_name, query_vector=query_vector, limit=top_n
        )
        df_result = pd.DataFrame([e.payload for e in search_result])
        return df_result


def read_queries(file_path):
    _, file_extension = os.path.splitext(file_path)

    if file_extension.lower() == ".txt":
        with open(file_path, "r") as file:
            data = file.readlines()
        return [query.strip() for query in data]

    # Read .parquet file
    elif file_extension.lower() == ".parquet":
        df = pd.read_parquet(file_path)
        return df["query"].tolist()

    else:
        raise ValueError("Unsupported file format")


def main():
    parser = argparse.ArgumentParser(description="Recipe Search Engine")
    parser.add_argument(
        "--algorithms",
        type=str,
        nargs="+",
        choices=["tfidf", "bm25", "embedder"],
        default=["tfidf", "bm25", "embedder"],
        help="Algorithm type (default: tfidf)",
    )
    parser.add_argument(
        "--print",
        dest="print_value",
        action="store_true",
        help="Print results",
        default=False,
    )
    parser.add_argument(
        "--search",
        dest="search",
        action="store_true",
        help="Search for queries",
        default=False,
    )
    args = parser.parse_args()
    print_value = args.print_value
    search_value = args.search

    queries = read_queries("data/queries.parquet")

    data = RecipeData(
        "data/raw_recipes_used.csv", "data/raw_recipes_used_preprocessed.csv"
    )
    data.load_data()

    if "tfidf" in args.algorithms:
        tfidf_engine = TfIdfSearchEngine(data, save_vectors=True)
        tfidf_engine.prepare()
        tfidf_engine.get_queries_vectors("data/queries.parquet")
    if "bm25" in args.algorithms:
        bm25_engine = BM25SearchEngine(data)
        bm25_engine.prepare()
    if "embedder" in args.algorithms:
        embedder_engine = TextEmbedderSearchEngine(
            "jinaai/jina-embeddings-v2-small-en", "/tmp/recipe_store_cli", "recipies"
        )
        embedder_engine.preprocess_and_upload(data.df)

    results = defaultdict(list)

    if search_value:
        for query in tqdm(queries):
            if print_value:
                print(f"Query: {query}")
            if "tfidf" in args.algorithms:
                if print_value:
                    print("TF-IDF Results:")
                tfidf_results, vectors = tfidf_engine.search(
                    RecipeData.clean_text(query)
                )
                if print_value:
                    print(tfidf_results)
                results["tfidf"].append({query: tfidf_results["id"].tolist()})
            if "bm25" in args.algorithms:
                if print_value:
                    print("\nBM25 Results:")
                bm25_results = bm25_engine.search(RecipeData.clean_text(query))
                if print_value:
                    print(bm25_results)
                results["bm25"].append({query: bm25_results["id"].tolist()})
            if "embedder" in args.algorithms:
                if print_value:
                    print("\nText Embedder Results:")
                embedder_results = embedder_engine.search(query)
                embedder_results_original = data.df[
                    data.df["id"].isin(embedder_results["id"].tolist())
                ]
                if print_value:
                    print(embedder_results_original)
                results["embedder"].append(
                    {query: embedder_results_original["id"].tolist()}
                )
            if print_value:
                print("\n" + "-" * 50 + "\n")

        for alg in results:
            for query_dict in results[alg]:
                for query, ids in query_dict.items():
                    query_dict[query] = [id for id in ids if id is not None]

    with open("results/tfidf_full_data.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
