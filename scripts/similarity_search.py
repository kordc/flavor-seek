import argparse
import gc
import json
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
    def __init__(self, recipe_data, max_features=1000, save_vectors=False):
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
        tfidf_matrix = self.load_or_create_tfidf_matrix()
        self.save_vectors_to_parquet_file(
            tfidf_matrix.toarray(), "data/tfidf_recipes_matrix.parquet"
        )
        self.prepare_index(tfidf_matrix)

    def search(self, query, top_n=10):
        query_vector = self.vectorizer.transform([query]).toarray().astype("float32")
        self.save_vectors_to_parquet_file(
            query_vector, "data/tfidf_eval_queries_matrix.parquet"
        )
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
    with open(file_path, "r") as file:
        queries = file.readlines()
    return [query.strip() for query in queries]


def main():
    parser = argparse.ArgumentParser(description="Recipe Search Engine")
    parser.add_argument(
        "--algorithms",
        type=str,
        nargs="+",
        choices=["tfidf", "bm25", "rag"],
        default=["tfidf", "bm25", "rag"],
        help="Algorithm type (default: tfidf)",
    )
    args = parser.parse_args()

    queries = read_queries("data/evaluation.txt")

    data = RecipeData(
        "data/raw_recipes_used.csv", "data/raw_recipes_used_preprocessed.csv"
    )
    data.load_data()
    cleaned_queries = [RecipeData.clean_text(query) for query in queries]

    if "tfidf" in args.algorithms:
        tfidf_engine = TfIdfSearchEngine(data, save_vectors=True)
        tfidf_engine.prepare()
    if "bm25" in args.algorithms:
        bm25_engine = BM25SearchEngine(data)
        bm25_engine.prepare()
    if "rag" in args.algorithms:
        rag_engine = TextEmbedderSearchEngine(
            "jinaai/jina-embeddings-v2-small-en", "/tmp/recipe_store_cli", "recipies"
        )
        rag_engine.preprocess_and_upload(data.df)

    results = defaultdict(list)

    for query in cleaned_queries:
        print(f"Query: {query}")
        if "tfidf" in args.algorithms:
            print("TF-IDF Results:")
            tfidf_results = tfidf_engine.search(query)
            print(tfidf_results)
            results["tfidf"].append({query: tfidf_results["id"].tolist()})
        if "bm25" in args.algorithms:
            print("\nBM25 Results:")
            bm25_results = bm25_engine.search(query)
            print(bm25_results)
            results["bm25"].append({query: bm25_results["id"].tolist()})
        if "rag" in args.algorithms:
            print("\nText Embedder Results:")
            rag_results = rag_engine.search(query)
            rag_results_original = data.df[
                data.df["id"].isin(rag_results["id"].tolist())
            ]
            print(rag_results_original)
            results["rag"].append({query: rag_results_original["id"].tolist()})
        print("\n" + "-" * 50 + "\n")

    for alg in results:
        for query_dict in results[alg]:
            for query, ids in query_dict.items():
                query_dict[query] = [id for id in ids if id is not None]

    with open("results/baselines_evaluation.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
