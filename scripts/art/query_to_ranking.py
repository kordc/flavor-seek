import faiss
import pandas as pd
import numpy as np
import torch
from base_model import EmbeddingHead
from transformers import AutoModel
import argparse
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import string
from dataset import TrainType


class RecipeSearchEngine:
    def __init__(self, alg_list):
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        models_paths = ["../../models/just_text_model.ckpt", "../../models/graph_model.ckpt", "../../models/model_combination.ckpt"]
        self.text_embedder = AutoModel.from_pretrained(
                "jinaai/jina-embeddings-v2-small-en", trust_remote_code=True
            )
        self.text_embedder.eval()
        self.models = {}
        self.embeddings = pd.read_parquet("../../data/recipe_embeddings_all.parquet")

        if "text" in alg_list:
            text_embeddings = self.embeddings["text_embeddings"].values
            text_embeddings = np.vstack(text_embeddings).astype(np.float32)
            dimension = text_embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            faiss.normalize_L2(text_embeddings)
            index.add(text_embeddings)

            model = EmbeddingHead(TrainType.text)
            checkpoint = torch.load(models_paths[0], map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['state_dict'])

            self.models["text"] = (model, index)

        if "graph" in alg_list:
            graph_embeddings = self.embeddings["graph_embeddings"].values
            graph_embeddings = np.vstack(graph_embeddings).astype(np.float32)
            dimension = graph_embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            faiss.normalize_L2(graph_embeddings)
            index.add(graph_embeddings)

            model = EmbeddingHead(TrainType.graph)
            checkpoint = torch.load(models_paths[1], map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['state_dict'])

            self.models["graph"] = (model, index)

        if "text+graph" in alg_list:
            concat_embeddings = self.embeddings["combined_embeddings"].values
            concat_embeddings = np.vstack(concat_embeddings).astype(np.float32)
            dimension = concat_embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            faiss.normalize_L2(concat_embeddings)
            index.add(concat_embeddings)

            model = EmbeddingHead(TrainType.both)
            checkpoint = torch.load(models_paths[2], map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['state_dict'])

            self.models["both"] = (model, index)

        self.df = pd.read_csv("../../data/dataframe.csv")
        self.df = self.df.sort_values(by=['id'])

    def clean_text(self, text):
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

    def get_query_embedding(self, query):
        with torch.no_grad():
            embedding = self.text_embedder.encode(query)
        return embedding
    


    def search(self, query, topk=10):
        query = self.clean_text(query)
        query_embedding = self.get_query_embedding(query)
        query_embedding = torch.from_numpy(query_embedding)
        for model in self.models:
            model, index = self.models[model]
            model.eval()
            with torch.no_grad():
                embedding = model.model_query(query_embedding)
            embedding = embedding.numpy()
            embedding = embedding.astype(np.float32)
            embedding = np.expand_dims(embedding, axis=0)
            faiss.normalize_L2(embedding) # to chyba nic nie robi gdy jest tylko jeden embedding
            _, indices = index.search(embedding, topk)
            print(f"Ranking for {model.train_type.name}:")
            for i in range(topk):
                index = indices[0][i]
                print(f"{i+1}. {self.df.iloc[index]['name']}")
            print("")
            

def main():
    parser = argparse.ArgumentParser(description="Recipe Search Engine")
    parser.add_argument(
        "--algorithm",
        type=str,
        nargs="+",
        choices=["text", "graph", "text+graph"],
        default=["text", "graph", "text+graph"],
        help="Algorithm type (default: text)",
    )
    # parser.add_argument(
    #     "--query",
    #     type=str,
    #     default="",
    #     help="Query to search for (default: '')",
    # )
    args = parser.parse_args()
    print(f"Algorithms choosen: {args.algorithm}")
    engine = RecipeSearchEngine(args.algorithm)
    while True:
        args.query = input("Enter query: ")
        if args.query == "exit":
            break
        print(f"Query: {args.query}")
        engine.search(args.query)
    


if __name__ == "__main__":
    main()