import argparse

import faiss
import numpy as np
import pandas as pd
import torch
from base_model import EmbeddingHead
from dataset import TrainType
from transformers import AutoModel


class RecipeSearchEngine:
    def __init__(self, alg_list):
        self.initialize_models(alg_list)
        self.load_data()

    def initialize_models(self, alg_list):
        models_paths = {
            "text": "../../models/just_text_model.ckpt",
            "graph": "../../models/graph_model.ckpt",
            "both": "../../models/model_combination.ckpt",
        }

        self.text_embedder = AutoModel.from_pretrained(
            "jinaai/jina-embeddings-v2-small-en", trust_remote_code=True
        ).eval()

        self.models = {}
        embeddings = pd.read_parquet("../../data/recipe_embeddings_all.parquet")
        embeddings.rename(
            columns={"combined_embeddings": "both_embeddings"}, inplace=True
        )

        for alg in alg_list:
            if alg in models_paths:
                self.models[alg] = self.create_model(alg, embeddings, models_paths[alg])

    def load_data(self):
        try:
            self.df = pd.read_csv("../../data/dataframe.csv").sort_values(by=["id"])
        except FileNotFoundError:
            print("Error: Data file not found.")
            self.df = None

    def create_model(self, alg_type, embeddings, model_path):
        # try:
        alg_embeddings = np.vstack(embeddings[f"{alg_type}_embeddings"].values).astype(
            np.float32
        )
        dimension = alg_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(alg_embeddings)
        index.add(alg_embeddings)

        model = EmbeddingHead(TrainType[alg_type])
        checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["state_dict"])

        return model, index

    # except Exception as e:
    #     print(f"Error loading model for {alg_type}: {e}")
    #     return None

    def get_query_embedding(self, query):
        with torch.no_grad():
            embedding = self.text_embedder.encode(query)
        return embedding

    def search(self, query, topk=10):
        if self.df is None:
            return {}

        query_embedding = self.get_query_embedding(query)
        query_embedding = torch.from_numpy(query_embedding)
        results_dict = {}

        for alg, (model, index) in self.models.items():
            if model is None:
                continue

            model.eval()
            with torch.no_grad():
                embedding = model.model_query(query_embedding)
            embedding = np.expand_dims(embedding.numpy().astype(np.float32), axis=0)
            faiss.normalize_L2(embedding)
            _, indices = index.search(embedding, topk)

            results = self.extract_results(indices, topk)
            results_dict[alg] = results

        return results_dict

    def extract_results(self, indices, topk):
        results = []
        for i in range(topk):
            index = indices[0][i]
            result = {
                "name": self.df.iloc[index]["name"],
                "id": self.df.iloc[index]["id"],
                "description": self.df.iloc[index]["description"],
                "ingredients": self.df.iloc[index]["ingredients"],
                "steps": self.df.iloc[index]["steps"],
            }
            results.append(result)
        return results


def main():
    parser = argparse.ArgumentParser(description="Recipe Search Engine")
    parser.add_argument(
        "--algorithm",
        type=str,
        nargs="+",
        choices=["text", "graph", "both"],
        default=["text"],
        help="Algorithm type (default: text)",
    )

    args = parser.parse_args()
    alg_name = args.algorithm
    print(f"Algorithms chosen: {', '.join(args.algorithm)}")
    engine = RecipeSearchEngine(args.algorithm)

    while True:
        query = input("Enter query or type 'exit' to exit: ").strip()
        if query.lower() == "exit":
            break

        results = engine.search(query)
        if not results:
            print("No results found.")
            continue

        print(f"Query: {query}")
        if isinstance(alg_name, str):
            for index, result in enumerate(results[alg_name], start=1):
                print(f"{index}. {result['name']}")
        else:
            for alg in alg_name:
                print(f"Algorithm: {alg}")
                for index, result in enumerate(results[alg], start=1):
                    print(f"{index}. {result['name']}")
                print("-" * 72)

        while True:
            detail_input = input(
                "Enter ranking number for details, 'exit' to exit, or press enter to continue: "
            ).strip()
            if detail_input.lower() == "exit" or not detail_input:
                break

            try:
                result_index = int(detail_input) - 1
                if result_index < 0 or result_index >= len(results):
                    raise ValueError

                result = results[result_index]
                print(f"Name: {result['name']}\n")
                print(f"Description: {result['description']}\n")
                print(f"Ingredients: {result['ingredients']}\n")
                print(f"Steps: {result['steps']}\n")
                print("-" * 72)
            except ValueError:
                print("Invalid input. Please enter a valid number.")


if __name__ == "__main__":
    main()
