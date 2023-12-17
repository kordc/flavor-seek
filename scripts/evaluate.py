import json

import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def calculate_metrics(query_embeddings, doc_embeddings, true_indices, k):
    dimension = query_embeddings.shape[1]
    print("Creating index...")
    # The search is performed using a dot product. This is equivalent to cosine similarity if the vectors are normalized.
    index = faiss.IndexFlatIP(dimension)
    print("Adding embeddings to index...")
    # Normalization of embeddings is required for cosine similarity
    faiss.normalize_L2(doc_embeddings)
    index.add(doc_embeddings)

    # Number of neighbors to retrieve
    print(f"Number of neighbors to retrieve: {k}")

    # Search the index
    print("Searching index...")
    # Normalization of embeddings is required for cosine similarity
    faiss.normalize_L2(query_embeddings)
    _, indices = index.search(query_embeddings, k)
    print("Done searching index!")

    # Initialize metrics
    mrr = 0.0
    hits_at_1 = 0
    hits_at_5 = 0
    hits_at_10 = 0

    # Calculate metrics
    print("Calculating metrics...")
    ranks = []
    for i, true_index in enumerate(true_indices):
        retrieved_indices = indices[i]
        if true_index in retrieved_indices:
            rank = np.where(retrieved_indices == true_index)[0][0] + 1
            ranks.append(rank)
            mrr += 1.0 / rank
            if rank == 1:
                hits_at_1 += 1
            if rank <= 5:
                hits_at_5 += 1
            if rank <= 10:
                hits_at_10 += 1
        else:
            ranks.append(k + 1)
    num_queries = len(query_embeddings)
    mrr /= num_queries
    hits_at_1 /= num_queries
    hits_at_5 /= num_queries
    hits_at_10 /= num_queries

    values, counts = np.unique(ranks, return_counts=True)
    plt.bar(values, counts)
    plt.title("Histogram of ranks")
    return mrr, hits_at_1, hits_at_5, hits_at_10


if __name__ == "__main__":
    # Enter your paths, and search size here
    path_to_queries_embeddings = "flavor-seek/data/queries_projected_both2.parquet"
    path_to_recipes_embeddings = "flavor-seek/data/recipes_projected_both2.parquet"
    size_of_search = 200

    queries = pd.read_parquet(path_to_queries_embeddings)
    recipes = pd.read_parquet(path_to_recipes_embeddings)

    with open("flavor-seek/data/recipe_id_splits.json") as f:
        splits_with_id = json.load(f)
        recipes_id_in_dataset = splits_with_id["valid"]

    queries = queries[queries["recipe_id"].isin(recipes_id_in_dataset)]

    query_embeddings = queries["query_embeddings"].values
    recipe_embeddings = recipes["embedding"].values

    query_embeddings = np.vstack(query_embeddings)
    recipe_embeddings = np.vstack(recipe_embeddings)

    id_to_index = {recipes.iloc[i]["recipe_id"]: i for i in range(len(recipes))}

    true_indices = []
    for i in range(len(queries)):
        true_indices.append(id_to_index[queries.iloc[i]["recipe_id"]])

    mrr, hits1, hits5, hits10 = calculate_metrics(
        query_embeddings.astype(np.float32),
        recipe_embeddings.astype(np.float32),
        true_indices,
        size_of_search,
    )
    print("MRR: ", mrr)
    print("Hits@1: ", hits1)
    print("Hits@5: ", hits5)
    print("Hits@10: ", hits10)
