import pyarrow.parquet as pq
import numpy as np
import faiss


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
    for i, true_index in enumerate(true_indices):
        retrieved_indices = indices[i]
        if true_index in retrieved_indices:
            rank = np.where(retrieved_indices == true_index)[0][0] + 1
            mrr += 1.0 / rank
            if rank == 1:
                hits_at_1 += 1
            if rank <= 5:
                hits_at_5 += 1
            if rank <= 10:
                hits_at_10 += 1
    num_queries = len(query_embeddings)
    mrr /= num_queries
    hits_at_1 /= num_queries
    hits_at_5 /= num_queries
    hits_at_10 /= num_queries

    return mrr, hits_at_1, hits_at_5, hits_at_10


if __name__ == "__main__":
    # Enter your paths, and search size here
    path_to_queries_embeddings = "../data/query_embeddings.parquet"
    path_to_recipes_embeddings = "../data/recipe_embeddings.parquet"
    size_of_search = 1000

    queries = pq.read_table(path_to_queries_embeddings).to_pandas()
    recipes = pq.read_table(path_to_recipes_embeddings).to_pandas()

    query_embeddings = queries["query_embeddings"].values
    recipe_embeddings = recipes["recipe_embeddings"].values

    query_embeddings = np.vstack(query_embeddings)
    recipe_embeddings = np.vstack(recipe_embeddings)

    id_to_index = {recipes.iloc[i]["id"]: i for i in range(len(recipes))}

    true_indices = []
    for i in range(len(queries)):
        true_indices.append(id_to_index[queries.iloc[i]["id"]])

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
