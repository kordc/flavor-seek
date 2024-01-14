import json
import random
from collections import defaultdict
from enum import Enum

import pandas as pd
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class Split(Enum):
    train = "train"
    valid = "valid"
    test = "test"


class TrainType(Enum):
    text = "text"
    graph = "graph"
    both = "both"


from common import GRAPH_EMB_PATH, QUERIES_PATH, RECIPIES_PATH, SPLIT_PATH


class EmbeddingDataset(Dataset):
    def __init__(
        self,
        split: Split,
        batch_size: int,
        queries: pd.DataFrame,
        recipes: pd.DataFrame,
        graph_embeddings: pd.DataFrame,
        use_text_embeddings: bool = True,
        use_graph_embeddings: bool = True,
    ):
        super().__init__()
        self.use_text_embeddings = use_text_embeddings
        self.use_graph_embeddings = use_graph_embeddings
        with open(SPLIT_PATH) as f:
            splits_with_id = json.load(f)
            self.recipes_id_in_dataset = splits_with_id[split.value]

        # it has recipe_id, list of query embeddings
        self.queries = {
            recipe_id: queries[queries.recipe_id == recipe_id]
            for recipe_id in self.recipes_id_in_dataset
        }

        self.recipes = recipes.loc[self.recipes_id_in_dataset]

        self.graph_embeddings = []
        if use_graph_embeddings:
            self.graph_embeddings = graph_embeddings.loc[self.recipes_id_in_dataset]

        self.n_queries = sum([len(q) for q in self.queries.values()])
        self.n_recipies = max(len(self.recipes), len(self.graph_embeddings))

        self.batch_size = batch_size

        self.return_text = False

        self.overfit_one_batch = False
        self.data_to_overfit = [
            (recipe_num, 0) for recipe_num in range(self.batch_size)
        ]
        # we govern shuffling by ourselfes
        self.reset_dataset()

    def __len__(self):
        return self.n_queries

    def __getitem__(self, idx):
        # Select recipe - these are ordered
        if self.overfit_one_batch:
            recipe_id, selected_query = self.data_to_overfit[idx]
        else:
            recipe_id = self.processed_in_batch % self.n_recipies

        recipe_id = self.recipes_id_in_dataset[recipe_id]

        # Select query that was not used before
        available_queries = self.queries[recipe_id]
        n_queries = len(available_queries)
        selected_query = random.choice(
            list(set(range(n_queries)) - self.already_used_queries[recipe_id])
        )

        selected_query = self.queries[recipe_id].iloc[selected_query]

        # update dataset state
        self.processed_in_batch += 1
        if self.processed_in_batch == self.n_queries and not self.overfit_one_batch:
            self.reset_dataset()

        if self.return_text:
            return {
                "query_texts": selected_query.query,
                "recipe_texts": self.recipes.loc[recipe_id].recipe_merged_info,
                "recipe_embeddings": self.build_recipe_embedding(recipe_id),
                "query_embeddings": torch.Tensor(selected_query.query_embeddings),
            }
        return {
            "recipe_embeddings": self.build_recipe_embedding(recipe_id),
            "query_embeddings": torch.Tensor(selected_query.query_embeddings),
        }

    def build_recipe_embedding(self, recipe_id):
        if self.use_graph_embeddings and self.use_text_embeddings:
            text_emb = self.recipes.loc[recipe_id].recipe_embeddings
            graph_emb = self.graph_embeddings.loc[recipe_id].embedding
            return torch.cat([torch.Tensor(text_emb), torch.Tensor(graph_emb)])
        elif self.use_text_embeddings:
            return torch.Tensor(self.recipes.loc[recipe_id].recipe_embeddings)

        return torch.Tensor(self.graph_embeddings.loc[recipe_id].embedding)

    def reset_dataset(self):
        self.processed_in_batch = 0
        random.shuffle(self.recipes_id_in_dataset)
        self.already_used_queries = defaultdict(set)


class EmbeddingDataModule(LightningDataModule):
    def __init__(self, train_type: TrainType, batch_size=10, num_workers=6):
        super().__init__()
        self.train_type = train_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.is_setup = False

    def setup(self, stage=None):
        if not self.is_setup:
            queries = pd.read_parquet(QUERIES_PATH)
            recipes = pd.read_parquet(RECIPIES_PATH).set_index("id")
            graph_embeddings = pd.read_parquet(GRAPH_EMB_PATH).set_index("recipe_id")

            use_text_embeddings, use_graph_embeddings = False, False
            if self.train_type == TrainType.text:
                use_text_embeddings = True
            elif self.train_type == TrainType.graph:
                use_graph_embeddings = True
            else:
                use_text_embeddings, use_graph_embeddings = True, True

            self.train_subset = EmbeddingDataset(
                Split.train,
                self.batch_size,
                queries,
                recipes,
                graph_embeddings,
                use_text_embeddings,
                use_graph_embeddings,
            )
            self.val_subset = EmbeddingDataset(
                Split.valid,
                self.batch_size,
                queries,
                recipes,
                graph_embeddings,
                use_text_embeddings,
                use_graph_embeddings,
            )
            self.test_subset = EmbeddingDataset(
                Split.test,
                self.batch_size,
                queries,
                recipes,
                graph_embeddings,
                use_text_embeddings,
                use_graph_embeddings,
            )

            self.is_setup = True

    def get_loader(self, dataset, return_text):
        dataset.return_text = return_text
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def train_dataloader(self, return_text=False):
        return self.get_loader(self.train_subset, return_text)

    def val_dataloader(self, return_text=False):
        return self.get_loader(self.val_subset, return_text)

    def test_dataloader(self, return_text=False):
        return self.get_loader(self.test_subset, return_text)

    def log_params(self):
        return {
            "batch_size": self.batch_size,
            "train_samples": len(self.train_subset),
            "val_samples": len(self.val_subset),
            "test_samples": len(self.test_subset),
        }