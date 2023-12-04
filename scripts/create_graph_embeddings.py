"""After training graph embeddings network we can use it to create embeddings for recipes.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from eatpim.rank_subs_in_recipe import content_to_ids, load_embedding_data

# Allows whole dish embedding calculation
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="EaT-PIM/data/recipe_parsed_sm")
parser.add_argument(
    "--model_dir",
    type=str,
    default="models/GraphOps_recipe_parsed_sm_graph_TransE",
    help="Path to model directory, relative w.r.t data_dir",
)
args = parser.parse_args()
main_dir = Path(args.data_dir)
embedding_calculator = calc = load_embedding_data(main_dir, args.model_dir)
with open((main_dir / "ingredient_list.json").resolve(), "r") as f:
    ingredient_list = json.load(f)


# I need mapping from Dish to embedding
recipes_file = (main_dir / "recipe_tree_data.json").resolve()
with open(recipes_file, "r") as f:
    recipe_data = json.load(f)

indices = []
embeddings = []
for dset in ["train.txt", "valid.txt", "test.txt"]:
    with open((main_dir / f"eatpim_triple_data/{dset}").resolve()) as fin:
        for line in fin:
            graph_dict = json.loads(line)
            # there should only be one item in the first depth of this dict
            # the key is the output recipe node, the value is the dictionary representation of the flowgraph
            for k, v in graph_dict.items():
                indices.append(int(k.split("_")[-1]))
                embeddings.append(
                    embedding_calculator.GOpTranseCalcOperation(
                        ops=content_to_ids(v), rem_ing=None
                    )
                )

indices = np.array(indices)
embeddings = np.array(embeddings)
df = pl.DataFrame(data={"recipe_id": indices, "embedding": embeddings})
df.write_parquet("recipe_embeddings.parquet")
