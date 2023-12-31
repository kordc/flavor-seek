import argparse
import json
from collections import defaultdict

import pandas as pd


def main(data_dir: str, raw_recipe_path: str):
    recipes_in_split = defaultdict(list)
    all_recipies_used = []
    for split in ["train", "valid", "test"]:
        with open(f"{data_dir}/{split}.txt", "r") as f:
            for line in f:
                recipe = json.loads(line)
                for k, v in recipe.items():
                    recipe_id = int(k.split("_")[-1])
                    recipes_in_split[split].append(recipe_id)
                    all_recipies_used.append(recipe_id)

    with open("recipe_id_splits.json", "w") as f:
        json.dump(recipes_in_split, f, indent=4)

    raw_recipes = pd.read_csv(raw_recipe_path, index_col="id")
    raw_recipes_used = raw_recipes.loc[all_recipies_used]
    raw_recipes_used.to_csv("raw_recipes_used.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Path to data directory generated by EaT-PIM with test.txt, train.txt and val.txt",
    )
    parser.add_argument(
        "--raw_recipe_path",
        type=str,
        default="RAW_recipes.csv",
        help="Path to RAW_recipes.csv file downloaded from here https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions",
    )

    args = parser.parse_args()
    main(args.data_dir, args.raw_recipe_path)
