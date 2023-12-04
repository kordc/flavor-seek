# flavor-seek


To start working please do

```sh
    git clone https://github.com/kordc/flavor-seek.git
    cd flavor-seek
    git submodule init
    git submodule update
```


As you modify EaT-PIM Please make sure to always commit and push the submodule and parent repository.

If you want to generate train/test based on EaT-PIM splits. Remember to download RAW_recipes.csv from [here](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)
```sh
cd flavor-seek
python .\scripts\generate_splits.py --data_dir .\EaT-PIM\data\recipe_parsed_sm\triple_data\ --raw_recipe_path RAW_recipes.csv
```

a json file with list of ids will be ganarated and smaller `csv` file that contains only used recipes.

If you want to generate new graph embeddings

```sh
cd flavor-seek
python .\scripts\create_graph_embeddings.py
```

a parquet file with 2 columns: `id` and `embedding` will be created. `id` corresponds to raw_recipes_used.csv field.


You can use `utils.get_recipes_from_a_split` to generate splits.

```python
df = pd.read_csv("data/raw_recipes_used.csv")
with open("data/recipe_id_splits.json", "r") as f:
    splits = json.load(f)

df_train = get_recipes_from_a_split(df, splits['train'])
```