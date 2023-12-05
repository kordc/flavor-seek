import pandas as pd


def get_recipes_from_a_split(df: pd.DataFrame, ids: list) -> pd.DataFrame:
    return df.loc[ids]
