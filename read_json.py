import pandas as pd

df = pd.read_csv("data/queries.csv")
print(df["llm_output"].iloc[0])
processed_data = [
    line.split(". ", 1)[1]
    for line in df["llm_output"].iloc[0].strip().split("\n")
    if line
]
print(processed_data)
