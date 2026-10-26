import pandas as pd
from src.dataset import make_df

df = make_df()

print(df.shape)
print(df.dtypes)
print(df.isna().sum())

med = df["age"].median()
filled = df.fillna({"age": med})
print(med, filled["age"].tolist())
print(len(df.dropna()), "rows left after dropna")

x = pd.DataFrame({"n": ["1", "2", "x"], "c": ["a", "b", "a"]})
x["n"] = pd.to_numeric(x["n"], errors="coerce")
x["c"] = x["c"].astype("category")
print(x.dtypes)
print(x["n"].tolist())

print(df.duplicated().tolist())
print(df.drop_duplicates().shape)
print(df.duplicated(subset=["name"]).sum())
