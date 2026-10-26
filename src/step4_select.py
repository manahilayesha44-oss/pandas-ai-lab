from src.dataset import make_df

df = make_df()

print(df[["name", "salary"]].head(2))
print(df.loc[1, "city"])
print(df.iloc[0:2, 0:2])
print(df[(df.salary > 50) & (df.city == "Karachi")])

top = df.sort_values("salary", ascending=False)
print(top[["name", "salary"]].head(3))

by = df.sort_values(["city", "age"])
print(by[["city", "age"]])
