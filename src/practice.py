import pandas as pd


def t1_1_weather_series(df):
    """Return the temp_c column as a Series whose index is the day names."""
    return df.set_index("day")["temp_c"]


def t1_2_hottest_day(df):
    """Return the NAME of the day with the highest temp_c."""
    return df.loc[df["temp_c"].idxmax(), "day"]


def t1_3_build_frame():
    """item = ['pen', 'Ink', 'bag'] and price = [10, 120, 800]. Return it."""
    return pd.DataFrame({"item": ["pen", "Ink", "bag"], "price": [1, 12, 8]})


def t1_4_series_vs_frame(df):
    return (type(df["temp_c"]).__name__, type(df[["temp_c"]]).__name__)


def t3_1_count_missing(df):
    """Return a dict of missing-value counts per column."""
    return df.isna().sum().to_dict()


def t3_2_math_numeric(df):
    """Convert the math column to numbers. Return the values as a list."""
    return pd.to_numeric(df["math"], errors="coerce").tolist()


def t3_3_remove_duplicates(df):
    """Remove duplicate rows and return the new shape."""
    return df.drop_duplicates().shape


def t3_4_fill_english_median(df):
    clean = df.drop_duplicates().copy()
    med = clean["english"].median()
    return clean["english"].fillna(med).tolist()


def t3_5_rows_after_dropna(df):
    clean = df.drop_duplicates().copy()
    clean["math"] = pd.to_numeric(clean["math"], errors="coerce")
    return len(clean.dropna())


def t3_6_section_category(df):
    return str(df["section"].astype("category").dtype)


def t3_7_clean_students(df):
    clean = df.drop_duplicates().copy()
    clean["math"] = pd.to_numeric(clean["math"], errors="coerce")
    math_mean = clean["math"].mean()
    eng_median = clean["english"].median()
    clean["math"] = clean["math"].fillna(math_mean)
    clean["english"] = clean["english"].fillna(eng_median)
    return clean


def t4_1_cheap_products(df):
    """Return the product names with price below 50, as a list."""
    return df[df["price"] < 50]["product"].tolist()


def t4_2_accessories_in_stock(df):
    """Return the names of 'Accessories' products that are in stock."""
    return df[(df["category"] == "Accessories") & (df["stock"] > 0)]["product"].tolist()


def t4_3_top_two_by_price(df):
    return df.sort_values("price", ascending=False).head(2)["product"].tolist()


def t4_4_loc_price(df):
    """Using loc, return the price in the row whose index label is 3."""
    return df.loc[3, "price"]


def t4_5_iloc_block(df):
    return df.iloc[0:2, 0:2].values.tolist()


def t4_6_low_stock_or_premium(df):
    """Return the names of products with stock below 6 OR price above 500."""
    return df[(df["stock"] < 6) | (df["price"] > 500)]["product"].tolist()


def t4_7_sort_two_keys(df):
    return df.sort_values(["category", "price"], ascending=[True, False])["product"].tolist()


def t5_1_stock_by_category(products):
    """Return a dict: total stock for each category."""
    return products.groupby("category")["stock"].sum().to_dict()


def t5_2_avg_price_by_category(products):
    """Return a dict: average price per category, rounded to 2 decimals."""
    return products.groupby("category")["price"].mean().round(2).to_dict()


def t5_3_category_summary(products):
    return products.groupby("category").agg(n=("product_id", "count"), max_price=("price", "max")).to_dict("index")


def t5_4_left_merge_names(orders, products):
    merged = orders.merge(products, on="product_id", how="left")
    return merged["product"].tolist()


def t5_5_inner_merge_rows(orders, products):
    """Inner-merge orders with products on product_id. Return the number of rows."""
    merged = orders.merge(products, on="product_id", how="inner")
    return len(merged)


def t5_6_revenue_list(orders, products):
    """Inner-merge, then return the revenue of each order (qty * price) as a list."""
    merged = orders.merge(products, on="product_id", how="inner")
    return (merged["qty"] * merged["price"]).tolist()


def t5_7_revenue_by_customer(orders, products):
    """Inner-merge, then return a dict: total revenue per customer."""
    merged = orders.merge(products, on="product_id", how="inner")
    merged["revenue"] = merged["qty"] * merged["price"]
    return merged.groupby("customer")["revenue"].sum().to_dict()


def t5_8_price_label(products):
    def label_price(price):
        if price < 50:
            return "cheap"
        elif price < 500:
            return "mid"
        else:
            return "premium"
    return products["price"].apply(label_price).tolist()


def t5_9_upper_names(products):
    """Return the product names in UPPER CASE (use a str method), as a list."""
    return products["product"].str.upper().tolist()
