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
