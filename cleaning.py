from pathlib import Path
import pandas as pd
import re

#Cleaning step 1: Standardise formats

DATA_PATH = Path("amazon_reviews.csv")
OUT_DIR = Path("outputs/cleaning")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

df["at_parsed"] = pd.to_datetime(df["at"], errors="coerce")
df["score"] = pd.to_numeric(df["score"], errors="coerce")
df["thumbsUpCount"] = pd.to_numeric(df["thumbsUpCount"], errors="coerce")

print("Shape:", df.shape)
print("Invalid dates:", df["at_parsed"].isna().sum())
print("Invalid scores:", df["score"].isna().sum())
print("Invalid thumbsUpCount:", df["thumbsUpCount"].isna().sum())

#Cleaning Step 2: Handle Missing Values

start_rows = len(df)

# Make missing truly missing first
df["content"] = df["content"].astype("string")

# Drop real missing
df = df[df["content"].notna()]

# Drop empty/whitespace
df = df[df["content"].str.strip().ne("")]

# Drop “fake missing” strings
df = df[~df["content"].str.strip().str.lower().isin(["nan", "none", "null"])]

rows_dropped_content = start_rows - len(df)

# Fill missing userName
df["userName"] = df["userName"].fillna("Unknown")

# Fill missing version-related fields
df["reviewCreatedVersion"] = df["reviewCreatedVersion"].fillna("Unknown")
df["appVersion"] = df["appVersion"].fillna("Unknown")

print("Rows dropped due to missing content:", rows_dropped_content)
print("Shape after Step 2:", df.shape)

# Cleaning Step 3: Handle duplicate reviewId

before = len(df)

# Sort by time so the most recent review is kept
df = df.sort_values("at_parsed")

# Remove duplicate reviewId, keep the most recent
df = df.drop_duplicates(subset="reviewId", keep="last")

removed_duplicates = before - len(df)

print("Duplicate reviewId removed:", removed_duplicates)
print("Shape after Step 3:", df.shape)

# Cleaning Step 4: Light text cleaning (no meaning change)

def light_clean_text(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)   # collapse multiple spaces/newlines into one space
    return s

before = len(df)

df["content_clean"] = df["content"].apply(light_clean_text)

# Drop rows that become empty after trimming (should be very small)
df = df[df["content_clean"].str.len() > 0].copy()

dropped_after_clean = before - len(df)

print("Rows dropped after light text cleaning:", dropped_after_clean)
print("Shape after Step 4:", df.shape)

# Cleaning Step 5: Create helper columns

df["review_len_chars"] = df["content_clean"].str.len()
df["review_len_words"] = df["content_clean"].apply(lambda x: len(x.split()))
df["has_link"] = df["content_clean"].str.lower().apply(
    lambda x: int("http://" in x or "https://" in x or "www." in x)
)

print("Shape after Step 5:", df.shape)
print(df[["review_len_chars", "review_len_words", "has_link"]].head())

# Cleaning Step 6: Save cleaned dataset and summary

OUT_CLEAN_PATH = Path("amazon_reviews_cleaned.csv")

final_columns = [
    "reviewId",
    "userName",
    "score",
    "thumbsUpCount",
    "at_parsed",
    "reviewCreatedVersion",
    "appVersion",
    "content",
    "content_clean",
    "review_len_chars",
    "review_len_words",
    "has_link"
]

df_final = df[final_columns].copy()
df_final.to_csv(OUT_CLEAN_PATH, index=False)

# Save cleaning summary
summary = {
    "rows_original": 82234,
    "rows_after_cleaning": len(df_final),
    "rows_removed_total": 82234 - len(df_final),
    "final_columns": len(df_final.columns)
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv(OUT_DIR / "cleaning_summary.csv", index=False)

