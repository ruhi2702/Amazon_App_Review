import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Step 1: Dataset Overview

# Read the Amazon reviews dataset
df = pd.read_csv("amazon_reviews.csv")

# Display data types and non-null counts for each column
print(df.info())

# Step 2: Missing Values Check
missing_values = df.isnull().sum()
missing_percent = (missing_values / len(df)) * 100

missing_summary = (
    pd.DataFrame({
        "Missing_Count": missing_values,
        "Missing_Percent": missing_percent.round(2)
    })
    .sort_values(by="Missing_Count", ascending=False)
)

missing_summary.to_csv("outputs/tables/missing_values_summary.csv")

print("\n MISSING VALUES SUMMARY")
print(missing_summary)


# Step 3: Duplicate Check

# Duplicate reviewId
dup_review_id = df.duplicated(subset="reviewId").sum()
print("\nDuplicate reviewId count:", dup_review_id)

# Duplicate review text
dup_content = df.duplicated(subset="content").sum()
print("Duplicate review text (content) count:", dup_content)

# Step 4: Rating (Score) Analysis

rating_counts = df["score"].value_counts().sort_index()
rating_percent = (rating_counts / len(df) * 100).round(2)

print("\nRATING COUNTS")
print(rating_counts)

print("\nRATING PERCENTAGES")
print(rating_percent)

# Save tables for report/group use
rating_counts.to_csv("outputs/tables/rating_counts.csv")
rating_percent.to_csv("outputs/tables/rating_percentages.csv")

# Plot rating distribution

plt.figure()
rating_counts.plot(kind="bar")
plt.title("Distribution of Review Ratings (1–5)")
plt.xlabel("Rating Score")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.savefig("outputs/figures/rating_distribution.png", dpi=300)
plt.close()

# Step 5: Thumbs-Up Analysis
# Basic statistics
thumbs_stats = df["thumbsUpCount"].describe()
print("\nTHUMBS-UP BASIC STATISTICS")
print(thumbs_stats)

# Save stats
thumbs_stats.to_csv("outputs/tables/thumbsUp_stats.csv")

# Distribution plot
plt.figure()
plt.hist(df["thumbsUpCount"], bins=50)
plt.title("Distribution of Thumbs-Up Counts")
plt.xlabel("thumbsUpCount")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("outputs/figures/thumbsUp_distribution.png", dpi=300)
plt.close()

# Helpful votes by rating (median is more robust than mean)
thumbs_by_score = df.groupby("score")["thumbsUpCount"].median()
print("\nMEDIAN THUMBS-UP BY SCORE")
print(thumbs_by_score)

thumbs_by_score.to_csv("outputs/tables/thumbsUp_median_by_score.csv")

# Step 6: Time Analysis
# Convert 'at' column to datetime (do NOT drop rows)
df["at_parsed"] = pd.to_datetime(df["at"], errors="coerce")

# Count invalid dates
invalid_dates = df["at_parsed"].isna().sum()
print("\nInvalid / unparseable dates:", invalid_dates)

# Date range (ignoring NaT automatically)
print("\nDATE RANGE")
print("Earliest review date:", df["at_parsed"].min())
print("Latest review date:", df["at_parsed"].max())

# Reviews per month (NaT values automatically excluded by resample)
reviews_per_month = (
    df
    .set_index("at_parsed")
    .resample("ME")["reviewId"]
    .count()
)
# Plot
plt.figure()
reviews_per_month.plot()
plt.title("Number of Reviews Over Time (Monthly)")
plt.xlabel("Time")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.savefig("outputs/figures/reviews_over_time.png", dpi=300)
plt.close()
# Save table
reviews_per_month.to_csv("outputs/tables/reviews_per_month.csv")
