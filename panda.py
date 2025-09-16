# ===============================
# Assignment: Data Analysis & Visualization
# ===============================

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# -------------------------------
# Task 1: Load and Explore the Dataset
# -------------------------------

# Option A: Load from CSV
# try:
#     df = pd.read_csv("your_dataset.csv")
# except FileNotFoundError:
#     print("Error: CSV file not found.")

# Option B: Use Iris dataset from sklearn
iris = load_iris(as_frame=True)
df = iris.frame  # DataFrame with features + target
df.rename(columns={"target": "species"}, inplace=True)

print("First 5 rows of dataset:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values per Column:")
print(df.isnull().sum())

# Clean dataset: drop missing values (if any)
df = df.dropna()

# -------------------------------
# Task 2: Basic Data Analysis
# -------------------------------

print("\nBasic Statistics:")
print(df.describe())

# Grouping: Mean values by species
species_means = df.groupby("species").mean()
print("\nMean values grouped by species:")
print(species_means)

# Interesting finding: Which species has the longest average petal length?
longest_petal_species = species_means["petal length (cm)"].idxmax()
print(f"\n Species with longest average petal length: {longest_petal_species}")

# -------------------------------
# Task 3: Data Visualization
# -------------------------------

# 1. Line chart (trend of sepal length across samples)
plt.figure(figsize=(8,5))
plt.plot(df.index, df["sepal length (cm)"], color="blue", label="Sepal Length")
plt.title("Line Chart: Sepal Length Trend")
plt.xlabel("Sample Index")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.grid(True)
plt.show()

# 2. Bar chart (average petal length per species)
plt.figure(figsize=(6,4))
species_means["petal length (cm)"].plot(kind="bar", color="skyblue", edgecolor="black")
plt.title("Bar Chart: Avg Petal Length per Species")
plt.ylabel("Petal Length (cm)")
plt.xlabel("Species")
plt.xticks(rotation=0)
plt.show()

# 3. Histogram (distribution of sepal width)
plt.figure(figsize=(6,4))
plt.hist(df["sepal width (cm)"], bins=15, color="green", edgecolor="black", alpha=0.7)
plt.title("Histogram: Sepal Width Distribution")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter plot (Sepal length vs Petal length, colored by species)
plt.figure(figsize=(7,5))
sns.scatterplot(data=df, x="sepal length (cm)", y="petal length (cm)", hue="species", palette="Set1")
plt.title("Scatter Plot: Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.show()

# -------------------------------
# Findings & Observations
# -------------------------------
print("\n Observations:")
print("- Sepal length generally increases across samples (line chart).")
print("- Iris-virginica has the longest average petal length (bar chart).")
print("- Sepal width is normally distributed around ~3.0 cm (histogram).")
print("- Strong positive correlation between sepal length and petal length, with clear species separation (scatter plot).")
