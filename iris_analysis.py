import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset from your local CSV
df = pd.read_csv("iris.csv")

# Step 1: Display basic info
print("\nFirst 5 rows of dataset:")
print(df.head())

print("\nDataset info:")
print(df.info())

print("\nSummary statistics:")
print(df.describe())

# Step 2: Visualization - Pairplot
sns.pairplot(df, hue="species")
plt.savefig("iris_pairplot.png")  # Save the plot
plt.show()

# Step 3: Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Iris Dataset")
plt.savefig("iris_heatmap.png")
plt.show()
