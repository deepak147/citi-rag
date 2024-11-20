import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd

file_path = 'evaluation_results.csv'
df = pd.read_csv(file_path)

heatmap_data = df[
    [
        "semantic_similarity",
        "context_recall",
        "faithfulness",
        "factual_correctness",
    ]
]

cmap = LinearSegmentedColormap.from_list("green_red", ["red", "green"])

plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", linewidths=0.5, cmap=cmap)

plt.yticks(ticks=range(len(df["user_input"])), labels=df["user_input"], rotation=0)

output_file_path = 'heatmap_plot.png'
plt.savefig(output_file_path)
plt.close()