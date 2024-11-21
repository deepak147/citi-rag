from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from datasets import Dataset
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import ast
from ragas.metrics import (
    Faithfulness,
    LLMContextRecall,
    FactualCorrectness,
    SemanticSimilarity,
)
from ragas import evaluate
from backend.retrieve import retrieve

df = pd.read_csv("testset.csv")
df["contexts"] = df["contexts"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)
df["response"] = df["question"].apply(lambda x: retrieve(x)["answer"])

eval_dataset = Dataset.from_pandas(df)
print(eval_dataset)

evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

metrics = [
    LLMContextRecall(llm=evaluator_llm),
    FactualCorrectness(llm=evaluator_llm),
    Faithfulness(llm=evaluator_llm),
    SemanticSimilarity(embeddings=evaluator_embeddings),
]
results = evaluate(dataset=eval_dataset, metrics=metrics, llm=evaluator_llm)

df = results.to_pandas().to_csv("evaluation_results.csv")

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

output_file_path = "heatmap_plot.png"
plt.savefig(output_file_path)
plt.close()
