from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from datasets import Dataset
import pandas as pd
import ast
from ragas.metrics import Faithfulness, LLMContextRecall, FactualCorrectness, SemanticSimilarity
from ragas import evaluate
from backend.retrieve import retrieve

df = pd.read_csv("testset.csv")
df['contexts'] = df['contexts'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
df['response'] = df['question'].apply(lambda x: retrieve(x)["answer"])

eval_dataset = Dataset.from_pandas(df)
print(eval_dataset)

evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

metrics = [
    LLMContextRecall(llm=evaluator_llm), 
    FactualCorrectness(llm=evaluator_llm), 
    Faithfulness(llm=evaluator_llm),
    SemanticSimilarity(embeddings=evaluator_embeddings)
]
results = evaluate(dataset=eval_dataset, metrics=metrics, llm=evaluator_llm)

df = results.to_pandas().to_csv("evaluation_results.csv")