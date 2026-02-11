"""
Evaluation script for generating CSV files with LLM answers and RAGAS metrics.

This script runs a full RAG evaluation pipeline on the IT-Grundschutz question-answer dataset,
generating both a CSV with all results and a README documenting the experiment configuration.

Usage from Jupyter notebook:
    from scripts.run_evaluation import generate_evaluation_results
    
    generate_evaluation_results(
        llm="openai/gpt-oss-120b",
        embedding_model="openai/octen-embedding-8b",
        input_data_description="XML Kompendium 2023, character-based chunking (4000 chars, 200 overlap)",
        chunk_size=4000,
        chunk_overlap=200,
        top_k=5,
        output_name="gpt-oss_kompendium-xml",
        temperature=0.2,
    )
"""
from __future__ import annotations

import asyncio
import os
import random
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd

# Ensure notebooks/ is importable for litellm_client
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Load .env from notebooks/ directory
try:
    from dotenv import load_dotenv
    env_path = NOTEBOOKS_DIR / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

if str(NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS_DIR))

from litellm_client import (
    LLMConfig,
    VectorDBConfig,
    chat_completion,
    get_embeddings,
    get_qdrant_client,
    load_llm_config,
    load_vectordb_config,
)


@dataclass
class EvaluationConfig:
    """Configuration for the evaluation run."""
    
    # Required parameters
    llm: str
    embedding_model: str
    input_data_description: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    output_name: str
    temperature: float
    
    # Optional parameters with defaults
    seed: int = 42
    eval_csv_path: str = "data/data_evaluation/GSKI_Fragen-Antworten-Fundstellen.csv"


def _set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


def _retrieve_contexts(
    question: str,
    top_k: int,
    client,
    collection_name: str,
    llm_cfg: LLMConfig,
) -> List[str]:
    """Retrieve top-k context chunks from Qdrant for a given question."""
    query_emb = get_embeddings([question], llm_cfg, batch_size=1)[0]
    results = client.query_points(
        collection_name=collection_name,
        query=query_emb,
        limit=top_k,
    ).points
    return [res.payload.get("text", "") for res in results]


def _build_messages(
    question: str,
    contexts: List[str],
    fewshot: List[dict],
) -> List[dict]:
    """Build chat messages for the LLM with few-shot examples and context."""
    context_text = "\n\n".join(contexts)
    messages = [
        {
            "role": "system",
            "content": (
                "Beantworte die Frage kurz, präzise und nutze ausschließlich den gelieferten Kontext! "
                "Antworte auf Deutsch. Die Antwort sollte maximal 2 Sätze lang sein."
            ),
        },
    ]

    for ex in fewshot:
        messages.append({
            "role": "user",
            "content": f"Frage: {ex['question']}\n\nKontext:\n<BEISPIEL-KONTEXT>",
        })
        messages.append({"role": "assistant", "content": ex["answer"]})

    messages.append({
        "role": "user",
        "content": f"Frage: {question}\n\nKontext:\n{context_text}",
    })
    return messages


def _generate_answer(
    messages: List[dict],
    llm_cfg: LLMConfig,
    temperature: float,
    seed: int,
) -> str:
    """Generate an answer using the LLM."""
    response = chat_completion(messages, llm_cfg, temperature=temperature, seed=seed)
    
    # Extract content from response
    if isinstance(response, dict):
        return response["choices"][0]["message"]["content"]
    if hasattr(response, "choices"):
        return response.choices[0].message.content
    raise TypeError(f"Unexpected response type: {type(response)}")


async def _score_row_async(
    row: dict,
    scorers: dict,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Score a single row with all RAGAS metrics asynchronously."""
    async with semaphore:
        try:
            context_precision = await scorers["context_precision"].ascore(
                user_input=row["question"],
                reference=row["ground_truth_context"],
                retrieved_contexts=row["contexts"],
            )
            context_recall = await scorers["context_recall"].ascore(
                user_input=row["question"],
                reference=row["ground_truth_context"],
                retrieved_contexts=row["contexts"],
            )
            faithfulness = await scorers["faithfulness"].ascore(
                user_input=row["question"],
                response=row["answer"],
                retrieved_contexts=row["contexts"],
            )
            answer_correctness = await scorers["answer_correctness"].ascore(
                user_input=row["question"],
                response=row["answer"],
                reference=row["ground_truth_answer"],
            )
            
            return {
                "context_precision": context_precision.value,
                "context_recall": context_recall.value,
                "faithfulness": faithfulness.value,
                "answer_correctness": answer_correctness.value,
            }
        except Exception as e:
            raise RuntimeError(
                f"RAGAS evaluation failed for question: '{row['question'][:100]}...'\n"
                f"Error: {type(e).__name__}: {e}\n\n"
                f"Please check:\n"
                f"  1. LLM API is accessible and responding\n"
                f"  2. The model supports the required API format\n"
                f"  3. Context and answer are not empty\n"
                f"  4. Network connectivity is stable"
            ) from e


async def _run_ragas_evaluation(
    records: List[dict],
    llm_cfg: LLMConfig,
    temperature: float,
    concurrency: int = 10,
) -> List[dict]:
    """Run RAGAS evaluation on all records."""
    # Import RAGAS dependencies
    try:
        from ragas.llms import llm_factory
        from ragas.embeddings.litellm_provider import LiteLLMEmbeddings
        from ragas.metrics.collections import (
            ContextPrecision,
            ContextRecall,
            Faithfulness,
            AnswerCorrectness,
        )
        import instructor
        import litellm
    except ImportError as e:
        raise ImportError(
            "RAGAS dependencies not installed. Please install with:\n"
            "  pip install ragas instructor litellm datasets"
        ) from e

    # Configure LiteLLM
    litellm.api_base = llm_cfg.api_base
    litellm.api_key = llm_cfg.api_key

    # Create RAGAS LLM and embeddings
    client = instructor.from_litellm(litellm.acompletion, mode=instructor.Mode.MD_JSON)
    llm = llm_factory(
        llm_cfg.model,
        client=client,
        adapter="litellm",
        model_args={"temperature": temperature},
    )
    embeddings = LiteLLMEmbeddings(
        model=llm_cfg.embedding_model,
        api_key=llm_cfg.api_key,
        api_base=llm_cfg.api_base,
        encoding_format="float",
    )

    # Create scorers
    scorers = {
        "context_precision": ContextPrecision(llm=llm),
        "context_recall": ContextRecall(llm=llm),
        "faithfulness": Faithfulness(llm=llm),
        "answer_correctness": AnswerCorrectness(llm=llm, embeddings=embeddings),
    }

    # Score all records
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [
        asyncio.create_task(_score_row_async(record, scorers, semaphore))
        for record in records
    ]
    
    scores = await asyncio.gather(*tasks)
    return scores


def _compute_statistics(values: List[float]) -> dict:
    """Compute statistics for a list of values."""
    return {
        "avg": statistics.mean(values),
        "min": min(values),
        "max": max(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _generate_readme(
    config: EvaluationConfig,
    stats: dict,
    num_questions: int,
    timestamp: str,
) -> str:
    """Generate README content documenting the evaluation run."""
    
    readme = f"""# Evaluation Results: {config.output_name}

Generated on: {timestamp}

## Input Data

- **Description**: {config.input_data_description}
- **Evaluation Dataset**: `{config.eval_csv_path}`
- **Number of Questions**: {num_questions}

## Model Configuration

| Parameter | Value |
|-----------|-------|
| LLM Model | `{config.llm}` |
| Embedding Model | `{config.embedding_model}` |
| Temperature | {config.temperature} |
| Seed | {config.seed} |

## Preprocessing & Retrieval

| Parameter | Value |
|-----------|-------|
| Chunk Size | {config.chunk_size} characters |
| Chunk Overlap | {config.chunk_overlap} characters |
| Top-K Retrieval | {config.top_k} |

## RAGAS Evaluation Metrics

| Metric | Average | Min | Max | Std Dev |
|--------|---------|-----|-----|---------|
| Context Precision | {stats['context_precision']['avg']*100:.1f}% | {stats['context_precision']['min']*100:.1f}% | {stats['context_precision']['max']*100:.1f}% | {stats['context_precision']['std']*100:.1f}% |
| Context Recall | {stats['context_recall']['avg']*100:.1f}% | {stats['context_recall']['min']*100:.1f}% | {stats['context_recall']['max']*100:.1f}% | {stats['context_recall']['std']*100:.1f}% |
| Faithfulness | {stats['faithfulness']['avg']*100:.1f}% | {stats['faithfulness']['min']*100:.1f}% | {stats['faithfulness']['max']*100:.1f}% | {stats['faithfulness']['std']*100:.1f}% |
| Answer Correctness | {stats['answer_correctness']['avg']*100:.1f}% | {stats['answer_correctness']['min']*100:.1f}% | {stats['answer_correctness']['max']*100:.1f}% | {stats['answer_correctness']['std']*100:.1f}% |

## Metrics Interpretation

- **Context Precision**: How much of the retrieved context is actually relevant (higher = less noise)
- **Context Recall**: How much of the relevant information is captured in the context (higher = better retrieval)
- **Faithfulness**: How well the answer is grounded in the provided context (higher = less hallucination)
- **Answer Correctness**: Semantic similarity between generated and ground truth answers (higher = more accurate)

### Rule of Thumb Analysis

"""
    
    # Add rule of thumb hints
    precision_avg = stats['context_precision']['avg']
    recall_avg = stats['context_recall']['avg']
    faithfulness_avg = stats['faithfulness']['avg']
    
    HIGH = 0.75
    LOW = 0.5
    
    hints = []
    if recall_avg >= HIGH and precision_avg <= LOW:
        hints.append("- ⚠️ High Recall + Low Precision: Too much context or poor answer formulation")
    if precision_avg >= HIGH and faithfulness_avg <= LOW:
        hints.append("- ⚠️ High Precision + Low Faithfulness: Answer doesn't properly use the context")
    if precision_avg <= LOW and recall_avg >= HIGH:
        hints.append("- ⚠️ Low Precision + High Recall: Retrieval returns a lot but imprecisely")
    
    if not hints:
        hints.append("- ✅ No concerning patterns detected in the metrics")
    
    readme += "\n".join(hints)
    
    readme += """

## Output Files

- Full CSV with retrieved contexts: `{output_name}.csv`
- Compact CSV without retrieved contexts: `{output_name}_compact.csv`
- This README: `{output_name}.md`
""".format(output_name=config.output_name)
    
    return readme


def generate_evaluation_results(
    llm: str,
    embedding_model: str,
    input_data_description: str,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    output_name: str,
    temperature: float,
    seed: int = 42,
    eval_csv_path: str = "data/data_evaluation/GSKI_Fragen-Antworten-Fundstellen.csv",
) -> Path:
    """
    Generate evaluation CSV and README files.
    
    This function runs the complete RAG evaluation pipeline:
    1. Loads questions from the evaluation CSV
    2. Retrieves context chunks for each question
    3. Generates answers using the specified LLM
    4. Evaluates answers using RAGAS metrics
    5. Saves results to CSV and generates documentation README
    
    Args:
        llm: LLM model identifier (e.g., "openai/gpt-oss-120b")
        embedding_model: Embedding model identifier (e.g., "openai/octen-embedding-8b")
        input_data_description: Free-text description of input data and preprocessing
        chunk_size: Character size of chunks in vector database
        chunk_overlap: Character overlap between chunks
        top_k: Number of chunks to retrieve per question
        output_name: Base name for output files (without extension)
        temperature: LLM temperature setting
        seed: Random seed for reproducibility (default: 42)
        eval_csv_path: Path to evaluation CSV file (default: standard location)
    
    Returns:
        Path to the generated CSV file
    
    Raises:
        RuntimeError: If RAGAS evaluation fails for any question
        FileNotFoundError: If evaluation CSV does not exist
        ValueError: If required environment variables are not set
    """
    
    # Create configuration
    config = EvaluationConfig(
        llm=llm,
        embedding_model=embedding_model,
        input_data_description=input_data_description,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        output_name=output_name,
        temperature=temperature,
        seed=seed,
        eval_csv_path=eval_csv_path,
    )
    
    # Set seed for reproducibility
    _set_seed(config.seed)
    
    print(f"Starting evaluation: {config.output_name}")
    print(f"  LLM: {config.llm}")
    print(f"  Embedding: {config.embedding_model}")
    print(f"  Temperature: {config.temperature}, Seed: {config.seed}")
    print()
    
    # Resolve paths
    if not os.path.isabs(config.eval_csv_path):
        eval_csv_full = PROJECT_ROOT / config.eval_csv_path
    else:
        eval_csv_full = Path(config.eval_csv_path)
    
    if not eval_csv_full.exists():
        raise FileNotFoundError(f"Evaluation CSV not found: {eval_csv_full}")
    
    # Load evaluation data
    print(f"Loading evaluation data from: {eval_csv_full}")
    df = pd.read_csv(eval_csv_full, sep=";", encoding="utf-8-sig")
    print(f"  Loaded {len(df)} questions")
    print()
    
    # Load LLM and VectorDB config from environment
    llm_cfg = LLMConfig(
        api_base=os.getenv("LITELLM_API_BASE"),
        api_key=os.getenv("LITELLM_API_KEY"),
        model=config.llm,
        embedding_model=config.embedding_model,
    )
    vec_cfg = load_vectordb_config()
    qdrant_client = get_qdrant_client(vec_cfg)
    collection_name = vec_cfg.collection or "grundschutz_xml"
    
    # Prepare few-shot example from first row
    first_row = df.iloc[0]
    fewshot = [{"question": first_row["Frage"], "answer": first_row["Antwort"]}]
    
    # Build records for all questions (skip first row used for few-shot)
    print("Retrieving contexts and generating answers...")
    records = []
    
    for idx, row in df.iloc[1:].iterrows():
        question = row["Frage"]
        ground_truth_answer = row["Antwort"]
        ground_truth_context = row["Fundstellen im IT-Grundschutz-Kompendium 2023"]
        
        # Retrieve contexts
        contexts = _retrieve_contexts(
            question, config.top_k, qdrant_client, collection_name, llm_cfg
        )
        
        # Build messages and generate answer
        messages = _build_messages(question, contexts, fewshot)
        answer = _generate_answer(messages, llm_cfg, config.temperature, config.seed)
        
        records.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth_answer": ground_truth_answer,
            "ground_truth_context": ground_truth_context,
        })
        
        if (idx) % 10 == 0:
            print(f"  Processed {idx}/{len(df)-1} questions...")
    
    print(f"  Generated {len(records)} answers")
    print()
    
    # Save intermediate answers file (in case RAGAS fails)
    output_dir = PROJECT_ROOT / "data" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    intermediate_path = output_dir / f"{config.output_name}_answers.csv"
    intermediate_records = []
    for record in records:
        intermediate_records.append({
            "Frage": record["question"],
            "Antwort": record["ground_truth_answer"],
            "Fundstellen": record["ground_truth_context"],
            "Generierte Antwort": record["answer"],
            "Ermittelte Fundstellen": "\n".join(record["contexts"]),
        })
    intermediate_df = pd.DataFrame(intermediate_records)
    intermediate_df.to_csv(intermediate_path, sep=";", index=False, encoding="utf-8-sig")
    print(f"  Saved intermediate answers: {intermediate_path}")
    print()
    
    # Run RAGAS evaluation
    print("Running RAGAS evaluation...")
    
    # Handle async execution (works in both Jupyter and script contexts)
    try:
        loop = asyncio.get_running_loop()
        # We're in an async context (e.g., Jupyter)
        import nest_asyncio
        nest_asyncio.apply()
        scores = asyncio.get_event_loop().run_until_complete(
            _run_ragas_evaluation(records, llm_cfg, config.temperature)
        )
    except RuntimeError:
        # No running event loop, create a new one
        scores = asyncio.run(
            _run_ragas_evaluation(records, llm_cfg, config.temperature)
        )
    
    print("  RAGAS evaluation complete")
    print()
    
    # Compute statistics
    stats = {
        metric: _compute_statistics([s[metric] for s in scores])
        for metric in ["context_precision", "context_recall", "faithfulness", "answer_correctness"]
    }
    
    # Create output directory
    output_dir = PROJECT_ROOT / "data" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare CSV data
    # Use newline as delimiter for retrieved contexts (same as Fundstellen)
    csv_records = []
    for record, score in zip(records, scores):
        csv_records.append({
            "Frage": record["question"],
            "Antwort": record["ground_truth_answer"],
            "Fundstellen": record["ground_truth_context"],
            "Generierte Antwort": record["answer"],
            "Ermittelte Fundstellen": "\n".join(record["contexts"]),
            "context_precision": score["context_precision"],
            "context_recall": score["context_recall"],
            "faithfulness": score["faithfulness"],
            "answer_correctness": score["answer_correctness"],
        })
    
    # Write full CSV (with retrieved contexts)
    csv_path = output_dir / f"{config.output_name}.csv"
    csv_df = pd.DataFrame(csv_records)
    csv_df.to_csv(csv_path, sep=";", index=False, encoding="utf-8-sig")
    print(f"Saved full CSV: {csv_path}")
    
    # Write compact CSV (without retrieved contexts)
    csv_compact_path = output_dir / f"{config.output_name}_compact.csv"
    csv_compact_df = csv_df.drop(columns=["Ermittelte Fundstellen"])
    csv_compact_df.to_csv(csv_compact_path, sep=";", index=False, encoding="utf-8-sig")
    print(f"Saved compact CSV: {csv_compact_path}")
    
    # Generate and write README
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    readme_content = _generate_readme(config, stats, len(records), timestamp)
    readme_path = output_dir / f"{config.output_name}.md"
    readme_path.write_text(readme_content, encoding="utf-8")
    print(f"Saved README: {readme_path}")
    
    # Print summary
    print()
    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Context Precision:  {stats['context_precision']['avg']*100:.1f}% (±{stats['context_precision']['std']*100:.1f}%)")
    print(f"Context Recall:     {stats['context_recall']['avg']*100:.1f}% (±{stats['context_recall']['std']*100:.1f}%)")
    print(f"Faithfulness:       {stats['faithfulness']['avg']*100:.1f}% (±{stats['faithfulness']['std']*100:.1f}%)")
    print(f"Answer Correctness: {stats['answer_correctness']['avg']*100:.1f}% (±{stats['answer_correctness']['std']*100:.1f}%)")
    print("=" * 60)
    
    return csv_path


if __name__ == "__main__":
    # Example usage when run directly
    print("This script is intended to be imported and called from a Jupyter notebook.")
    print("See notebooks/02_run_evaluation_example.ipynb for usage examples.")
