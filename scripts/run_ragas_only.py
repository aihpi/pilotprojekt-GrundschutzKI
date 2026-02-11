"""
Run RAGAS evaluation on existing answer files using gpt-oss-120b.

This script evaluates pre-generated answers using RAGAS metrics,
using gpt-oss-120b as the evaluation model regardless of which model
generated the answers.
"""
from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List

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

from litellm_client import LLMConfig

# RAGAS evaluation model - always use gpt-oss-120b for reliable JSON output
RAGAS_MODEL = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "openai/octen-embedding-8b"


async def _score_row_async(
    row: dict,
    scorers: dict,
    semaphore: asyncio.Semaphore,
    question_idx: int,
    total_questions: int,
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
            
            if (question_idx + 1) % 10 == 0:
                print(f"    RAGAS: {question_idx + 1}/{total_questions} questions evaluated...")
            
            # Extract numeric .value from MetricResult objects
            return {
                "context_precision": context_precision.value if hasattr(context_precision, 'value') else float(context_precision),
                "context_recall": context_recall.value if hasattr(context_recall, 'value') else float(context_recall),
                "faithfulness": faithfulness.value if hasattr(faithfulness, 'value') else float(faithfulness),
                "answer_correctness": answer_correctness.value if hasattr(answer_correctness, 'value') else float(answer_correctness),
            }
        except Exception as e:
            print(f"    ⚠ RAGAS failed for question {question_idx + 1}: {str(e)[:200]}")
            # Return NaN values instead of failing completely
            return {
                "context_precision": float('nan'),
                "context_recall": float('nan'),
                "faithfulness": float('nan'),
                "answer_correctness": float('nan'),
            }


async def _run_ragas_evaluation(
    records: List[dict],
    llm_cfg: LLMConfig,
    temperature: float = 0.0,
    concurrency: int = 2,
) -> List[dict]:
    """Run RAGAS evaluation on all records using gpt-oss-120b."""
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

    # Create RAGAS LLM and embeddings - always use gpt-oss-120b
    client = instructor.from_litellm(litellm.acompletion, mode=instructor.Mode.MD_JSON)
    llm = llm_factory(
        RAGAS_MODEL,  # Always use gpt-oss-120b for RAGAS
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
    total = len(records)
    tasks = [
        asyncio.create_task(_score_row_async(record, scorers, semaphore, idx, total))
        for idx, record in enumerate(records)
    ]
    
    scores = await asyncio.gather(*tasks)
    return scores


def run_ragas_on_answers_file(answers_csv_path: str) -> Path:
    """
    Run RAGAS evaluation on an existing answers CSV file.
    
    Args:
        answers_csv_path: Path to the *_answers.csv file
        
    Returns:
        Path to the evaluated CSV file
    """
    answers_path = Path(answers_csv_path)
    if not answers_path.exists():
        raise FileNotFoundError(f"Answers file not found: {answers_path}")
    
    # Derive output name from input file
    output_name = answers_path.stem.replace("_answers", "")
    
    print(f"\n{'='*60}")
    print(f"RAGAS Evaluation: {output_name}")
    print(f"{'='*60}")
    print(f"  Answers file: {answers_path.name}")
    print(f"  RAGAS model: {RAGAS_MODEL}")
    print()
    
    # Load answers
    df = pd.read_csv(answers_path, sep=";", encoding="utf-8-sig")
    print(f"  Loaded {len(df)} answers")
    
    # Prepare records for RAGAS
    records = []
    for _, row in df.iterrows():
        contexts = row["Ermittelte Fundstellen"].split("\n") if pd.notna(row["Ermittelte Fundstellen"]) else []
        records.append({
            "question": row["Frage"],
            "answer": row["Generierte Antwort"],
            "contexts": contexts,
            "ground_truth_answer": row["Antwort"],
            "ground_truth_context": row["Fundstellen"] if pd.notna(row["Fundstellen"]) else "",
        })
    
    # Configure LLM for RAGAS
    llm_cfg = LLMConfig(
        api_base=os.getenv("LITELLM_API_BASE"),
        api_key=os.getenv("LITELLM_API_KEY"),
        model=RAGAS_MODEL,
        embedding_model=EMBEDDING_MODEL,
    )
    
    # Run RAGAS evaluation
    print("  Running RAGAS evaluation...")
    try:
        loop = asyncio.get_running_loop()
        import nest_asyncio
        nest_asyncio.apply()
        scores = asyncio.get_event_loop().run_until_complete(
            _run_ragas_evaluation(records, llm_cfg)
        )
    except RuntimeError:
        scores = asyncio.run(_run_ragas_evaluation(records, llm_cfg))
    
    # Count successful evaluations
    successful = sum(1 for s in scores if not any(pd.isna(v) for v in s.values()))
    print(f"  RAGAS evaluation complete: {successful}/{len(scores)} successful")
    
    # Add scores to dataframe
    df["context_precision"] = [s["context_precision"] for s in scores]
    df["context_recall"] = [s["context_recall"] for s in scores]
    df["faithfulness"] = [s["faithfulness"] for s in scores]
    df["answer_correctness"] = [s["answer_correctness"] for s in scores]
    
    # Save evaluated file
    output_path = answers_path.parent / f"{output_name}_evaluated.csv"
    df.to_csv(output_path, sep=";", index=False, encoding="utf-8-sig")
    print(f"  Saved: {output_path.name}")
    
    # Print summary statistics
    valid_scores = [s for s in scores if not any(pd.isna(v) for v in s.values())]
    if valid_scores:
        import statistics
        print()
        print("  RAGAS Metrics Summary:")
        for metric in ["context_precision", "context_recall", "faithfulness", "answer_correctness"]:
            values = [s[metric] for s in valid_scores]
            avg = statistics.mean(values)
            print(f"    {metric}: {avg*100:.1f}%")
    
    return output_path


def run_all_ragas_evaluations():
    """Run RAGAS evaluation on all existing answer files."""
    results_dir = PROJECT_ROOT / "data" / "results"
    
    # Find all answer files
    answer_files = sorted(results_dir.glob("*_answers.csv"))
    
    print("=" * 70)
    print("RAGAS EVALUATION ON ALL ANSWER FILES")
    print(f"Using evaluation model: {RAGAS_MODEL}")
    print("=" * 70)
    print(f"\nFound {len(answer_files)} answer files to evaluate:\n")
    
    for f in answer_files:
        print(f"  - {f.name}")
    
    print()
    
    successful = []
    failed = []
    
    for answers_file in answer_files:
        try:
            output_path = run_ragas_on_answers_file(str(answers_file))
            successful.append(answers_file.stem.replace("_answers", ""))
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed.append(answers_file.stem.replace("_answers", ""))
    
    # Print final summary
    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\nSuccessful: {len(successful)}/{len(answer_files)}")
    for name in successful:
        print(f"  ✓ {name}")
    
    if failed:
        print(f"\nFailed: {len(failed)}/{len(answer_files)}")
        for name in failed:
            print(f"  ✗ {name}")
    
    print("=" * 70)


if __name__ == "__main__":
    run_all_ragas_evaluations()
