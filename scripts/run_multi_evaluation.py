"""
Multi-run evaluation script for measuring variance in RAGAS metrics.

This script runs the RAG evaluation multiple times with different seeds to measure
the statistical distribution of RAGAS metrics, providing insight into the
reproducibility and variance of the RAG system.

Usage from Jupyter notebook:
    from scripts.run_multi_evaluation import run_multi_evaluation
    
    run_multi_evaluation(
        llm="openai/gpt-oss-120b",
        embedding_model="openai/octen-embedding-8b",
        input_data_description="XML Kompendium 2023, character-based chunking",
        chunk_size=4000,
        chunk_overlap=200,
        top_k=5,
        output_name="gpt-oss-120b_kompendium-xml_multi",
        temperature=0.2,
        num_runs=100,
        master_seed=42,
    )
"""
from __future__ import annotations

import random
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

# Ensure scripts/ and notebooks/ are importable
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS_DIR))

# Import the single-run evaluation function
from run_evaluation import generate_evaluation_results


@dataclass
class MultiRunConfig:
    """Configuration for multi-run evaluation."""
    
    # Required parameters
    llm: str
    embedding_model: str
    input_data_description: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    output_name: str
    temperature: float
    
    # Multi-run specific parameters
    num_runs: int = 100
    master_seed: int = 42
    
    # Optional parameters with defaults
    eval_csv_path: str = "data/data_evaluation/GSKI_Fragen-Antworten-Fundstellen.csv"


def _generate_run_seeds(master_seed: int, num_runs: int) -> List[int]:
    """Generate reproducible seeds for each run based on master seed."""
    random.seed(master_seed)
    return [random.randint(0, 2**31 - 1) for _ in range(num_runs)]


def _compute_statistics(values: List[float]) -> Dict[str, float]:
    """Compute statistics for a list of values."""
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
    return {
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _generate_readme(
    config: MultiRunConfig,
    per_question_stats: pd.DataFrame,
    overall_stats: Dict[str, Dict[str, float]],
    run_seeds: List[int],
    timestamp: str,
) -> str:
    """Generate README content documenting the multi-run evaluation."""
    
    readme = f"""# Multi-Run Evaluation Results: {config.output_name}

Generated on: {timestamp}

## Configuration

### Input Data

- **Description**: {config.input_data_description}
- **Evaluation Dataset**: `{config.eval_csv_path}`

### Model Configuration

| Parameter | Value |
|-----------|-------|
| LLM Model | `{config.llm}` |
| Embedding Model | `{config.embedding_model}` |
| Temperature | {config.temperature} |

### Preprocessing & Retrieval

| Parameter | Value |
|-----------|-------|
| Chunk Size | {config.chunk_size} characters |
| Chunk Overlap | {config.chunk_overlap} characters |
| Top-K Retrieval | {config.top_k} |

### Multi-Run Parameters

| Parameter | Value |
|-----------|-------|
| Number of Runs | {config.num_runs} |
| Master Seed | {config.master_seed} |

## Overall System Statistics (All Questions × All Runs)

These statistics represent the distribution across ALL question-answer pairs in ALL runs.

| Metric | Mean | Min | Max | Std Dev |
|--------|------|-----|-----|---------|
| Context Precision | {overall_stats['context_precision']['mean']*100:.1f}% | {overall_stats['context_precision']['min']*100:.1f}% | {overall_stats['context_precision']['max']*100:.1f}% | {overall_stats['context_precision']['std']*100:.1f}% |
| Context Recall | {overall_stats['context_recall']['mean']*100:.1f}% | {overall_stats['context_recall']['min']*100:.1f}% | {overall_stats['context_recall']['max']*100:.1f}% | {overall_stats['context_recall']['std']*100:.1f}% |
| Faithfulness | {overall_stats['faithfulness']['mean']*100:.1f}% | {overall_stats['faithfulness']['min']*100:.1f}% | {overall_stats['faithfulness']['max']*100:.1f}% | {overall_stats['faithfulness']['std']*100:.1f}% |
| Answer Correctness | {overall_stats['answer_correctness']['mean']*100:.1f}% | {overall_stats['answer_correctness']['min']*100:.1f}% | {overall_stats['answer_correctness']['max']*100:.1f}% | {overall_stats['answer_correctness']['std']*100:.1f}% |

## Per-Question Statistics (Distribution Across Runs)

See `{config.output_name}_per_question.csv` for detailed per-question statistics showing how each question's metrics varied across the {config.num_runs} runs.

## Metrics Interpretation

- **Context Precision**: How much of the retrieved context is actually relevant (higher = less noise)
- **Context Recall**: How much of the relevant information is captured in the context (higher = better retrieval)
- **Faithfulness**: How well the answer is grounded in the provided context (higher = less hallucination)
- **Answer Correctness**: Semantic similarity between generated and ground truth answers (higher = more accurate)

## Variance Analysis

- **Low Std Dev** (< 5%): Highly consistent results across runs
- **Medium Std Dev** (5-15%): Moderate variance, some sensitivity to randomness
- **High Std Dev** (> 15%): High variance, results are sensitive to random seed

## Output Files

- Per-question statistics: `{config.output_name}_per_question.csv`
- Overall system statistics: `{config.output_name}_overall.csv`
- This README: `{config.output_name}.md`

## Run Seeds Used

The following seeds were generated from master seed {config.master_seed}:

```
{', '.join(str(s) for s in run_seeds[:10])}{'...' if len(run_seeds) > 10 else ''}
```
"""
    
    return readme


def run_multi_evaluation(
    llm: str,
    embedding_model: str,
    input_data_description: str,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    output_name: str,
    temperature: float,
    num_runs: int = 100,
    master_seed: int = 42,
    eval_csv_path: str = "data/data_evaluation/GSKI_Fragen-Antworten-Fundstellen.csv",
) -> Path:
    """
    Run evaluation multiple times and compute statistics on RAGAS metrics.
    
    This function runs the complete RAG evaluation pipeline multiple times
    with different seeds to measure variance in the results.
    
    Args:
        llm: LLM model identifier (e.g., "openai/gpt-oss-120b")
        embedding_model: Embedding model identifier (e.g., "openai/octen-embedding-8b")
        input_data_description: Free-text description of input data and preprocessing
        chunk_size: Character size of chunks in vector database
        chunk_overlap: Character overlap between chunks
        top_k: Number of chunks to retrieve per question
        output_name: Base name for output files (should include model info)
        temperature: LLM temperature setting
        num_runs: Number of evaluation runs (default: 100)
        master_seed: Master seed for generating run seeds (default: 42)
        eval_csv_path: Path to evaluation CSV file
    
    Returns:
        Path to the generated per-question CSV file
    
    Raises:
        RuntimeError: If evaluation fails for any run
    """
    
    # Create configuration
    config = MultiRunConfig(
        llm=llm,
        embedding_model=embedding_model,
        input_data_description=input_data_description,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        output_name=output_name,
        temperature=temperature,
        num_runs=num_runs,
        master_seed=master_seed,
        eval_csv_path=eval_csv_path,
    )
    
    print(f"Starting multi-run evaluation: {config.output_name}")
    print(f"  LLM: {config.llm}")
    print(f"  Embedding: {config.embedding_model}")
    print(f"  Temperature: {config.temperature}")
    print(f"  Number of runs: {config.num_runs}")
    print(f"  Master seed: {config.master_seed}")
    print()
    
    # Generate reproducible seeds for each run
    run_seeds = _generate_run_seeds(config.master_seed, config.num_runs)
    print(f"Generated {len(run_seeds)} run seeds from master seed {config.master_seed}")
    print()
    
    # Create output directory
    output_dir = PROJECT_ROOT / "data" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Storage for all run results
    all_run_results: List[pd.DataFrame] = []
    metrics = ["context_precision", "context_recall", "faithfulness", "answer_correctness"]
    
    # Run evaluation multiple times
    for run_idx, seed in enumerate(run_seeds):
        print(f"=" * 60)
        print(f"Run {run_idx + 1}/{config.num_runs} (seed: {seed})")
        print(f"=" * 60)
        
        # Generate a temporary output name for this run
        run_output_name = f"{config.output_name}_run_{run_idx + 1:03d}"
        
        try:
            # Run single evaluation
            csv_path = generate_evaluation_results(
                llm=config.llm,
                embedding_model=config.embedding_model,
                input_data_description=config.input_data_description,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                top_k=config.top_k,
                output_name=run_output_name,
                temperature=config.temperature,
                seed=seed,
                eval_csv_path=config.eval_csv_path,
            )
            
            # Read the results
            run_df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig")
            run_df["run"] = run_idx + 1
            run_df["seed"] = seed
            all_run_results.append(run_df)
            
            # Clean up individual run files (keep only aggregated results)
            csv_path.unlink(missing_ok=True)
            compact_path = csv_path.with_name(f"{run_output_name}_compact.csv")
            compact_path.unlink(missing_ok=True)
            readme_path = csv_path.with_name(f"{run_output_name}.md")
            readme_path.unlink(missing_ok=True)
            
        except Exception as e:
            print(f"ERROR in run {run_idx + 1}: {e}")
            raise RuntimeError(
                f"Multi-run evaluation failed at run {run_idx + 1} (seed: {seed})\n"
                f"Error: {type(e).__name__}: {e}"
            ) from e
        
        print()
    
    print(f"All {config.num_runs} runs completed successfully!")
    print()
    
    # Combine all results
    combined_df = pd.concat(all_run_results, ignore_index=True)
    
    # Compute per-question statistics (how each question varies across runs)
    print("Computing per-question statistics...")
    per_question_stats = []
    
    questions = combined_df["Frage"].unique()
    for question in questions:
        question_data = combined_df[combined_df["Frage"] == question]
        
        stats_row = {"Frage": question}
        for metric in metrics:
            values = question_data[metric].tolist()
            metric_stats = _compute_statistics(values)
            stats_row[f"{metric}_mean"] = metric_stats["mean"]
            stats_row[f"{metric}_min"] = metric_stats["min"]
            stats_row[f"{metric}_max"] = metric_stats["max"]
            stats_row[f"{metric}_std"] = metric_stats["std"]
        
        per_question_stats.append(stats_row)
    
    per_question_df = pd.DataFrame(per_question_stats)
    
    # Compute overall statistics (all questions × all runs)
    print("Computing overall system statistics...")
    overall_stats = {}
    for metric in metrics:
        all_values = combined_df[metric].tolist()
        overall_stats[metric] = _compute_statistics(all_values)
    
    # Create overall stats DataFrame
    overall_rows = []
    for metric in metrics:
        overall_rows.append({
            "metric": metric,
            "mean": overall_stats[metric]["mean"],
            "min": overall_stats[metric]["min"],
            "max": overall_stats[metric]["max"],
            "std": overall_stats[metric]["std"],
        })
    overall_df = pd.DataFrame(overall_rows)
    
    # Save per-question statistics
    per_question_path = output_dir / f"{config.output_name}_per_question.csv"
    per_question_df.to_csv(per_question_path, sep=";", index=False, encoding="utf-8-sig")
    print(f"Saved per-question statistics: {per_question_path}")
    
    # Save overall statistics
    overall_path = output_dir / f"{config.output_name}_overall.csv"
    overall_df.to_csv(overall_path, sep=";", index=False, encoding="utf-8-sig")
    print(f"Saved overall statistics: {overall_path}")
    
    # Generate and save README
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    readme_content = _generate_readme(config, per_question_df, overall_stats, run_seeds, timestamp)
    readme_path = output_dir / f"{config.output_name}.md"
    readme_path.write_text(readme_content, encoding="utf-8")
    print(f"Saved README: {readme_path}")
    
    # Print summary
    print()
    print("=" * 60)
    print("MULTI-RUN EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Number of runs: {config.num_runs}")
    print(f"Number of questions: {len(questions)}")
    print(f"Total evaluations: {len(combined_df)}")
    print()
    print("Overall System Statistics:")
    print(f"  Context Precision:  {overall_stats['context_precision']['mean']*100:.1f}% (±{overall_stats['context_precision']['std']*100:.1f}%)")
    print(f"  Context Recall:     {overall_stats['context_recall']['mean']*100:.1f}% (±{overall_stats['context_recall']['std']*100:.1f}%)")
    print(f"  Faithfulness:       {overall_stats['faithfulness']['mean']*100:.1f}% (±{overall_stats['faithfulness']['std']*100:.1f}%)")
    print(f"  Answer Correctness: {overall_stats['answer_correctness']['mean']*100:.1f}% (±{overall_stats['answer_correctness']['std']*100:.1f}%)")
    print("=" * 60)
    
    return per_question_path


if __name__ == "__main__":
    print("This script is intended to be imported and called from a Jupyter notebook.")
    print("See notebooks/03_multi_run_evaluation_example.ipynb for usage examples.")
