#!/usr/bin/env python3
"""
Human Evaluation Runner

Runs evaluations for:
- 2 LLMs: gpt-oss-120b, granite-4-h-tiny
- 2 Databases: gski_json_pdfs, gski_xml_pdfs
- 2 Question Sets: 123_Einfach, 43_Komplex

Total: 2 × 2 × 2 = 8 evaluation result files

Usage:
    python scripts/run_human_evaluation.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Setup paths
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS_DIR))

# Load environment
try:
    from dotenv import load_dotenv
    env_path = NOTEBOOKS_DIR / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

from run_evaluation import generate_evaluation_results


@dataclass
class EvaluationJob:
    """Configuration for a single evaluation job."""
    llm: str
    collection: str
    csv_path: str
    output_name: str
    description: str


# Configuration
MODELS = [
    ("openai/gpt-oss-120b", "gpt-oss-120b"),
    ("openai/granite-4-h-tiny", "granite-4-h-tiny"),
]

DATABASES = [
    ("gski_json_pdfs", "json-pdfs"),
    ("gski_xml_pdfs", "xml-pdfs"),
    ("gski_baseline", "baseline"),
]

QUESTION_SETS = [
    ("data/data_evaluation/GSKI_Fragen-Antworten-Fundstellen_123_Einfach.csv", "123-einfach"),
    ("data/data_evaluation/GSKI_Fragen-Antworten-Fundstellen_43_Komplex.csv", "43-komplex"),
]

EMBEDDING_MODEL = "openai/octen-embedding-8b"
CHUNK_SIZE = 4000
CHUNK_OVERLAP = 200
TOP_K = 5
TEMPERATURE = 0.2


def generate_jobs() -> list[EvaluationJob]:
    """Generate all evaluation job configurations."""
    jobs = []
    
    for llm_full, llm_short in MODELS:
        for collection, db_short in DATABASES:
            for csv_path, qs_short in QUESTION_SETS:
                output_name = f"{llm_short}_{db_short}_{qs_short}"
                description = f"{llm_short} on {db_short} with {qs_short}"
                
                jobs.append(EvaluationJob(
                    llm=llm_full,
                    collection=collection,
                    csv_path=csv_path,
                    output_name=output_name,
                    description=description,
                ))
    
    return jobs


def run_evaluation_job(job: EvaluationJob) -> Optional[Path]:
    """Run a single evaluation job."""
    print("\n" + "=" * 70)
    print(f"EVALUATION: {job.description}")
    print("=" * 70)
    print(f"  LLM: {job.llm}")
    print(f"  Collection: {job.collection}")
    print(f"  Questions: {job.csv_path}")
    print(f"  Output: {job.output_name}")
    print()
    
    # Set the collection in environment for this run
    os.environ["QDRANT_COLLECTION"] = job.collection
    os.environ["VECTORDB_COLLECTION"] = job.collection
    
    try:
        result_path = generate_evaluation_results(
            llm=job.llm,
            embedding_model=EMBEDDING_MODEL,
            input_data_description=f"Collection: {job.collection}",
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            top_k=TOP_K,
            output_name=job.output_name,
            temperature=TEMPERATURE,
            seed=42,
            eval_csv_path=job.csv_path,
        )
        print(f"\n✓ Completed: {result_path}")
        return result_path
    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        return None


def main():
    jobs = generate_jobs()
    
    print("=" * 70)
    print("HUMAN EVALUATION - 8 Evaluation Runs")
    print("=" * 70)
    print(f"\nModels: {', '.join(m[1] for m in MODELS)}")
    print(f"Databases: {', '.join(d[0] for d in DATABASES)}")
    print(f"Question Sets: {', '.join(q[1] for q in QUESTION_SETS)}")
    print(f"\nTotal jobs: {len(jobs)}")
    print()
    
    # List all jobs
    for i, job in enumerate(jobs, 1):
        print(f"  {i}. {job.output_name}")
    
    print("\n" + "-" * 70)
    input("Press Enter to start evaluations (Ctrl+C to cancel)...")
    
    # Run all jobs
    results = []
    for i, job in enumerate(jobs, 1):
        print(f"\n[{i}/{len(jobs)}] Starting: {job.output_name}")
        result = run_evaluation_job(job)
        results.append((job, result))
    
    # Summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    
    successful = [(j, r) for j, r in results if r is not None]
    failed = [(j, r) for j, r in results if r is None]
    
    print(f"\nSuccessful: {len(successful)}/{len(results)}")
    for job, path in successful:
        print(f"  ✓ {job.output_name}: {path}")
    
    if failed:
        print(f"\nFailed: {len(failed)}")
        for job, _ in failed:
            print(f"  ✗ {job.output_name}")
    
    print("\n" + "=" * 70)
    
    return 0 if not failed else 1


if __name__ == "__main__":
    exit(main())
