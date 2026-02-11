"""
Generate missing gpt-oss-120b answer files and run RAGAS evaluation.

This script:
1. Checks which answer files are missing
2. Generates answers for missing configurations
3. Runs RAGAS evaluation on all answer files using gpt-oss-120b
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from qdrant_client import QdrantClient

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
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from litellm_client import LLMConfig, get_embeddings, chat_completion


# Configuration
LLM_MODEL = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "openai/octen-embedding-8b"
TEMPERATURE = 0.2
SEED = 42
TOP_K = 5

# All expected jobs
ALL_JOBS = [
    # gpt-oss-120b
    {"llm": "gpt-oss-120b", "collection": "gski_json_pdfs", "questions": "123_Einfach", "name": "gpt-oss-120b_json-pdfs_123-einfach"},
    {"llm": "gpt-oss-120b", "collection": "gski_json_pdfs", "questions": "43_Komplex", "name": "gpt-oss-120b_json-pdfs_43-komplex"},
    {"llm": "gpt-oss-120b", "collection": "gski_xml_pdfs", "questions": "123_Einfach", "name": "gpt-oss-120b_xml-pdfs_123-einfach"},
    {"llm": "gpt-oss-120b", "collection": "gski_xml_pdfs", "questions": "43_Komplex", "name": "gpt-oss-120b_xml-pdfs_43-komplex"},
    {"llm": "gpt-oss-120b", "collection": "gski_baseline", "questions": "123_Einfach", "name": "gpt-oss-120b_baseline_123-einfach"},
    {"llm": "gpt-oss-120b", "collection": "gski_baseline", "questions": "43_Komplex", "name": "gpt-oss-120b_baseline_43-komplex"},
    # granite-4-h-tiny
    {"llm": "granite-4-h-tiny", "collection": "gski_json_pdfs", "questions": "123_Einfach", "name": "granite-4-h-tiny_json-pdfs_123-einfach"},
    {"llm": "granite-4-h-tiny", "collection": "gski_json_pdfs", "questions": "43_Komplex", "name": "granite-4-h-tiny_json-pdfs_43-komplex"},
    {"llm": "granite-4-h-tiny", "collection": "gski_xml_pdfs", "questions": "123_Einfach", "name": "granite-4-h-tiny_xml-pdfs_123-einfach"},
    {"llm": "granite-4-h-tiny", "collection": "gski_xml_pdfs", "questions": "43_Komplex", "name": "granite-4-h-tiny_xml-pdfs_43-komplex"},
    {"llm": "granite-4-h-tiny", "collection": "gski_baseline", "questions": "123_Einfach", "name": "granite-4-h-tiny_baseline_123-einfach"},
    {"llm": "granite-4-h-tiny", "collection": "gski_baseline", "questions": "43_Komplex", "name": "granite-4-h-tiny_baseline_43-komplex"},
]

SYSTEM_PROMPT = """Du bist ein hilfreicher Assistent für IT-Sicherheit und BSI IT-Grundschutz.
Beantworte die Frage basierend auf dem gegebenen Kontext.
Antworte präzise und vollständig auf Deutsch.
Wenn der Kontext die Frage nicht beantwortet, sage das ehrlich."""


def get_llm_config(llm_model: str = LLM_MODEL) -> LLMConfig:
    """Get LLM configuration."""
    return LLMConfig(
        api_base=os.getenv("LITELLM_API_BASE"),
        api_key=os.getenv("LITELLM_API_KEY"),
        model=f"openai/{llm_model}",
        embedding_model=EMBEDDING_MODEL,
    )


def retrieve_contexts(question: str, collection: str, llm_cfg: LLMConfig, top_k: int = TOP_K) -> list[str]:
    """Retrieve relevant contexts from Qdrant."""
    client = QdrantClient(host="localhost", port=6333)
    
    # Embed the question
    question_embedding = get_embeddings([question], llm_cfg)[0]
    
    # Search using query_points (qdrant-client v2 API)
    response = client.query_points(
        collection_name=collection,
        query=question_embedding,
        limit=top_k,
    )
    
    # Extract text from results
    contexts = []
    for result in response.points:
        text = result.payload.get("text", "") or result.payload.get("content", "")
        if text:
            contexts.append(text)
    
    return contexts


def generate_answer(question: str, contexts: list[str], llm_cfg: LLMConfig) -> str:
    """Generate an answer using the LLM."""
    context_text = "\n\n".join(contexts)
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"""Kontext:
{context_text}

Frage: {question}

Antwort:"""}
    ]

    response = chat_completion(
        messages=messages,
        config=llm_cfg,
        temperature=TEMPERATURE,
        seed=SEED,
    )
    
    return response.choices[0].message.content


def generate_answers_for_job(job: dict) -> Path:
    """Generate answers for a single job configuration."""
    results_dir = PROJECT_ROOT / "data" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    output_name = job["name"]
    answers_file = results_dir / f"{output_name}_answers.csv"
    
    print(f"\n{'='*60}")
    print(f"Generating: {output_name}")
    print(f"{'='*60}")
    print(f"  LLM: openai/{job['llm']}")
    print(f"  Collection: {job['collection']}")
    print(f"  Questions: {job['questions']}")
    
    # Load questions
    if job["questions"] == "123_Einfach":
        csv_path = PROJECT_ROOT / "data" / "data_evaluation" / "GSKI_Fragen-Antworten-Fundstellen_123_Einfach.csv"
    else:
        csv_path = PROJECT_ROOT / "data" / "data_evaluation" / "GSKI_Fragen-Antworten-Fundstellen_43_Komplex.csv"
    
    df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig")
    print(f"  Loaded {len(df)} questions")
    
    # Get LLM config
    llm_cfg = get_llm_config(job["llm"])
    
    # Generate answers
    generated_answers = []
    retrieved_contexts = []
    
    for idx, row in df.iterrows():
        question = row["Frage"]
        
        # Retrieve contexts
        contexts = retrieve_contexts(question, job["collection"], llm_cfg)
        
        # Generate answer
        answer = generate_answer(question, contexts, llm_cfg)
        
        generated_answers.append(answer)
        retrieved_contexts.append("\n".join(contexts))
        
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(df)} questions...")
    
    print(f"  Generated {len(generated_answers)} answers")
    
    # Save results
    df["Generierte Antwort"] = generated_answers
    df["Ermittelte Fundstellen"] = retrieved_contexts
    
    df.to_csv(answers_file, sep=";", index=False, encoding="utf-8-sig")
    print(f"  Saved: {answers_file.name}")
    
    return answers_file


def main():
    """Main function to generate missing answers and run RAGAS."""
    results_dir = PROJECT_ROOT / "data" / "results"
    
    print("=" * 70)
    print("HUMAN EVALUATION - COMPLETE PIPELINE")
    print("=" * 70)
    
    # Check which files are missing
    missing_jobs = []
    existing_jobs = []
    
    for job in ALL_JOBS:
        answers_file = results_dir / f"{job['name']}_answers.csv"
        if answers_file.exists():
            existing_jobs.append(job)
        else:
            missing_jobs.append(job)
    
    print(f"\nExisting answer files: {len(existing_jobs)}")
    for job in existing_jobs:
        print(f"  ✓ {job['name']}")
    
    print(f"\nMissing answer files: {len(missing_jobs)}")
    for job in missing_jobs:
        print(f"  ✗ {job['name']}")
    
    # Generate missing answers
    if missing_jobs:
        print("\n" + "=" * 70)
        print("GENERATING MISSING ANSWERS")
        print("=" * 70)
        
        for job in missing_jobs:
            try:
                generate_answers_for_job(job)
            except Exception as e:
                print(f"  ✗ Failed: {e}")
    
    # Run RAGAS evaluation on all answer files
    print("\n" + "=" * 70)
    print("RUNNING RAGAS EVALUATION")
    print("=" * 70)
    
    from run_ragas_only import run_all_ragas_evaluations
    run_all_ragas_evaluations()


if __name__ == "__main__":
    main()
