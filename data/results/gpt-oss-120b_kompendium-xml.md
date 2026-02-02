# Evaluation Results: gpt-oss-120b_kompendium-xml

Generated on: 2026-02-02 12:57:03

## Input Data

- **Description**: XML Kompendium 2023 (data/grundschutz.xml), character-based chunking with 4000 char chunks and 200 char overlap
- **Evaluation Dataset**: `data/data_evaluation/GSKI_Fragen-Antworten-Fundstellen.csv`
- **Number of Questions**: 41

## Model Configuration

| Parameter | Value |
|-----------|-------|
| LLM Model | `openai/gpt-oss-120b` |
| Embedding Model | `openai/octen-embedding-8b` |
| Temperature | 0.2 |
| Seed | 42 |

## Preprocessing & Retrieval

| Parameter | Value |
|-----------|-------|
| Chunk Size | 4000 characters |
| Chunk Overlap | 200 characters |
| Top-K Retrieval | 5 |

## RAGAS Evaluation Metrics

| Metric | Average | Min | Max | Std Dev |
|--------|---------|-----|-----|---------|
| Context Precision | 87.5% | 45.0% | 100.0% | 17.1% |
| Context Recall | 89.1% | 0.0% | 100.0% | 25.1% |
| Faithfulness | 79.9% | 0.0% | 100.0% | 28.6% |
| Answer Correctness | 64.6% | 15.8% | 98.1% | 22.8% |

## Metrics Interpretation

- **Context Precision**: How much of the retrieved context is actually relevant (higher = less noise)
- **Context Recall**: How much of the relevant information is captured in the context (higher = better retrieval)
- **Faithfulness**: How well the answer is grounded in the provided context (higher = less hallucination)
- **Answer Correctness**: Semantic similarity between generated and ground truth answers (higher = more accurate)

### Rule of Thumb Analysis

- ✅ No concerning patterns detected in the metrics

## Output Files

- CSV with detailed results: `gpt-oss-120b_kompendium-xml.csv`
- This README: `gpt-oss-120b_kompendium-xml.md`
