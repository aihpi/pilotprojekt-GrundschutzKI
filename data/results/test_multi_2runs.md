# Multi-Run Evaluation Results: test_multi_2runs

Generated on: 2026-02-02 14:40:35

## Configuration

### Input Data

- **Description**: XML Kompendium 2023, character-based chunking
- **Evaluation Dataset**: `data/data_evaluation/GSKI_Fragen-Antworten-Fundstellen.csv`

### Model Configuration

| Parameter | Value |
|-----------|-------|
| LLM Model | `openai/gpt-oss-120b` |
| Embedding Model | `openai/octen-embedding-8b` |
| Temperature | 0.2 |

### Preprocessing & Retrieval

| Parameter | Value |
|-----------|-------|
| Chunk Size | 4000 characters |
| Chunk Overlap | 200 characters |
| Top-K Retrieval | 5 |

### Multi-Run Parameters

| Parameter | Value |
|-----------|-------|
| Number of Runs | 2 |
| Master Seed | 42 |

## Overall System Statistics (All Questions × All Runs)

These statistics represent the distribution across ALL question-answer pairs in ALL runs.

| Metric | Mean | Min | Max | Std Dev |
|--------|------|-----|-----|---------|
| Context Precision | 88.2% | 45.0% | 100.0% | 17.0% |
| Context Recall | 87.6% | 0.0% | 100.0% | 25.3% |
| Faithfulness | 75.8% | 11.1% | 100.0% | 27.4% |
| Answer Correctness | 62.9% | 12.5% | 98.8% | 23.6% |

## Per-Question Statistics (Distribution Across Runs)

See `test_multi_2runs_per_question.csv` for detailed per-question statistics showing how each question's metrics varied across the 2 runs.

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

- Per-question statistics: `test_multi_2runs_per_question.csv`
- Overall system statistics: `test_multi_2runs_overall.csv`
- This README: `test_multi_2runs.md`

## Run Seeds Used

The following seeds were generated from master seed 42:

```
478163327, 107420369
```
