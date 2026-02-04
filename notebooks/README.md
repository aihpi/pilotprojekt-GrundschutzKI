# Notebooks

Dieses Verzeichnis enthaelt kurze Jupyter-Notebooks zum Testen von RAG, GraphRAG, Docling und LangGraph.

## Struktur
- 00_upper_bound.ipynb (Upper-Bound Evaluation mit Real-Context)
- 01_rag_baseline.ipynb (RAG Baseline: XML, Qdrant, RAGAS)
- 02_dspy_rag.ipynb (DSPy RAG Experimente)
- 05_json_preprocessing.ipynb (JSON-Preprocessing + Ingestion)

## Setup (Beispiel)
- Python-Umgebung aktivieren
- Abhaengigkeiten installieren (siehe pyproject.toml)
- .env basierend auf notebooks/.env.example anlegen (LiteLLM)

## Konventionen
- Jeder Notebook laeuft end-to-end und enthaelt eine kurze Auswertung.
- Datenpfade und Modellparameter sind oben dokumentiert.
- Gemeinsame LLM-Helper liegen in notebooks/litellm_client.py.
