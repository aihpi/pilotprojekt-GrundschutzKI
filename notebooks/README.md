# Notebooks

Dieses Verzeichnis enthaelt kurze Jupyter-Notebooks zum Testen von RAG, GraphRAG, Docling und LangGraph.

## Struktur
- 01_rag_baseline.ipynb
- 02_graphrag.ipynb
- 03_docling_ingestion.ipynb
- 04_langgraph_flow.ipynb

## Setup (Beispiel)
- Python-Umgebung aktivieren
- Abhaengigkeiten installieren (siehe pyproject.toml)
- .env basierend auf notebooks/.env.example anlegen (LiteLLM)

## Konventionen
- Jeder Notebook laeuft end-to-end und enthaelt eine kurze Auswertung.
- Datenpfade und Modellparameter sind oben dokumentiert.
- Gemeinsame LLM-Helper liegen in notebooks/litellm_client.py.
