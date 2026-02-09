# Results

## Summary Table (Avg)
| Approach | Model | context_precision | context_recall | faithfulness | answer_correctness |
|---|---|---:|---:|---:|---:|
| Original (contexts from previous pipeline) | gpt-oss-120b | 95.59% | 87.02% | 86.55% | 59.29% |
| JSON-based retrieval (grundschutz.json) — top_k=6 | gpt-oss-120b | 92.89% | 88.64% | 77.92% | 60.57% |
| JSON-based retrieval (grundschutz.json) — top_k=3 | gpt-oss-120b | 93.22% | 84.69% | 76.47% | 55.38% |
| JSON-based retrieval (grundschutz.json) — top_k=8 | gpt-oss-120b | 87.62% | 91.85% | 74.86% | 58.10% |

## Summary (What’s Better)
- Best answer correctness: JSON-based retrieval with top_k=6 (60.57%).
- Best context recall: JSON-based retrieval with top_k=8 (91.85%), but with lower faithfulness.
- Best faithfulness and precision: Original pipeline (86.55% faithfulness, 95.59% precision).
- Overall trade-off: top_k=6 is the best balance for correctness without a large drop in precision/recall.

## Details (Min/Max)
### Approach: Original (contexts from previous pipeline)
- Model: gpt-oss-120b (DSPy + LiteLLM proxy)
- Metrics:
  - context_precision avg: 0.9558823529 (min 0.3333333333, max 0.99999999997)
  - context_recall avg: 0.8702264239 (min 0.125, max 1.0)
  - faithfulness avg: 0.8655388471 (min 0.0, max 1.0)
  - answer_correctness avg: 0.5928906245 (min 0.05245033445, max 0.98908575215)

### Approach: JSON-based retrieval (grundschutz.json) — top_k=6
- Model: gpt-oss-120b (DSPy + LiteLLM proxy)
- Metrics:
  - context_precision avg: 0.9289460784 (min 0.67916666665, max 0.99999999998)
  - context_recall avg: 0.8864379085 (min 0.3333333333, max 1.0)
  - faithfulness avg: 0.7791520571 (min 0.0, max 1.0)
  - answer_correctness avg: 0.6056857923 (min 0.19844235099, max 0.98803911515)

### Approach: JSON-based retrieval (grundschutz.json) — top_k=3
- Model: gpt-oss-120b (DSPy + LiteLLM proxy)
- Metrics:
  - context_precision avg: 0.9321895424 (min 0.41666666665, max 0.99999999998)
  - context_recall avg: 0.8468707091 (min 0.2, max 1.0)
  - faithfulness avg: 0.7647026993 (min 0.0, max 1.0)
  - answer_correctness avg: 0.5538225867 (min 0.14387315124, max 0.98611004443)

### Approach: JSON-based retrieval (grundschutz.json) — top_k=8
- Model: gpt-oss-120b (DSPy + LiteLLM proxy)
- Metrics:
  - context_precision avg: 0.8762413298 (min 0.42063492062, max 0.99999999999)
  - context_recall avg: 0.9185022494 (min 0.2727272727, max 1.0)
  - faithfulness avg: 0.7485822184 (min 0.0, max 1.0)
  - answer_correctness avg: 0.5810184240 (min 0.13081568164, max 0.97831007942)


## Ground-Truth-Qualitaetsnotizen

- **Zusammenfassung**: Faithfulness liegt im Mittel bei ~0.83 und Answer Relevancy bei ~0.63. Das deutet darauf hin, dass viele Gold-Antworten nicht vollstaendig durch den gegebenen Kontext gedeckt sind und nur locker zur exakten Frage passen.
- **Beobachtete Probleme**:
  - Uebergeneraliserte oder unvollstaendige Zusammenfassungen (z. B. „KPIs, Register, Berichte“, ohne die konkrete Anforderung zu treffen).
  - Antworten paraphrasieren breitere Leitlinien, statt sich strikt am zitierten Kontext zu orientieren (z. B. Outsourcing- oder ISMS-Fragen, bei denen mehrere Abschnitte vermischt werden).
- **Auswirkung auf die Metriken**:
  - Niedrige Faithfulness, wenn Details fehlen oder nicht explizit im Kontext stehen.
  - Niedrige Relevancy, wenn die Antwort das Thema allgemein trifft, aber nicht die konkrete Frage.
- **Empfehlungen**:
  - Die Zuordnung Frage → Kontext → Gold-Antwort schaerfen.
  - Antworten extraktiver formulieren (z. B. Schluesselsaetze oder -klauseln uebernehmen).
  - Zusatzinformationen entfernen, die nicht im Kontext stehen.
  - Pro Frage nur den minimal notwendigen, praezisen Kontext bereitstellen.
