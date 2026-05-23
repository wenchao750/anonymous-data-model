# Runway Retrieval Dataset

This directory contains a domain-specific text retrieval dataset for runway maintenance and airfield operations. It can be used for semantic retrieval, ranking, and query-text matching experiments.

Files:

- `runway-queries.json`: query set organized by anonymized IDs
- `runway-document.json`: text/document set organized by anonymized IDs
- `train.csv`, `dev.csv`, `test.csv`: triples of queries, related texts, and relevance labels for training, validation, and testing
- `runway_retrieval.py`: example script for retrieval training and evaluation

Dataset scale:

- 766 queries
- 1352 text/document entries
- 80844 training triples
- 10105 validation triples
- 10105 test triples

Notes:

- The data comes from a specialized operational domain, so this README only provides a high-level description.
- Relevance labels are graded annotations intended for retrieval evaluation and model training.
- To protect domain privacy in a public repository, the detailed data source, label policy, and business context are intentionally omitted.
