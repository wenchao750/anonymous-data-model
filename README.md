diff --git a/d:\PycharmProjects\PythonProject\data\runway-final\README.md b/d:\PycharmProjects\PythonProject\data\runway-final\README.md
new file mode 100644
--- /dev/null
+++ b/d:\PycharmProjects\PythonProject\data\runway-final\README.md
@@ -0,0 +1,26 @@
+# Runway Retrieval Dataset
+
+This folder contains a domain-specific text retrieval dataset for runway maintenance and airfield operation support.
+
+The data is organized around two text collections and three split files:
+
+- `runway-queries.json`: query set with anonymized query IDs
+- `runway-document.json`: related text/document set with anonymized document IDs
+- `train.csv`, `dev.csv`, `test.csv`: query-document relevance triples for training, validation, and testing
+- `runway_retrieval.py`: an example script for retrieval training and evaluation
+
+At a high level, the dataset supports experiments in semantic retrieval, ranking, and domain adaptation for maintenance-oriented question-text matching.
+
+Current scale:
+
+- 766 queries
+- 1352 documents
+- 80844 training triples
+- 10105 validation triples
+- 10105 test triples
+
+Notes:
+
+- The query/document texts come from a specialized operational domain.
+- Relevance labels are provided as graded annotations for retrieval tasks.
+- To avoid exposing sensitive industry details, this README intentionally keeps the data description brief.
