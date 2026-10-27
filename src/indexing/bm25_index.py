
import re
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """Normalize text into lowercase tokens with simple plural handling."""
    tokens = re.findall(r"\b\w+\b", text.lower())

    normalized = []

    for token in tokens:
        if len(token) > 4 and token.endswith("s"):
            token = token[:-1]

        normalized.append(token)

    return normalized
class BM25Index:
    def __init__(self, records: list[dict]):
        if not records:
            raise ValueError("Cannot build an index from empty records.")

        self.records = records

        self.tokenized_corpus = [
            tokenize(record.get("text", ""))
            for record in records
        ]

        self.index = BM25Okapi(self.tokenized_corpus)

    def search(
        self,
        query: str,
        top_k: int = 3,
        heading_boost: float = 4.0,
    ) -> list[dict]:

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        scores = self.index.get_scores(query_tokens)
        query_terms = set(query_tokens)

        ranked_results = []

        for i, record in enumerate(self.records):
            heading = record.get("heading_path", "")
            heading_terms = set(tokenize(heading))

            overlap = len(query_terms & heading_terms)

            adjusted_score = (
                float(scores[i])
                + heading_boost * overlap
            )

            result = record.copy()
            result["bm25_score"] = float(scores[i])
            result["heading_overlap"] = overlap
            result["retrieval_score"] = adjusted_score

            ranked_results.append(result)

        ranked_results.sort(
            key=lambda item: item["retrieval_score"],
            reverse=True,
        )

        return ranked_results[:top_k]