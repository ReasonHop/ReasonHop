import re

from app.domain.models import EvidencePassage


class KeywordRetriever:
    _STOP_WORDS = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "to",
        "of",
        "in",
        "on",
        "for",
        "and",
        "or",
        "with",
        "as",
        "by",
        "what",
        "which",
        "who",
        "where",
        "when",
        "why",
        "how",
    }

    @classmethod
    def _tokenize(cls, text: str) -> set[str]:
        tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
        return {token for token in tokens if token not in cls._STOP_WORDS}

    def retrieve(self, question: str, contexts: list[str], top_k: int) -> list[EvidencePassage]:
        if not contexts:
            return []

        query_tokens = self._tokenize(question)
        scored: list[tuple[int, float]] = []

        for index, passage in enumerate(contexts):
            passage_tokens = self._tokenize(passage)
            overlap = query_tokens.intersection(passage_tokens)
            score = len(overlap) / max(1, len(query_tokens))
            if score > 0:
                scored.append((index, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        top_items = scored[: max(1, top_k)]

        return [
            EvidencePassage(
                id=f"p{index + 1}",
                text=contexts[index],
                score=round(score, 4),
            )
            for index, score in top_items
        ]
