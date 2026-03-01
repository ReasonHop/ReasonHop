import re


STOP_WORDS = {
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
    "did",
    "does",
    "do",
}


def tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-zA-Z0-9']+", text.lower()) if token not in STOP_WORDS]


def split_sentences(text: str) -> list[str]:
    pieces = re.split(r"(?<=[.!?])\s+", text.strip())
    return [piece.strip() for piece in pieces if piece.strip()]


def overlap_score(question: str, sentence: str) -> float:
    q_tokens = set(tokenize(question))
    s_tokens = set(tokenize(sentence))
    if not q_tokens:
        return 0.0
    overlap = len(q_tokens.intersection(s_tokens))
    return overlap / len(q_tokens)


def pick_best_sentence(question: str, passages: list[str]) -> str:
    best_sentence = ""
    best_score = -1.0

    for passage in passages:
        for sentence in split_sentences(passage):
            score = overlap_score(question, sentence)
            if score > best_score:
                best_score = score
                best_sentence = sentence

    return best_sentence if best_sentence else (passages[0] if passages else "")


def extract_candidate_entities(sentence: str) -> list[str]:
    patterns = [
        r"\b(?:[A-Z][\w'\-]+(?:\s+[A-Z][\w'\-]+){0,5})\b",
        r"\b\d{4}\b",
        r"\b\d+(?:,\d+)*(?:\.\d+)?\b",
        r'"([^"]+)"',
    ]
    candidates: list[str] = []

    for pattern in patterns:
        for match in re.finditer(pattern, sentence):
            value = match.group(1) if match.lastindex else match.group(0)
            clean = value.strip(" .,!?:;\"'")
            if clean and clean.lower() not in {"the", "a", "an"}:
                candidates.append(clean)

    unique_candidates: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = candidate.lower()
        if key not in seen:
            seen.add(key)
            unique_candidates.append(candidate)

    return unique_candidates


def extract_answer(question: str, sentence: str) -> str:
    if not sentence:
        return ""

    lower_question = question.lower()
    candidates = extract_candidate_entities(sentence)
    filtered = [item for item in candidates if item.lower() not in lower_question]
    if filtered:
        filtered.sort(key=lambda item: len(item.split()), reverse=True)
        return filtered[0]

    words = sentence.split()
    if len(words) <= 8:
        return sentence.strip(" .,!?:;")

    return " ".join(words[:8]).strip(" .,!?:;")
