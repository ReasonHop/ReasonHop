import re
import string


def normalize_answer(value: str) -> str:
    value = value.lower()
    value = "".join(ch for ch in value if ch not in string.punctuation)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    value = " ".join(value.split())
    return value


def exact_match(prediction: str, gold: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0


def f1_score(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_freq: dict[str, int] = {}
    gold_freq: dict[str, int] = {}

    for token in pred_tokens:
        pred_freq[token] = pred_freq.get(token, 0) + 1
    for token in gold_tokens:
        gold_freq[token] = gold_freq.get(token, 0) + 1

    common = 0
    for token, count in pred_freq.items():
        common += min(count, gold_freq.get(token, 0))

    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def supporting_title_recall(retrieved_titles: list[str], gold_titles: set[str]) -> float:
    if not gold_titles:
        return 1.0
    if not retrieved_titles:
        return 0.0

    retrieved_set = set(retrieved_titles)
    matched = len(gold_titles.intersection(retrieved_set))
    return matched / len(gold_titles)


def supporting_fact_recall(
    retrieved_facts: list[tuple[str, int]],
    gold_facts: set[tuple[str, int]],
) -> float:
    if not gold_facts:
        return 1.0
    if not retrieved_facts:
        return 0.0

    retrieved_set = set(retrieved_facts)
    matched = len(gold_facts.intersection(retrieved_set))
    return matched / len(gold_facts)
