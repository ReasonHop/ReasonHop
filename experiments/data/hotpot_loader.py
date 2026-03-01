from dataclasses import dataclass
from itertools import islice

from datasets import load_dataset


@dataclass(frozen=True)
class HotpotSample:
    sample_id: str
    question: str
    answer: str
    contexts: list[str]
    context_titles: list[str]
    supporting_titles: set[str]
    context_sentences: list[tuple[str, int, str]]
    supporting_facts: set[tuple[str, int]]


def _build_contexts(raw_item: dict) -> tuple[list[str], list[str], list[tuple[str, int, str]]]:
    context_block = raw_item["context"]
    titles: list[str] = context_block["title"]
    sentence_lists: list[list[str]] = context_block["sentences"]

    passages: list[str] = []
    sentence_records: list[tuple[str, int, str]] = []
    for title, sentences in zip(titles, sentence_lists):
        for sentence_index, sentence in enumerate(sentences):
            cleaned_sentence = sentence.strip()
            if cleaned_sentence:
                sentence_records.append((title, sentence_index, cleaned_sentence))
        merged = " ".join(sentence.strip() for sentence in sentences if sentence.strip())
        passages.append(f"{title}. {merged}".strip())

    return passages, titles, sentence_records


def _build_supporting_titles(raw_item: dict) -> set[str]:
    supporting_facts = raw_item["supporting_facts"]
    titles: list[str] = supporting_facts["title"]
    return set(titles)


def _build_supporting_facts(raw_item: dict) -> set[tuple[str, int]]:
    supporting_facts = raw_item["supporting_facts"]
    titles: list[str] = supporting_facts["title"]
    sentence_ids: list[int] = supporting_facts["sent_id"]
    return {(title, sentence_id) for title, sentence_id in zip(titles, sentence_ids)}


def load_hotpot_samples(split: str = "validation", limit: int = 100) -> list[HotpotSample]:
    dataset = load_dataset("hotpot_qa", "distractor", split=split, streaming=True)

    samples: list[HotpotSample] = []
    rows = dataset if limit <= 0 else islice(dataset, limit)
    for row in rows:
        contexts, context_titles, context_sentences = _build_contexts(row)
        samples.append(
            HotpotSample(
                sample_id=row["id"],
                question=row["question"],
                answer=row["answer"],
                contexts=contexts,
                context_titles=context_titles,
                supporting_titles=_build_supporting_titles(row),
                context_sentences=context_sentences,
                supporting_facts=_build_supporting_facts(row),
            )
        )

    return samples
