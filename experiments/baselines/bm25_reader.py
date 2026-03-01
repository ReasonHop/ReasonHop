from rank_bm25 import BM25Okapi

from experiments.baselines.base import BaselinePrediction
from experiments.baselines.reader_utils import extract_answer, pick_best_sentence, tokenize
from experiments.data.hotpot_loader import HotpotSample


class BM25ReaderBaseline:
    name = "baseline_2_bm25_plus_reader"

    def __init__(self, top_k: int = 3) -> None:
        self._top_k = top_k

    def predict(self, sample: HotpotSample) -> BaselinePrediction:
        if not sample.contexts:
            return BaselinePrediction(answer="", retrieved_titles=[], retrieved_facts=[])

        tokenized_passages = [tokenize(passage) for passage in sample.contexts]
        query_tokens = tokenize(sample.question)

        bm25 = BM25Okapi(tokenized_passages)
        scores = bm25.get_scores(query_tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
        selected_indices = ranked_indices[: max(1, min(self._top_k, len(ranked_indices)))]

        selected_passages = [sample.contexts[index] for index in selected_indices]
        selected_titles = [sample.context_titles[index] for index in selected_indices]

        best_sentence = pick_best_sentence(sample.question, selected_passages)
        answer = extract_answer(sample.question, best_sentence)

        selected_title_set = set(selected_titles)
        selected_facts = [
            (title, sentence_id)
            for title, sentence_id, _ in sample.context_sentences
            if title in selected_title_set
        ]

        return BaselinePrediction(
            answer=answer,
            retrieved_titles=selected_titles,
            retrieved_facts=selected_facts,
        )
