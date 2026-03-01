from rank_bm25 import BM25Okapi

from experiments.baselines.base import BaselinePrediction
from experiments.baselines.reader_utils import extract_answer, pick_best_sentence, tokenize
from experiments.data.hotpot_loader import HotpotSample


class SupportingFactNaiveBaseline:
    name = "baseline_3_supporting_fact_naive"

    def __init__(self, top_k_sentences: int = 5) -> None:
        self._top_k_sentences = top_k_sentences

    def predict(self, sample: HotpotSample) -> BaselinePrediction:
        if not sample.context_sentences:
            return BaselinePrediction(answer="", retrieved_titles=[], retrieved_facts=[])

        sentence_texts = [sentence for _, _, sentence in sample.context_sentences]
        tokenized_sentences = [tokenize(sentence) for sentence in sentence_texts]
        query_tokens = tokenize(sample.question)

        bm25 = BM25Okapi(tokenized_sentences)
        scores = bm25.get_scores(query_tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
        selected_indices = ranked_indices[: max(1, min(self._top_k_sentences, len(ranked_indices)))]

        selected_records = [sample.context_sentences[index] for index in selected_indices]
        selected_sentences = [sentence for _, _, sentence in selected_records]

        best_sentence = pick_best_sentence(sample.question, selected_sentences)
        answer = extract_answer(sample.question, best_sentence)

        selected_titles = list(dict.fromkeys(title for title, _, _ in selected_records))
        selected_facts = [(title, sentence_id) for title, sentence_id, _ in selected_records]

        return BaselinePrediction(
            answer=answer,
            retrieved_titles=selected_titles,
            retrieved_facts=selected_facts,
        )
