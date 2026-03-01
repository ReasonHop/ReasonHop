from experiments.baselines.base import BaselinePrediction
from experiments.baselines.reader_utils import extract_answer, pick_best_sentence
from experiments.data.hotpot_loader import HotpotSample


class ImplicitReaderBaseline:
    name = "baseline_1_implicit_reader"

    def predict(self, sample: HotpotSample) -> BaselinePrediction:
        best_sentence = pick_best_sentence(sample.question, sample.contexts)
        answer = extract_answer(sample.question, best_sentence)

        return BaselinePrediction(
            answer=answer,
            retrieved_titles=sample.context_titles,
            retrieved_facts=[(title, sentence_id) for title, sentence_id, _ in sample.context_sentences],
        )
