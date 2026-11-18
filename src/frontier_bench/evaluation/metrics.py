from __future__ import annotations
import re
from typing import Callable


def exact_match(reference: str, prediction: str) -> float:
    return float(reference.strip().casefold() == prediction.strip().casefold())


def normalized_exact_match(reference: str, prediction: str) -> float:
    normalize = lambda text: " ".join(re.sub(r"[^\w\s]", " ", text.casefold()).split())
    return float(normalize(reference) == normalize(prediction))


def word_error_rate(reference: str, prediction: str) -> float:
    ref, hyp = reference.split(), prediction.split()
    if not ref:
        return 0.0 if not hyp else 1.0
    row = list(range(len(hyp) + 1))
    for i, token in enumerate(ref, 1):
        prev, row = row, [i] + [0] * len(hyp)
        for j, word in enumerate(hyp, 1):
            row[j] = min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + (token != word))
    return row[-1] / len(ref)


METRICS: dict[str, Callable[[str, str], float]] = {
    "exact_match": exact_match, "normalized_exact_match": normalized_exact_match,
    "wer": word_error_rate,
}
