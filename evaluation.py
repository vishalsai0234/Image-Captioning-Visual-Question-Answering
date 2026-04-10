import math
import re
import numpy as np
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import nltk

nltk.download("wordnet",  quiet=True)
nltk.download("punkt",    quiet=True)
nltk.download("punkt_tab",quiet=True)
nltk.download("omw-1.4",  quiet=True)


# ─── BLEU ─────────────────────────────────────────────────────────────────────

def compute_bleu_scores(hypothesis: str, references: list[str]) -> dict:
    smoothie   = SmoothingFunction().method4
    hyp_tokens = hypothesis.lower().split()
    ref_tokens = [r.lower().split() for r in references]
    scores     = {}
    for n in range(1, 5):
        weights      = tuple(1.0 / n if i < n else 0.0 for i in range(4))
        scores[f"BLEU-{n}"] = round(
            sentence_bleu(ref_tokens, hyp_tokens,
                          weights=weights, smoothing_function=smoothie), 4
        )
    return scores


# ─── METEOR ───────────────────────────────────────────────────────────────────

def compute_meteor(hypothesis: str, references: list[str]) -> float:
    hyp_tokens = hypothesis.lower().split()
    ref_tokens = [r.lower().split() for r in references]
    return round(
        max(meteor_score([ref], hyp_tokens) for ref in ref_tokens), 4
    )


# ─── CIDEr ────────────────────────────────────────────────────────────────────

def compute_cider(hypothesis: str, references: list[str], n: int = 4) -> float:
    def get_ngrams(tokens, max_n):
        ng = Counter()
        for k in range(1, max_n + 1):
            for i in range(len(tokens) - k + 1):
                ng[tuple(tokens[i : i + k])] += 1
        return ng

    def tfidf_weights(ng_list):
        doc_freq = Counter()
        for ng in ng_list:
            for key in set(ng.keys()):
                doc_freq[key] += 1
        nd = len(ng_list)
        weighted = []
        for ng in ng_list:
            w = {
                key: tf * math.log((nd + 1) / (doc_freq[key] + 1))
                for key, tf in ng.items()
            }
            weighted.append(w)
        return weighted

    hyp_tokens = hypothesis.lower().split()
    ref_tokens = [r.lower().split() for r in references]
    hyp_ng     = get_ngrams(hyp_tokens, n)
    ref_ngs    = [get_ngrams(r, n) for r in ref_tokens]
    weighted   = tfidf_weights(ref_ngs + [hyp_ng])
    w_refs, w_hyp = weighted[:-1], weighted[-1]

    def vec_norm(v):
        return math.sqrt(sum(x**2 for x in v.values())) + 1e-9

    score = sum(
        sum(w_hyp.get(k, 0) * w_ref.get(k, 0) for k in set(w_hyp) & set(w_ref))
        / (vec_norm(w_hyp) * vec_norm(w_ref))
        for w_ref in w_refs
    )
    return round(score / len(w_refs) * 10, 4)


# ─── VQA Accuracy ─────────────────────────────────────────────────────────────

def normalize_answer(ans: str) -> str:
    ans = ans.lower().strip()
    ans = re.sub(r"[^\w\s]", "", ans)
    ans = re.sub(r"\s+",     " ", ans)
    return ans


def compute_vqa_accuracy(predicted: str, ground_truth: str) -> float:
    return 1.0 if normalize_answer(predicted) == normalize_answer(ground_truth) else 0.0