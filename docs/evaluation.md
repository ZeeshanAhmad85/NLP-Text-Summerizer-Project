# Evaluation Methodology & Benchmark Analysis

This document describes the empirical evaluation framework, benchmark metrics, dataset characteristics, and baseline comparisons for the **NLP Text Summarization System**.

---

## 1. Evaluation Dataset: SAMSum

The system is evaluated on the **SAMSum Corpus** (Gliwa et al., 2019):

- **Domain**: Multi-turn dialogue and messenger conversations.
- **Language**: English (dialogic, informal, multi-speaker conversational turns).
- **Scale**: ~16,369 conversations annotated with third-person human reference summaries:
  - **Train**: 14,732 conversations
  - **Validation**: 818 conversations
  - **Test**: 819 conversations
- **License**: Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0).
- **Attribution**: *Gliwa, B., Mochol, M., Biesek, M., & Wróbel, P. (2019). SAMSum Corpus: A Human-annotated Dialogue Dataset for Abstractive Summarization. In Proceedings of the 2nd Workshop on New Frontiers in Summarization (EMNLP-IJCNLP 2019).*

---

## 2. Evaluation Metrics (ROUGE)

We evaluate abstractive summary fidelity using standard **ROUGE** (*Recall-Oriented Understudy for Gisting Evaluation*) metrics (Lin, 2004):

### ROUGE-1 (Unigram Overlap)
Measures the unigram recall and precision between generated candidate summary ($C$) and human reference ($R$):
$$\text{ROUGE-1} = \frac{\sum_{w \in R} \text{Count}_{\text{match}}(w)}{\sum_{w \in R} \text{Count}(w)}$$

### ROUGE-2 (Bigram Overlap)
Measures informative coherence and local phrase fluency through consecutive word pairs:
$$\text{ROUGE-2} = \frac{\sum_{\text{bigram} \in R} \text{Count}_{\text{match}}(\text{bigram})}{\sum_{\text{bigram} \in R} \text{Count}(\text{bigram})}$$

### ROUGE-L (Longest Common Subsequence)
Measures sentence-level structure by identifying the longest common sequence of words without requiring consecutive adjacency:
$$R_{\text{LCS}} = \frac{\text{LCS}(C, R)}{m}, \quad P_{\text{LCS}} = \frac{\text{LCS}(C, R)}{n}, \quad F_{\text{LCS}} = \frac{(1 + \beta^2) R_{\text{LCS}} P_{\text{LCS}}}{R_{\text{LCS}} + \beta^2 P_{\text{LCS}}}$$

### ROUGE-Lsum
Computes the Longest Common Subsequence across summary line breaks, reflecting multi-sentence coherence.

---

## 3. Baseline Comparison: Lead-3 Extractive Baseline

To provide an empirical anchor, the system benchmarks against the canonical **Lead-3 baseline** (Nallapati et al., 2016):
- Selects the first three sentences of the input document/dialogue.
- In news summarization (e.g., CNN/DailyMail), Lead-3 is an exceptionally strong baseline due to the "inverted pyramid" journalistic style.
- In **dialogue summarization**, however, conversational turn greetings ("Hi", "How are you?") dominate initial lines. Abstractive sequence-to-sequence models significantly outperform Lead-3 by synthesizing dialogue turns across multiple speakers into coherent third-person narratives.

---

## 4. Reproducing Empirical Evaluation

The evaluation framework is automated and self-contained:

```bash
# Run standalone benchmark suite with Lead-3 comparison
python evaluate_model.py --samples 5 --output-dir artifacts/model_evaluation

# Run full evaluation stage from the pipeline orchestrator
python main.py --stage evaluation --eval-samples 10
```

Results are deterministically exported to:
- `artifacts/model_evaluation/evaluation_report.json`
- `artifacts/model_evaluation/metrics_summary.csv`
