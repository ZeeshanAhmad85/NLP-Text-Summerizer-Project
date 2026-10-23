from src.TextSummarizer.evaluation.evaluator import Lead3Baseline, SummarizationEvaluator


def test_lead3_baseline_extraction():
    baseline = Lead3Baseline()
    text = "First sentence here. Second sentence follows. Third sentence appears. Fourth sentence should be ignored."
    extracted = baseline.summarize(text)
    assert "First sentence here." in extracted
    assert "Second sentence follows." in extracted
    assert "Third sentence appears." in extracted
    assert "Fourth sentence should be ignored." not in extracted


def test_lead3_short_text():
    baseline = Lead3Baseline()
    text = "Only one sentence here."
    extracted = baseline.summarize(text)
    assert extracted == text


def test_summarization_evaluator_rouge():
    evaluator = SummarizationEvaluator(seed=42)
    predictions = ["Amanda gave Betty's number to Hannah."]
    references = ["Amanda provides Betty's phone number to Hannah."]
    scores = evaluator.compute_rouge(predictions, references)

    assert "rouge1" in scores
    assert "rouge2" in scores
    assert "rougeL" in scores
    assert scores["rouge1"] > 0.0


def test_summarization_evaluator_report_structure():
    evaluator = SummarizationEvaluator(seed=42)
    dialogues = ["Hannah: Hey do you have Betty's number? Amanda: Yes, it is 555-1234."]
    references = ["Amanda gives Betty's phone number to Hannah."]
    predictions = ["Amanda gives Betty's phone number to Hannah."]

    report = evaluator.evaluate_model_vs_baseline(
        dialogues=dialogues,
        references=references,
        model_predictions=predictions,
        model_name="Test-Model",
        inference_time_sec=0.1
    )

    assert "evaluation_metadata" in report
    assert "metrics_comparison" in report
    assert "Test-Model" in report["metrics_comparison"]
    assert "Lead-3 (Baseline)" in report["metrics_comparison"]
    assert len(report["sample_qualitative_outputs"]) == 1
