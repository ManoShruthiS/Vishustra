"""End-to-end smoke test: the Kernel must synthesize, execute, and explain."""


def test_kernel_review_pipeline(kernel, sample_review):
    report = kernel.process(
        "clean this up, hide personal info, make it polite",
        sample_review,
    )
    assert report["mode"] in ("rules", "llm")
    assert report["pipeline"]
    assert len(report["reasons"]) == len(report["pipeline"])
    assert report["result"]["trace"]


def test_kernel_translate_and_tone(kernel):
    report = kernel.process(
        "translate to hindi and make it formal",
        "Please tell the manager about the delay in delivery.",
    )
    assert report["pipeline"][0] == "language_translator"
    assert report["params"]["target_language"] == "hi"
    assert report["params"]["target_tone"] == "formal"


def test_report_is_json_serializable(kernel):
    import json

    report = kernel.process("summarize this", "a b c d e f g h i j k l m n o p q r s t u v")
    json.dumps(report, default=str)