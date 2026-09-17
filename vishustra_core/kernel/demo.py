"""VISHUSTRA KERNEL demo — the "wow" moment.

    python -m vishustra_core.kernel.demo --intent "clean this, make it polite and short" --text "..."
    python -m vishustra_core.kernel.demo --scenario review   # sample scenarios, no text needed
"""

import argparse
import logging
import sys
import json

from vishustra_core.kernel.executor import VishustraKernel, VishustraKernelError

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

SCENARIOS = {
    "review": {
        "intent": "clean this review, remove private info, tell me sentiment and the important keywords",
        "text": "I love this app but it keeps crashing on my phone 9840012345. support@fix.com. Absolutely amazing features though!!",
    },
    "polite": {
        "intent": "make this message polite and friendly",
        "text": "your product is total crap, give my money back NOW",
    },
    "hindi": {
        "intent": "translate to hindi and make it formal",
        "text": "Please tell the manager about the delay in delivery of order #9942.",
    },
    "facts": {
        "intent": "check if this claim is true and summarize it",
        "text": "The capital of France is Paris. It is a very famous and long text worth summarizing because the original is a bit too lengthy to read quickly.",
    },
    "urls": {
        "intent": "extract all links and summarize",
        "text": "See https://example.com and www.news.org for details. The story is quite long and deserves a short summary for quick reading.",
    },
}


def pretty_print(result: Dict) -> None:
    plan = result
    print("\n" + "=" * 62)
    print("VISHUSTRA KERNEL  |  synthesized intent -> pipeline")
    print("=" * 62)
    print(f"INTENT : {plan['intent']}")
    print(f"MODE   : {plan['mode']}")
    print(f"PIPELINE: {' -> '.join(plan['pipeline'])}")
    print("-" * 62)
    print("WHY:")
    for r in plan["reasons"]:
        print(f"  - {r}")
    print("-" * 62)
    print("PARAMS:", json.dumps(plan["params"], indent=2) if plan["params"] else "(none)")
    print("-" * 62)
    print("EXECUTION TRACE:")
    outputs = plan["result"]["outputs"]
    for t in plan["result"]["trace"]:
        print(f"  [{t['step']}] in={t['input_type']} out={t['output_type']} status={t['status']}")
    print("-" * 62)
    print("FINAL RESULT:")
    final = plan["result"]["final"]
    if isinstance(final, dict):
        print(json.dumps(final, indent=2, ensure_ascii=False))
    else:
        print(final)
    print("=" * 62)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="VISHUSTRA KERNEL demo")
    parser.add_argument("--intent", help="what you want done to the text, in plain language")
    parser.add_argument("--text", help="the text to process")
    parser.add_argument("--scenario", choices=list(SCENARIOS.keys()), help="run a built-in demo scenario")
    parser.add_argument("--raw", action="store_true", help="print raw JSON instead of the pretty report")
    args = parser.parse_args(argv)

    if args.scenario:
        demo = SCENARIOS[args.scenario]
        intent, text = demo["intent"], demo["text"]
    else:
        intent = args.intent
        text = args.text
        if not intent:
            parser.print_help()
            print("\nTry: python -m vishustra_core.kernel.demo --scenario review")
            return 2

    text = text if text is not None else "Sample input text for the pipeline."
    kernel = VishustraKernel()
    try:
        result = kernel.process(intent, text)
    except VishustraKernelError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    if args.raw:
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    else:
        pretty_print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())