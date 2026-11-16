import argparse
import json
import sys
from pathlib import Path

from classifier import DEFAULT_LABELS, NewsClassifier


def build_parser():
    parser = argparse.ArgumentParser(description="Zero-shot classify an English passage.")
    parser.add_argument("text", help="Passage to classify.")
    parser.add_argument("--labels", nargs="+", default=DEFAULT_LABELS,
                        help="At least two candidate labels.")
    parser.add_argument("--multi-label", action="store_true",
                        help="Score each candidate independently.")
    parser.add_argument("--revision", default=None, help="Optional Hugging Face model revision.")
    parser.add_argument("--output", help="Optional path to write result JSON.")
    parser.add_argument("--device", type=int, default=-1,
                        help="Transformers device: -1 for CPU, 0 for first CUDA GPU.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        result = NewsClassifier(device=args.device, revision=args.revision).classify(
            args.text, args.labels, multi_label=args.multi_label
        )
        rendered = json.dumps(result, indent=2, ensure_ascii=False)
        print(rendered)
        if args.output:
            Path(args.output).write_text(rendered + "\n", encoding="utf-8")
        return 0
    except Exception as exc:
        print(f"Classification failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
