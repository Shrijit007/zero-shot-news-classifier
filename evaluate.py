import json
from pathlib import Path

from classifier import DEFAULT_LABELS, NewsClassifier

ROOT = Path(__file__).resolve().parent


def main():
    cases = json.loads((ROOT / "examples" / "demo_passages.json").read_text(encoding="utf-8"))
    classifier = NewsClassifier(device=-1)
    records = []
    correct = 0

    for index, case in enumerate(cases, start=1):
        result = classifier.classify(case["text"], DEFAULT_LABELS, multi_label=False)
        predicted = result["labels"][0]
        is_correct = predicted == case["label"]
        correct += int(is_correct)
        records.append({
            "example": index,
            "expected": case["label"],
            "predicted": predicted,
            "correct": is_correct,
            "result": result,
        })
        print(f'{index:02d}. expected={case["label"]:<12} predicted={predicted:<12} '
              f'{"OK" if is_correct else "CHECK"}')

    accuracy = correct / len(cases) if cases else 0.0
    output = {
        "model": classifier.model_id,
        "examples": len(cases),
        "correct_first_ranked": correct,
        "accuracy": accuracy,
        "records": records,
        "note": "Small hand-written demonstration only; not an independent benchmark.",
    }
    output_path = ROOT / "examples" / "sample_results.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nDemonstration accuracy: {correct}/{len(cases)} = {accuracy:.1%}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
