import unittest

from classifier import NewsClassifier
from cli import build_parser


class FakeTokenizer:
    model_max_length = 512

    def __call__(self, text, hypothesis, **kwargs):
        return {"input_ids": [1, 2, 3]}


class FakePipeline:
    tokenizer = FakeTokenizer()

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []

    def __call__(self, text, candidate_labels, hypothesis_template, multi_label):
        self.calls.append((text, candidate_labels, multi_label))
        return {"labels": list(reversed(candidate_labels)),
                "scores": [0.8, 0.2]}


class ClassifierTests(unittest.TestCase):
    def setUp(self):
        self.pipeline_instances = []

        def factory(**kwargs):
            instance = FakePipeline(**kwargs)
            self.pipeline_instances.append(instance)
            return instance

        self.classifier = NewsClassifier(pipeline_factory=factory)

    def test_blank_passage_rejected(self):
        with self.assertRaises(ValueError):
            self.classifier.classify("   ", ["sports", "politics"])

    def test_requires_two_labels(self):
        with self.assertRaises(ValueError):
            self.classifier.classify("A passage", ["sports"])

    def test_blank_label_rejected(self):
        with self.assertRaises(ValueError):
            self.classifier.classify("A passage", ["sports", " "])

    def test_duplicate_labels_case_insensitive(self):
        with self.assertRaises(ValueError):
            self.classifier.classify("A passage", ["Sports", "sports"])

    def test_scores_stay_aligned_with_sorted_labels(self):
        result = self.classifier.classify("A passage", ["sports", "politics"])
        self.assertEqual(result["labels"], ["politics", "sports"])
        self.assertEqual(result["scores"], [0.8, 0.2])

    def test_pipeline_is_loaded_once_and_reused(self):
        self.classifier.classify("First passage", ["sports", "politics"])
        self.classifier.classify("Second passage", ["sports", "politics"])
        self.assertEqual(len(self.pipeline_instances), 1)

    def test_cli_parser_accepts_multilabel_and_output(self):
        args = build_parser().parse_args([
            "Example passage", "--labels", "sports", "politics",
            "--multi-label", "--output", "result.json"
        ])
        self.assertTrue(args.multi_label)
        self.assertEqual(args.labels, ["sports", "politics"])
        self.assertEqual(args.output, "result.json")


if __name__ == "__main__":
    unittest.main()
