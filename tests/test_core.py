import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from corets.evidence import build_packs, gather_evidence
from corets.llm import parse_object
from corets.quality import question_package
from corets.sources import validate_repository


class CoreTests(unittest.TestCase):
    def test_evidence_is_bounded_and_paths_cannot_escape(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'example.py').write_text('def answer():\n    return 42\n')
            packs = build_packs(root)
            self.assertEqual(packs[0]['symbol'], 'answer')
            self.assertIn('return 42', gather_evidence(root, ['example.py'])[0]['text'])
            with self.assertRaises(ValueError):
                gather_evidence(root, ['../outside'])

    def test_json_and_question_contract(self):
        self.assertEqual(parse_object('```json\n{"x": 1}\n```', {'x'}), {'x': 1})
        with self.assertRaises(ValueError):
            question_package({'question': 'Incomplete'})

    def test_repository_requires_pinned_commit(self):
        self.assertEqual(validate_repository({'repository': 'owner/repo', 'commit': 'a'*40, 'source_url': 'file:///tmp/repository'})['commit'], 'a'*40)
        with self.assertRaises(ValueError):
            validate_repository({'repository': 'owner/repo', 'commit': 'main'})
        with self.assertRaises(ValueError):
            validate_repository({'repository': 'owner/repo', 'commit': 'a'*40})


if __name__ == '__main__': unittest.main()
