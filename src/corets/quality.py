"""Validate question packages and evidence-grounded trajectory reviews."""
from __future__ import annotations

from .llm import Client, parse_object

DIMENSIONS = ('correctness', 'completeness', 'relevance', 'clarity', 'reasoning', 'grounding')


def question_package(value):
    required = ('question', 'reference_answer', 'positive_criteria', 'negative_criteria', 'source_paths')
    if not isinstance(value, dict) or any(k not in value for k in required):
        raise ValueError('Incomplete question package')
    if any(not isinstance(value[k], str) or not value[k].strip() for k in required[:2]):
        raise ValueError('Question and reference answer must be nonempty strings')
    for key in required[2:]:
        if not isinstance(value[key], list) or not value[key] or any(not isinstance(x, str) or not x.strip() for x in value[key]):
            raise ValueError(f'{key} must be a nonempty list of strings')
    return value


def review(profile, package, answer, sources, thresholds):
    import json
    prompt = ('Assess the answer against the question, reference, criteria, and source excerpts. '
              'Return only JSON with scores for correctness, completeness, relevance, clarity, reasoning, '
              'and grounding (integers 1 to 10), supported (boolean), source_conflict (boolean), and '
              'evidence (a nonempty list of exact quotes from the candidate answer). '
              'Reject unsupported assertions or contradictions with source code.\n' +
              json.dumps({'package': package, 'candidate_answer': answer, 'sources': sources}, ensure_ascii=False))
    result = parse_object(Client(profile).request([{'role': 'user', 'content': prompt}]).content,
                          {'scores', 'supported', 'source_conflict', 'evidence'})
    scores = result['scores']
    if not isinstance(scores, dict) or set(scores) != set(DIMENSIONS):
        raise ValueError('Invalid review dimensions')
    if any(type(scores[d]) is not int or not 1 <= scores[d] <= 10 for d in DIMENSIONS):
        raise ValueError('Invalid review score')
    if type(result['supported']) is not bool or type(result['source_conflict']) is not bool:
        raise ValueError('Invalid review decision')
    quotes = result['evidence']
    grounded = isinstance(quotes, list) and bool(quotes) and all(isinstance(q, str) and len(q.strip()) >= 8 and q in answer for q in quotes)
    result['accepted'] = (grounded and result['supported'] and not result['source_conflict']
                          and all(scores[d] >= thresholds[d] for d in DIMENSIONS))
    return result
