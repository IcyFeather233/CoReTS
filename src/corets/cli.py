"""Portable orchestration for code-grounded task synthesis."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import shutil
from pathlib import Path

from .config import credentials, load_config
from .evidence import build_packs, gather_evidence
from .io import fingerprint, read_json, write_json
from .llm import Client, parse_object
from .quality import question_package, review
from .sources import load_manifest, prepare_snapshot, snapshot_key
from .upstream import build_tools_image, fetch_agent, verify_agent


def ask(profile, prompt, required):
    reply = Client(profile).request([{'role': 'user', 'content': prompt}])
    return parse_object(reply.content, set(required))


def generate_repository(row, config, work, upstream):
    from .agent import run_agent
    root = work / 'repositories' / snapshot_key(row)
    root.mkdir(parents=True, exist_ok=True)
    snapshot = prepare_snapshot(row, work)
    packs = build_packs(snapshot, config['max_evidence_packs'], config['evidence_characters'], config['seed'])
    if not packs:
        raise ValueError('No Python source evidence found')
    accepted = []
    for index, pack in enumerate(packs[:config['max_author_attempts_per_repository']]):
        if len(accepted) >= config['questions_per_repository']:
            break
        directory = root / f'attempt-{index + 1:03d}'
        directory.mkdir(exist_ok=True)
        try:
            scan = ask(config['models']['scanner'], 'Identify a repository comprehension opportunity grounded in this code. Return JSON with "opportunity" as a concise string.\n' + json.dumps(pack, ensure_ascii=False), ['opportunity'])
            author_input = {'opportunity': scan, 'evidence': pack, 'previous_questions': [x['question'] for x in accepted]}
            package = question_package(ask(config['models']['author'], 'Create a distinct, concise repository comprehension question with a source-grounded answer. Return JSON with question, reference_answer, positive_criteria (list), negative_criteria (list), and source_paths (list of repository-relative paths).\n' + json.dumps(author_input, ensure_ascii=False), ['question', 'reference_answer', 'positive_criteria', 'negative_criteria', 'source_paths']))
            if any(package['question'].casefold().strip() == x['question'].casefold().strip() for x in accepted):
                raise ValueError('Duplicate question')
            sources = [pack['source']] + gather_evidence(snapshot, package['source_paths'], config['source_excerpt_characters'])
            write_json(directory / 'package.json', package)
            gold = review(config['models']['reviewer'], package, package['reference_answer'], sources, config['quality_thresholds'])
            write_json(directory / 'gold_review.json', gold)
            if not gold['accepted']:
                write_json(directory / 'status.json', {'status': 'gold_rejected'})
                continue
            result, sft = run_agent(package['question'], snapshot, config['models']['teacher'], config, upstream, directory / 'trajectory')
            if result['status'] != 'success' or not sft:
                write_json(directory / 'status.json', {'status': 'rollout_rejected'})
                continue
            answer = sft['messages'][-1]['content']
            judged = review(config['models']['reviewer'], package, answer, sources, config['quality_thresholds'])
            write_json(directory / 'review.json', judged)
            write_json(directory / 'status.json', {'status': 'accepted' if judged['accepted'] else 'answer_rejected'})
            if judged['accepted']:
                accepted.append({'question': package['question'], 'sft': sft, 'task_id': fingerprint([row, package['question']])[:24]})
        except Exception as error:
            write_json(directory / 'status.json', {'status': 'error', 'error_type': type(error).__name__, 'error': str(error)})
    write_json(root / 'accepted.json', accepted)
    return {'repository': row['repository'], 'accepted': len(accepted), 'attempts': min(len(packs), config['max_author_attempts_per_repository'])}


def main():
    parser = argparse.ArgumentParser(prog='corets')
    sub = parser.add_subparsers(dest='command', required=True)
    setup = sub.add_parser('setup-agent'); setup.add_argument('--directory', type=Path, required=True); setup.add_argument('--context', type=Path, required=True); setup.add_argument('--image', default='corets-tools:93ac6a4')
    run = sub.add_parser('generate'); run.add_argument('--repositories', type=Path, required=True); run.add_argument('--config', type=Path, required=True); run.add_argument('--work', type=Path, required=True); run.add_argument('--agent', type=Path, required=True)
    export = sub.add_parser('export'); export.add_argument('--work', type=Path, required=True); export.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'setup-agent':
        build_tools_image(fetch_agent(args.directory), args.context, args.image)
    elif args.command == 'generate':
        config = load_config(args.config); rows = load_manifest(args.repositories); upstream = verify_agent(args.agent)
        for profile in config['models'].values(): credentials(profile)
        args.work.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(args.work).free < config['minimum_free_gib'] * 1024**3: raise RuntimeError('Insufficient disk space')
        with concurrent.futures.ThreadPoolExecutor(max_workers=config['concurrency']) as pool:
            futures = [pool.submit(generate_repository, row, config, args.work, upstream) for row in rows]
            for future in concurrent.futures.as_completed(futures):
                try: print(json.dumps(future.result()), flush=True)
                except Exception as error: print(json.dumps({'status': 'repository_error', 'error': str(error)}), flush=True)
    else:
        files = sorted((args.work / 'repositories').glob('*/accepted.json'))
        if not files: raise ValueError('No accepted repositories found')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('w', encoding='utf-8') as output:
            for file in files:
                for item in read_json(file): output.write(json.dumps(item['sft'], ensure_ascii=False) + '\n')


if __name__ == '__main__': main()
