"""Commit-pinned source snapshots with no authoring or review artifacts."""
from __future__ import annotations
import fcntl
import json
import os
import re
import subprocess
import tarfile
import tempfile
from pathlib import Path
from urllib.parse import urlsplit
from .io import read_json, write_json

REPO = re.compile(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+')
COMMIT = re.compile(r'[0-9a-f]{40}')


def validate_repository(row: dict) -> dict:
    name = row.get('repository', '')
    commit = row.get('commit', row.get('base_commit', ''))
    if not REPO.fullmatch(name) or '..' in name or not COMMIT.fullmatch(commit):
        raise ValueError('Each repository needs owner/name and an exact 40-character commit')
    source = row.get('source_url', row.get('source', ''))
    parsed = urlsplit(source)
    if parsed.scheme not in ('https', 'http', 'file') or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('Source URL must not contain credentials or command options')
    return {'repository': name, 'commit': commit, 'source_url': source}


def load_manifest(path: Path) -> list[dict]:
    if Path(path).suffix == '.jsonl':
        raw = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    else:
        raw = read_json(path)
        if isinstance(raw, dict):
            raw = raw.get('repositories')
    if not isinstance(raw, list) or not raw:
        raise ValueError('Repository manifest must contain a nonempty list')
    rows = [validate_repository(row) for row in raw]
    ids = [(r['repository'].lower(), r['commit']) for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate repository revision in manifest')
    return rows


def snapshot_key(row: dict) -> str:
    return row['repository'].replace('/', '__') + '-' + row['commit']


def _git(*args, **kwargs):
    env = dict(os.environ, GIT_TERMINAL_PROMPT='0', GIT_LFS_SKIP_SMUDGE='1')
    return subprocess.check_output(['git', *map(str, args)], env=env, stderr=subprocess.PIPE, timeout=300, **kwargs)


def prepare_snapshot(row: dict, work: Path) -> Path:
    row = validate_repository(row)
    work = Path(work).resolve()
    key = snapshot_key(row)
    target = work / 'snapshots' / key
    marker = work / 'snapshot_metadata' / (key + '.json')
    lock_path = work / 'locks' / (row['repository'].replace('/', '__') + '.lock')
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if marker.exists() and target.is_dir():
            saved = read_json(marker)
            if saved['commit'] != row['commit'] or saved['repository'] != row['repository']:
                raise ValueError('Snapshot identity mismatch')
            return target
        if target.exists():
            raise ValueError(f'Unmarked snapshot exists; inspect it before retrying: {target}')
        cache = work / 'git_cache' / row['repository'].replace('/', '__')
        cache.parent.mkdir(parents=True, exist_ok=True)
        if not cache.exists():
            _git('init', '--bare', cache)
        exists = subprocess.run(['git', '-C', str(cache), 'cat-file', '-e', row['commit'] + '^{commit}'], capture_output=True)
        if exists.returncode:
            _git('-C', cache, 'fetch', '--depth', '1', row['source_url'], row['commit'])
        info = cache / 'info'; info.mkdir(exist_ok=True)
        (info / 'attributes').write_text('* -export-ignore -export-subst\n')
        temporary_root = work / 'tmp'; temporary_root.mkdir(parents=True, exist_ok=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root, prefix='snapshot-') as tmp:
            tmp = Path(tmp); archive = tmp / 'source.tar'; tree = tmp / 'tree'; tree.mkdir()
            with archive.open('wb') as stream:
                subprocess.run(['git', '-C', str(cache), 'archive', row['commit']], stdout=stream, stderr=subprocess.PIPE, check=True, timeout=180)
            if not hasattr(tarfile, 'data_filter'):
                raise RuntimeError('Use Python 3.12 or a Python 3.11 release with tarfile.data_filter')
            with tarfile.open(archive) as stream:
                stream.extractall(tree, filter='data')
            tree.rename(target)
        write_json(marker, {**row, 'tree': _git('-C', cache, 'rev-parse', row['commit'] + '^{tree}', text=True).strip(), 'source_only': True})
        return target
