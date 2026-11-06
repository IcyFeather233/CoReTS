"""Deterministic bounded Python evidence packs from immutable source trees."""
from __future__ import annotations
import ast
import hashlib
import random
from pathlib import Path

EXCLUDE = {'.git', '.venv', 'venv', 'node_modules', '__pycache__', 'build', 'dist', 'site-packages'}


def safe_file(root: Path, relative: str) -> Path:
    root = root.resolve()
    name = relative.removeprefix('/repo/').removeprefix('/app/')
    if Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError('Source paths must stay within the repository')
    target = root / name
    if not target.is_file() or not target.resolve().is_relative_to(root):
        raise ValueError('Missing or escaping source path: ' + relative)
    if target.stat().st_size > 1_000_000:
        raise ValueError('Source file exceeds the evidence size limit')
    return target


def excerpt(root: Path, relative: str, start: int = 1, end: int | None = None) -> dict:
    target = safe_file(root, relative)
    lines = target.read_text(encoding='utf-8', errors='replace').splitlines()
    first = max(1, start); last = min(len(lines), end if end is not None else first + 179)
    return {'path': target.relative_to(root).as_posix(), 'start_line': first, 'end_line': last,
            'text': '\n'.join(f'{i+1}: {lines[i]}' for i in range(first-1, last))}


def build_packs(root: Path, maximum: int = 32, character_budget: int = 24000, seed: int = 42) -> list[dict]:
    root = root.resolve(); candidates = []
    for path in sorted(root.rglob('*.py')):
        relative = path.relative_to(root)
        if any(part in EXCLUDE for part in relative.parts) or path.is_symlink() or path.stat().st_size > 500_000:
            continue
        text = path.read_text(encoding='utf-8', errors='replace')
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError, RecursionError):
            continue
        symbols = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        for symbol in symbols:
            start = max(1, symbol.lineno-8); end = min(getattr(symbol, 'end_lineno', symbol.lineno+60)+8, start+179)
            ex = excerpt(root, relative.as_posix(), start, end)
            imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
            context = '\n'.join(ast.get_source_segment(text, n) or '' for n in imports)
            if len(ex['text'])+len(context) > character_budget:
                continue
            identifier = hashlib.sha256(f'{relative}:{symbol.lineno}:{symbol.name}'.encode()).hexdigest()[:16]
            candidates.append({'id': identifier, 'path': relative.as_posix(), 'symbol': symbol.name,
                               'imports': context[:4000], 'source': ex})
    random.Random(seed).shuffle(candidates)
    return candidates[:maximum]


def gather_evidence(root: Path, paths: list[str], budget: int = 36000) -> list[dict]:
    output = []; seen = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        try:
            ex = excerpt(root, path)
        except ValueError:
            continue
        if len(ex['text']) > budget:
            lines = ex['text'].splitlines(); kept=[]; size=0
            for line in lines:
                if size+len(line)+1 > budget:
                    break
                kept.append(line);size+=len(line)+1
            ex['text']='\n'.join(kept);ex['end_line']=ex['start_line']+len(kept)-1
        if ex['text']:
            output.append(ex);budget-=len(ex['text'])
        if budget < 200:
            break
    if not output:
        raise ValueError('No verifiable source excerpt for the review')
    return output
