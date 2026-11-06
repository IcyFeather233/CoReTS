"""OpenAI-compatible streaming transport with explicit model identity checks."""
from __future__ import annotations
from dataclasses import dataclass
import json
import time
from .config import credentials

class TransportError(RuntimeError): pass
class IdentityError(RuntimeError): pass
class ProtocolError(RuntimeError): pass

@dataclass
class Reply:
    content: str
    reasoning_content: str
    tool_calls: list[dict]
    usage: dict
    model: str
    finish_reason: str


def parse_object(text: str, required: set[str]) -> dict:
    def unique(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError('Duplicate JSON key')
            out[key] = value
        return out
    decoder = json.JSONDecoder(object_pairs_hook=unique)
    try:
        value = decoder.decode(text.strip())
        if isinstance(value, dict) and required <= set(value):
            return value
    except ValueError:
        pass
    candidates = []
    for i, char in enumerate(text):
        if char != '{':
            continue
        try:
            value, _ = decoder.raw_decode(text[i:])
        except ValueError:
            continue
        if isinstance(value, dict) and required <= set(value):
            candidates.append(value)
    if len(candidates) != 1:
        raise ProtocolError('Expected one unambiguous JSON object')
    return candidates[0]


class Client:
    def __init__(self, profile: dict, *, deadline: float | None = None, heartbeat=None):
        self.profile = profile
        self.deadline = deadline
        self.heartbeat = heartbeat or (lambda: None)

    def request(self, messages: list[dict], tools: list[dict] | None = None) -> Reply:
        import httpx
        base, key = credentials(self.profile)
        requested = self.profile['model']
        if requested.startswith('YOUR_'):
            raise ValueError('Replace the model name in the generation config')
        accepted = set(self.profile.get('accepted_model_ids') or [requested])
        parameters = dict(self.profile.get('parameters', {}))
        payload = dict(parameters, model=requested, messages=messages, stream=True, stream_options={'include_usage': True})
        if tools:
            payload['tools'] = tools
        for attempt in range(3):
            if self.deadline is not None and time.monotonic() >= self.deadline:
                raise TransportError('Episode deadline exceeded')
            chunks = 0; content = []; reasoning = []; calls = {}; usage = {}; finish = None; returned = None
            timeout = min(300, max(1, self.deadline-time.monotonic())) if self.deadline else 300
            try:
                with httpx.Client(timeout=httpx.Timeout(timeout, connect=min(20, timeout)), follow_redirects=False) as client:
                    with client.stream('POST', base+'/chat/completions', headers={'Authorization': 'Bearer '+key}, json=payload) as response:
                        if response.status_code in (401, 403):
                            raise IdentityError('Endpoint denied access to the configured model')
                        if response.status_code >= 400:
                            raise TransportError('Upstream HTTP '+str(response.status_code))
                        for line in response.iter_lines():
                            if self.deadline is not None and time.monotonic() >= self.deadline:
                                raise TransportError('Episode deadline exceeded during response')
                            if not line.startswith('data:'):
                                continue
                            text = line[5:].strip()
                            if text == '[DONE]':
                                break
                            try:
                                item = json.loads(text)
                            except ValueError as error:
                                raise ProtocolError('Invalid streaming JSON') from error
                            chunks += 1; self.heartbeat()
                            if item.get('error'):
                                raise TransportError('Provider reported a stream error')
                            if item.get('model'):
                                if item['model'] not in accepted:
                                    raise IdentityError('Unexpected returned model: '+str(item['model']))
                                returned = item['model']
                            if item.get('usage'):
                                usage = item['usage']
                            if not item.get('choices'):
                                continue
                            choice = item['choices'][0]; delta = choice.get('delta') or {}
                            if delta.get('content'):
                                content.append(delta['content'])
                            if delta.get('reasoning_content'):
                                reasoning.append(delta['reasoning_content'])
                            for fallback, call in enumerate(delta.get('tool_calls') or []):
                                state = calls.setdefault(call.get('index', fallback), {'id': None, 'name': '', 'arguments': ''})
                                if call.get('id'):
                                    if state['id'] is not None and state['id'] != call['id']:
                                        raise ProtocolError('Tool-call identifier changed during stream')
                                    state['id'] = call['id']
                                function = call.get('function') or {}
                                state['name'] += function.get('name') or ''
                                state['arguments'] += function.get('arguments') or ''
                            finish = choice.get('finish_reason') or finish
                if returned is None or finish not in ('stop', 'tool_calls'):
                    raise ProtocolError('Missing model identity or incomplete response')
                if not usage.get('prompt_tokens'):
                    raise ProtocolError('Usage is required to enforce the context budget')
                parsed = []
                for _, call in sorted(calls.items()):
                    if not call['id'] or not call['name']:
                        raise ProtocolError('Incomplete tool call')
                    try:
                        arguments = json.loads(call['arguments'])
                    except ValueError as error:
                        raise ProtocolError('Invalid tool arguments') from error
                    if not isinstance(arguments, dict):
                        raise ProtocolError('Tool arguments must be an object')
                    parsed.append({'id': call['id'], 'name': call['name'], 'args': arguments, 'type': 'tool_call'})
                if len({c['id'] for c in parsed}) != len(parsed):
                    raise ProtocolError('Duplicate tool-call identifiers')
                return Reply(''.join(content), ''.join(reasoning), parsed, usage, returned, finish)
            except (httpx.HTTPError, TransportError) as error:
                # Never concatenate or retry a partially consumed model response.
                if chunks or attempt == 2:
                    raise TransportError(str(error)) from error
                time.sleep(min(2**attempt, 4))
        raise AssertionError('Unreachable')
