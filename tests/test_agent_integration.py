import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from corets.agent import run_agent
from corets.llm import Reply


@unittest.skipUnless(os.environ.get('CORETS_OFFICIAL_AGENT'), 'Set CORETS_OFFICIAL_AGENT to run upstream integration')
class AgentTest(unittest.TestCase):
    def test_official_agent_exports_visible_tool_trajectory(self):
        class FakeClient:
            def __init__(self, *args, **kwargs): self.calls = 0
            def request(self, messages, tools=None):
                self.calls += 1
                usage = {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120}
                if self.calls == 1:
                    return Reply('I will inspect the repository.', 'private reasoning', [{'id':'call_1', 'name':'execute_readonly_command', 'args':{'command':'ls'}, 'type':'tool_call'}], usage, 'fake', 'tool_calls')
                return Reply('<think>private reasoning</think><finish>The answer is 42.</finish>', '', [], usage, 'fake', 'stop')

        class Completed:
            returncode = 0
            stdout = json.dumps({'output': 'example.py'})

        with TemporaryDirectory() as tmp, patch('corets.agent.subprocess.run', return_value=Completed()):
            config = {'episode_timeout_seconds': 60, 'tools_image': 'test', 'max_iterations': 10, 'max_context_tokens': 32768, 'context_warning_threshold': .825}
            result, sft = run_agent('What is the answer?', Path(tmp), {'model': 'fake'}, config, Path(os.environ['CORETS_OFFICIAL_AGENT']), Path(tmp) / 'output', client_factory=FakeClient)
            self.assertEqual(result['status'], 'success')
            text = json.dumps(sft)
            self.assertIn('tool_response', text)
            self.assertIn('The answer is 42', text)
            self.assertNotIn('private reasoning', text)


if __name__ == '__main__': unittest.main()
