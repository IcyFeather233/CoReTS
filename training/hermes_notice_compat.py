"""Keep user notices following tool observations in the Hermes input turn."""
from swift.llm.template.base import Template
import os

original_prepare = getattr(Template, '_corets_original_prepare', Template._swift_prepare_inputs)


def prepare(self, inputs):
    original_prepare(self, inputs)
    if self.template_backend != 'swift' or self._agent_template != 'hermes' or self.template_meta.template_type != os.environ.get('MODEL_TYPE'):
        return
    messages = inputs.messages
    i = 0
    while i + 1 < len(messages):
        if messages[i]['role'] != 'tool' or messages[i + 1]['role'] != 'user':
            i += 1
            continue
        prompt = messages[i]['content']
        if not isinstance(prompt, list) or not all(isinstance(x, str) for x in prompt):
            raise ValueError('Unexpected tool prompt')
        j = i + 1
        following = []
        while j < len(messages) and messages[j]['role'] == 'user':
            row = messages[j]
            if not isinstance(row['content'], str) or any(v is not None for k, v in row.items() if k not in {'role', 'content'}):
                raise ValueError('Unexpected user metadata')
            following.append(row['content'])
            j += 1
        marker = '</tool_response>'
        index = next((k for k in range(len(prompt)-1, -1, -1) if marker in prompt[k]), None)
        if index is None:
            raise ValueError('Missing tool response boundary')
        position = prompt[index].rfind(marker) + len(marker)
        prompt[index] = prompt[index][:position] + '\n\n' + '\n\n'.join(following) + prompt[index][position:]
        del messages[i+1:j]
        i += 1


Template._corets_original_prepare = original_prepare
Template._swift_prepare_inputs = prepare
