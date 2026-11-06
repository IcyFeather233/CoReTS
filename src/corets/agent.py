"""The pinned SWE-QA-Pro agent with Docker tools and visible Hermes export."""
from __future__ import annotations
import json
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from .io import redact, write_json
from .llm import Client
from .upstream import verify_agent


def visible_text(content):
    if isinstance(content, str):
        return content
    return ''.join(x.get('text', '') for x in content or [] if isinstance(x, dict) and x.get('type') == 'text')


class Events(list):
    def __init__(self, path: Path):
        super().__init__(); path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path; self.stream = path.open('a', encoding='utf-8')
    def emit(self, item):
        self.stream.write(redact(json.dumps(item, ensure_ascii=False, default=str))+'\n'); self.stream.flush()
    def append(self, item):
        super().append(item); self.emit(item)
    def close(self):
        self.stream.close()


class History:
    def __init__(self): self.responses = []
    @staticmethod
    def key(message):
        return visible_text(message.content), json.dumps(getattr(message, 'tool_calls', []) or [], sort_keys=True)
    def restore(self, messages):
        out = []; cursor = 0
        for message in messages:
            chosen = message
            if message.type == 'ai':
                for i in range(cursor, len(self.responses)):
                    if self.key(self.responses[i]) == self.key(message):
                        chosen = self.responses[i]; cursor = i+1; break
            out.append(chosen)
        return out


def serialize(messages):
    result=[]
    for message in messages:
        role={'system':'system','human':'user','ai':'assistant','tool':'tool'}[message.type]
        row={'role':role,'content':visible_text(message.content)}
        if role=='assistant':
            if 'reasoning_content' in message.additional_kwargs:
                row['reasoning_content']=message.additional_kwargs['reasoning_content']
            calls=getattr(message,'tool_calls',[]) or []
            if calls:row['tool_calls']=[{'id':c['id'],'type':'function','function':{'name':c['name'],'arguments':json.dumps(c['args'],ensure_ascii=False)}} for c in calls]
        if role=='tool':row['tool_call_id']=message.tool_call_id
        result.append(row)
    return result


class Recorder:
    def __init__(self, client, history, events, tools):
        self.client=client;self.history=history;self.events=events;self.tools=tools
    def invoke(self, messages, **kwargs):
        from langchain_core.messages import AIMessage
        actual=self.history.restore(messages);event={'kind':'llm','role':'with_tools' if self.tools else 'forced_finish','messages':[m.model_dump(mode='json') for m in actual]}
        started=time.monotonic()
        try:
            reply=self.client.request(serialize(actual),self.tools)
            usage={'input_tokens':int(reply.usage.get('prompt_tokens',0)),'output_tokens':int(reply.usage.get('completion_tokens',0)),'total_tokens':int(reply.usage.get('total_tokens',0))}
            if not usage['total_tokens']:usage['total_tokens']=usage['input_tokens']+usage['output_tokens']
            response=AIMessage(content=reply.content,tool_calls=reply.tool_calls,additional_kwargs={'reasoning_content':reply.reasoning_content},response_metadata={'model':reply.model,'finish_reason':reply.finish_reason,'token_usage':reply.usage},usage_metadata=usage)
            event['response']=response.model_dump(mode='json');self.history.responses.append(response);return response
        except Exception as error:
            event.update(error_type=type(error).__name__,error=str(error));raise
        finally:
            event['elapsed_seconds']=time.monotonic()-started;self.events.append(event)


def export_hermes(events, schemas):
    calls=[e for e in events if e.get('kind')=='llm' and e.get('response')]
    if not calls:return None,'no_successful_response'
    source=calls[-1]['messages']+[calls[-1]['response']];messages=[];pending={};has_tools=False
    for message in source:
        kind=message['type'];text=visible_text(message.get('content'))
        if kind in ('system','human'):messages.append({'role':'system' if kind=='system' else 'user','content':text})
        elif kind=='ai':
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
            if '<think>' in text or '</think>' in text:return None,'unbalanced_think_tag'
            if text:messages.append({'role':'assistant','content':text})
            for call in message.get('tool_calls',[]):
                if call['id'] in pending:return None,'duplicate_tool_call'
                pending[call['id']]=call;has_tools=True
        elif kind=='tool':
            call=pending.pop(message.get('tool_call_id'),None)
            if call is None:return None,'unmatched_tool_response'
            messages.append({'role':'tool_call','content':json.dumps({'name':call['name'],'arguments':call['args']},ensure_ascii=False)})
            if call['name']=='semantic_search':
                try:value=json.loads(text)
                except ValueError:value={'error':text}
            else:
                field={'execute_readonly_command':'command_output','view_codebase':'view_content'}.get(call['name'])
                if field is None:return None,'unknown_tool'
                value={field:text}
            messages.append({'role':'tool_response','content':json.dumps(value,ensure_ascii=False)})
    if pending:return None,'unanswered_tool_call'
    if not has_tools:return None,'no_structured_tools'
    if not messages or messages[-1]['role']!='assistant' or '<finish>' not in messages[-1]['content'] or '</finish>' not in messages[-1]['content']:
        return None,'missing_final_finish'
    return {'tools':schemas,'messages':messages},'eligible'


def run_agent(question: str, snapshot: Path, profile: dict, config: dict, upstream: Path, output: Path, *, client_factory=Client):
    upstream=verify_agent(upstream);sys.path.insert(0,str(upstream/'eval'))
    os.environ['LANGSMITH_TRACING']='false';os.environ['LANGCHAIN_TRACING_V2']='false'
    from sweqapro.agent import ToolCallingAgent, _build_messages
    from sweqapro.tool_schemas import TOOL_SCHEMAS
    from langchain_core.messages import HumanMessage
    output=Path(output);output.mkdir(parents=True,exist_ok=True);events=Events(output/'events.jsonl')
    class IsolatedAgent(ToolCallingAgent):
        def _run_tool(self,call,repo_path):
            started=time.monotonic()
            try:
                p=subprocess.run(['docker','exec','-i',container,'python3','/opt/tool_bridge.py'],input=json.dumps({'call':call,'repo_path':'/repo'}),text=True,capture_output=True,timeout=120)
                value=json.loads(p.stdout)['output'] if p.returncode==0 else 'ERROR: isolated tool process failed'
            except (subprocess.TimeoutExpired,ValueError):value='ERROR: isolated tool failure'
            events.append({'kind':'tool','call':call,'output':value,'elapsed_seconds':time.monotonic()-started})
            return value
        def _force_finish(self,state,reason):
            original=self.llm_no_tools
            notice=f'--- {reason.upper()} ---\nThis is the final allowed turn. You MUST NOT call any tools. Please provide your best possible final answer now following OUTPUT PROTOCOL using the <finish> tag based on all information gathered so far.'
            events.emit({'kind':'force_finish_notice','reason':reason})
            class FullHistory:
                def invoke(inner,messages,**kwargs):
                    if len(messages)==2 and messages[0].content=='You are an expert code analyst. Provide a final answer based on collected information.':
                        messages=_build_messages(state,turn_human=HumanMessage(content=notice))
                    return original.invoke(messages,**kwargs)
            self.llm_no_tools=FullHistory()
            try:return super()._force_finish(state,reason)
            finally:self.llm_no_tools=original
    container='corets-'+uuid.uuid4().hex[:20];started=time.monotonic();deadline=started+config['episode_timeout_seconds']
    try:
        subprocess.run(['docker','run','-d','--name',container,'--label','corets.ephemeral=true','--network','none','--read-only','--cap-drop','ALL','--security-opt','no-new-privileges','--pids-limit','128','--memory','1g','--cpus','1','--user','65534:65534','--tmpfs','/tmp:rw,noexec,nosuid,size=32m','-v',str(Path(snapshot).resolve())+':/repo:ro',config['tools_image']],capture_output=True,check=True,timeout=60)
        client=client_factory(profile,deadline=deadline);history=History()
        agent=IsolatedAgent(llm=Recorder(client,history,events,TOOL_SCHEMAS),llm_no_tools=Recorder(client,history,events,None),provider='openai',model_label=profile['model'],max_iterations=config['max_iterations'],history_window=config['max_iterations'],max_context_length=config['max_context_tokens'],context_warning_threshold=config['context_warning_threshold'],quiet=True)
        answer=agent.query(question=question,repo_path='/repo');sft,structural=export_hermes(events,TOOL_SCHEMAS)
        result={'status':answer.get('status','failed'),'agent_result':answer,'structural_status':structural,'model':profile['model'],'model_calls':sum(e['kind']=='llm' for e in events),'tool_calls':sum(e['kind']=='tool' for e in events),'tool_errors':sum(e['kind']=='tool' and str(e.get('output','')).lstrip().startswith('ERROR:') for e in events),'wall_seconds':time.monotonic()-started}
        if result['status']=='success' and sft:
            write_json(output/'sft.json',sft)
        write_json(output/'result.json',result);return result,sft
    except Exception as error:
        result={'status':'infrastructure_error','error_type':type(error).__name__,'error':str(error),'wall_seconds':time.monotonic()-started}
        write_json(output/'result.json',result);raise
    finally:
        subprocess.run(['docker','rm','-f',container],capture_output=True,timeout=30)
        events.close()
