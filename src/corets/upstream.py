"""Fetch the public, MIT-licensed agent at a fixed commit; no benchmark data is bundled."""
from __future__ import annotations
import os
import subprocess
from pathlib import Path
from .io import sha256, write_json

REVISION = '93ac6a4f3af3fe3f86580f62142f47e97e2cc897'
AGENT_SHA256 = 'c7afddb2e2d4ac6479c45368cc313685d68823a7dd9bee9935ca4093b319c142'


def verify_agent(path: Path) -> Path:
    path = Path(path).expanduser().resolve()
    agent = path/'eval/sweqapro/agent.py'
    if not agent.exists() or sha256(agent.read_bytes()) != AGENT_SHA256:
        raise ValueError('Official agent is missing or differs from pinned revision ' + REVISION)
    return path


def fetch_agent(path: Path) -> Path:
    path = Path(path).expanduser().resolve()
    env=dict(os.environ, GIT_TERMINAL_PROMPT='0', GIT_LFS_SKIP_SMUDGE='1')
    def git(*args):
        return subprocess.check_output(['git', *map(str,args)], env=env, stderr=subprocess.PIPE, timeout=300, text=True).strip()
    if path.exists():
        return verify_agent(path)
    url = os.environ.get('CORETS_AGENT_REPO_URL', '')
    if not url or url.startswith('YOUR_'):
        raise ValueError('Set CORETS_AGENT_REPO_URL to the agent repository URL')
    path.parent.mkdir(parents=True,exist_ok=True)
    git('init',path)
    git('-C',path,'remote','add','origin',url)
    git('-C',path,'fetch','--depth','1','origin',REVISION)
    git('-C',path,'sparse-checkout','init','--no-cone')
    git('-C',path,'sparse-checkout','set','--no-cone','/eval/','/LICENSE')
    git('-C',path,'checkout','--detach',REVISION)
    verify_agent(path)
    write_json(path.parent/'agent_source.json',{'url':url,'revision':REVISION,'agent_sha256':AGENT_SHA256})
    return path


def build_tools_image(upstream: Path, context: Path, tag: str = 'corets-tools:93ac6a4') -> None:
    import shutil
    upstream=verify_agent(upstream);context=Path(context).resolve()
    if context.exists() and any(context.iterdir()):
        raise ValueError('Use a new build-context directory')
    context.mkdir(parents=True,exist_ok=True)
    shutil.copytree(upstream/'eval/sweqapro/tools',context/'official_tools')
    (context/'tool_bridge.py').write_text(TOOL_BRIDGE)
    (context/'Dockerfile').write_text('''FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends ripgrep tree && rm -rf /var/lib/apt/lists/*
COPY official_tools /opt/official_tools
COPY tool_bridge.py /opt/tool_bridge.py
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /repo
CMD ["sleep", "infinity"]
''')
    subprocess.run(['docker','build','-t',tag,str(context)],check=True)


TOOL_BRIDGE = '''from pathlib import Path
import importlib.util,json,sys

def load(name):
    path=Path('/opt/official_tools')/(name+'.py')
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return getattr(module,name)

payload=json.load(sys.stdin);call=payload['call'];name=call['name'];args=call.get('args') or {};repo='/repo'
try:
    if name=='semantic_search':
        output=json.dumps(load(name)(term=args['term'],path=args.get('path',repo),python_only=args.get('python_only',False),max_files=args.get('max_files',100),include_lines=args.get('include_lines',False)))
    elif name=='view_codebase':
        output=load(name)(path=args['path'],view_range=args.get('view_range'),concise=args.get('concise',False),python_only=args.get('python_only',False))
    elif name=='execute_readonly_command':output=load(name)(args['command'])
    else:output='ERROR: Unknown tool '+name
except Exception as error:output='ERROR: Tool failed: '+str(error)
print(json.dumps({'output':output}))
'''
