import json
import subprocess
from types import SimpleNamespace

import pytest

FILES = {
    "README.md": "# Demo\n\nDemo is a tiny package used to test the repository agent end to end.\n",
    "pyproject.toml": '[project]\nname = "demo"\ndependencies = ["flask>=3", "requests"]\n'
                      '[project.scripts]\ndemo = "demo.cli:main"\n',
    "demo/__init__.py": "from .core import add_numbers\n",
    "demo/core.py": (
        "import os\n"
        "from .util import clamp\n\n\n"
        "def add_numbers(a, b):\n"
        '    """Add two numbers and clamp the result."""\n'
        "    return clamp(a + b, 0, 100)\n\n\n"
        "class Store:\n"
        "    def __init__(self, items=[]):\n"
        "        self.items = items\n\n"
        "    def load(self, path):\n"
        "        try:\n"
        "            return open(path).read()\n"
        "        except:\n"
        "            return None\n"
    ),
    "demo/util.py": (
        "import subprocess\n\n\n"
        "def clamp(x, lo, hi):\n"
        "    if x < lo:\n"
        "        return lo\n"
        "    if x > hi:\n"
        "        return hi\n"
        "    return x\n\n\n"
        "def run(cmd):\n"
        "    return subprocess.run(cmd, shell=True)\n\n\n"
        "API_KEY = 'sk_live_1234567890abcdef'\n"
    ),
    "demo/cli.py": (
        "from .core import add_numbers\n\n\n"
        "def main():\n"
        "    print(add_numbers(1, 2))\n\n\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    ),
    "web/app.js": "import { helper } from './helper';\nconst x = 1;\nexport function start() { return helper(x); }\n",
    "web/helper.js": "export function helper(v) { document.body.innerHTML = v; return v; }\n",
    "tests/test_core.py": "from demo.core import add_numbers\n\n\ndef test_add():\n    assert add_numbers(1, 2) == 3\n",
}


@pytest.fixture()
def demo_repo(tmp_path, monkeypatch):
    root = tmp_path / "demo"
    for rel, text in FILES.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    run = lambda *a: subprocess.run(["git", "-C", str(root), *a], check=True, capture_output=True)
    run("init", "-q")
    run("-c", "user.email=t@t", "-c", "user.name=T", "add", ".")
    run("-c", "user.email=t@t", "-c", "user.name=T", "commit", "-q", "-m", "Initial demo")
    (root / "demo/util.py").write_text(FILES["demo/util.py"].replace("if x > hi", "if x >= hi"))
    run("-c", "user.email=t@t", "-c", "user.name=T", "commit", "-q", "-am", "Fix clamp upper bound")
    ws = tmp_path / "ws"
    from repo_agent import config
    monkeypatch.setattr(config, "WORKSPACE", ws)
    return root


@pytest.fixture()
def demo_index(demo_repo):
    from repo_agent.index import RepoIndex
    return RepoIndex.build(str(demo_repo))


class FakeMessages:
    """Mimics client.beta.messages: returns queued responses and records requests."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.responses.pop(0)


def fake_client(responses):
    msgs = FakeMessages(responses)
    return SimpleNamespace(beta=SimpleNamespace(messages=msgs)), msgs


def text_resp(text, stop="end_turn"):
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], stop_reason=stop,
                           usage=SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=0))


def tool_resp(*calls):
    blocks = [SimpleNamespace(type="tool_use", id=f"tu{i}", name=n, input=inp) for i, (n, inp) in enumerate(calls)]
    return SimpleNamespace(content=blocks, stop_reason="tool_use",
                           usage=SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=0))


def json_resp(obj):
    return text_resp(json.dumps(obj))
