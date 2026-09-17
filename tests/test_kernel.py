"""Tests for the VishustraKernel facade, including runtime skill registration."""

import json
from pathlib import Path

import pytest

from vishustra_core.kernel.executor import VishustraKernel, VishustraKernelError
from vishustra_core.kernel.skills import SKILLS
from vishustra_core.kernel.synthesizer import synthesize


@pytest.fixture(autouse=True)
def isolate_runtime_skills(tmp_path, monkeypatch):
    """Keep test-refistered skills out of the real repo runtime_skills.json."""
    import vishustra_core.kernel.skills as skills_module

    fake_path = tmp_path / "runtime_skills.json"
    monkeypatch.setattr(skills_module, "_RUNTIME_SKILLS_PATH", str(fake_path))
    yield fake_path
    if fake_path.exists():
        fake_path.unlink()


NODE_SOURCE = '''\
from typing import Any, Dict
from vishustra_core.nodes.base_node import BaseNode

class ShoutifierNode(BaseNode):
    """Make the text shout loudly."""

    @property
    def node_name(self) -> str:
        return "Shoutifier"

    def process(self, data, context):
        if not isinstance(data, str):
            raise TypeError("need a string")
        return data.upper() + "!!!"
'''


def test_register_node_learns_an_external_module(tmp_path, isolate_runtime_skills):
    node_file = tmp_path / "shoutifier.py"
    node_file.write_text(NODE_SOURCE, encoding="utf-8")

    kernel = VishustraKernel()
    skill_id = kernel.register_node(str(node_file))
    assert skill_id == "shoutifier"

    # Registry + synthesizer both know it now.
    assert skill_id in SKILLS
    plan = synthesize("make this text shout loudly")
    assert plan["pipeline"] == ["shoutifier"]

    # Executes end to end.
    report = kernel.process("make this text shout loudly", "hello world")
    assert report["result"]["final"] == "HELLO WORLD!!!"

    # Persisted to the isolated runtime store.
    saved = json.loads(isolate_runtime_skills.read_text(encoding="utf-8"))
    assert skill_id in saved
    assert "shout" in saved[skill_id]["keywords"]


def test_register_node_derives_suggestive_keywords(tmp_path, isolate_runtime_skills):
    node_file = tmp_path / "shoutifier.py"
    node_file.write_text(NODE_SOURCE, encoding="utf-8")

    skill_id = VishustraKernel().register_node(str(node_file))
    keywords = SKILLS[skill_id]["keywords"]
    assert "shout" in keywords
    assert "text" not in keywords  # stopword filtered


def test_register_node_rejects_file_without_node(tmp_path):
    node_file = tmp_path / "plain.py"
    node_file.write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(VishustraKernelError):
        VishustraKernel().register_node(str(node_file))


def test_register_node_rejects_missing_file():
    with pytest.raises(FileNotFoundError):
        VishustraKernel().register_node(str(Path("does_not_exist_xyz.py")))


def test_registered_skill_does_not_pollute_real_store():
    """The rest of the suite should never see a runtime_skills.json in the repo."""
    from vishustra_core.kernel.skills import _RUNTIME_SKILLS_PATH

    assert not Path(_RUNTIME_SKILLS_PATH).exists()