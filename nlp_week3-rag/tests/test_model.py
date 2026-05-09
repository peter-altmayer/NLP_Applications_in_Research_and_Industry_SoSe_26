import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import build_prompt


def test_build_prompt_no_context_structure():
    messages = build_prompt("Who invented the telephone?", [])
    roles = [m["role"] for m in messages]
    assert "system" in roles
    assert "user" in roles


def test_build_prompt_no_context_content():
    messages = build_prompt("Who invented the telephone?", [])
    user_msg = next(m["content"] for m in messages if m["role"] == "user")
    assert "Who invented the telephone?" in user_msg
    assert "Context" not in user_msg


def test_build_prompt_with_context_includes_docs():
    docs = ["Alexander Graham Bell invented the telephone.", "Bell was born in 1847."]
    messages = build_prompt("Who invented the telephone?", docs)
    user_msg = next(m["content"] for m in messages if m["role"] == "user")
    assert "Alexander Graham Bell" in user_msg
    assert "Bell was born in 1847" in user_msg


def test_build_prompt_with_context_numbered():
    docs = ["doc one", "doc two"]
    messages = build_prompt("Q?", docs)
    user_msg = next(m["content"] for m in messages if m["role"] == "user")
    assert "1." in user_msg
    assert "2." in user_msg


def test_build_prompt_returns_list_of_role_content_dicts():
    messages = build_prompt("Q?", ["doc"])
    assert isinstance(messages, list)
    for m in messages:
        assert set(m.keys()) == {"role", "content"}
