"""Permission rule grammar: command splitting, path patterns, rule matching and proposals."""

from __future__ import annotations

import pytest

from pi_durable.approval import (
    Anchors,
    PathPattern,
    PermissionSubject,
    Rule,
    RuleCall,
    RuleError,
    allowed_by,
    apply_permissions_change,
    is_within,
    propose_rule,
    split_command,
    validate_rules,
)
from pi_durable.approval.commands import command_matches, command_stem
from pi_durable.approval.paths import is_within_directory, parent_rule_path


def split(command: str) -> list[str]:
    parts = split_command(command)
    assert parts is not None
    return parts


def test_splits_at_operators_outside_quotes() -> None:
    assert split("ls && echo 'a && b' | wc -l; pwd") == ["ls", "echo 'a && b'", "wc -l", "pwd"]
    assert split("a || b |& c & d\ne") == ["a", "b", "c", "d", "e"]
    assert split('echo "x; y" $(a; b) `c | d`') == ['echo "x; y" $(a; b) `c | d`']
    assert split("make 2>&1 >&2 &> log") == ["make 2>&1 >&2 &> log"]
    assert split("echo a\\;b") == ["echo a\\;b"]
    assert split("  ;  ") == []


def test_unbalanced_commands_do_not_split() -> None:
    assert split_command("echo 'open") is None
    assert split_command("echo $(date") is None
    assert split_command("echo `x") is None


def test_assignments_and_wrappers_are_stripped() -> None:
    assert split('FOO=1 BAR="a b" cargo test') == ["cargo test"]
    assert split("timeout -s KILL 30 nice -n 5 nohup git push") == ["git push"]
    assert split("time -p command -v ls") == ["ls"]
    assert split("stdbuf -o0 tail -f log") == ["tail -f log"]


def test_command_patterns_match_prefixes_and_wildcards() -> None:
    assert command_matches("ls *", "ls")
    assert command_matches("ls *", "ls -la")
    assert not command_matches("ls *", "lsof")
    assert command_matches("npm run test:*", "npm run test")
    assert command_matches("npm run test:*", "npm run test -- --watch")
    assert not command_matches("npm run test:*", "npm run testing")
    assert command_matches("git * --dry-run", "git push origin --dry-run")
    assert command_matches("cargo test", "cargo test")
    assert not command_matches("cargo test", "cargo test --all")
    assert command_matches("*", "anything at all")


def test_stems_skip_options() -> None:
    assert command_stem("git push origin main") == "git push"
    assert command_stem("ls -la") == "ls"
    assert command_stem("pwd") == "pwd"


ANCHORS = Anchors(cwd="/work/project", home="/home/me", root="/repo")


def matches(pattern: str, path: str) -> bool:
    return PathPattern.parse(pattern).matches(path, ANCHORS)


def test_path_anchors_resolve() -> None:
    assert matches("//etc/passwd", "/etc/passwd")
    assert matches("~/.ssh", "/home/me/.ssh/id_rsa")
    assert matches("/config", "/repo/config/app.toml")
    assert matches("./.env", "/work/project/.env")
    assert matches(".env", "/work/project/.env")
    assert not matches(".env", "/work/project/sub/.env")
    assert matches(".", "/work/project/anything")
    assert not PathPattern.parse("/config").matches("/repo/config", Anchors(cwd="/work/project", home="/home/me"))


def test_path_globs_match_segments() -> None:
    assert matches("src/**", "/work/project/src/a/b.rs")
    assert matches("src/**/*.rs", "/work/project/src/a/b.rs")
    assert matches("src/**/*.rs", "/work/project/src/b.rs")
    assert not matches("src/*.rs", "/work/project/src/a/b.toml")
    assert matches("**/.env", "/work/project/deep/down/.env")
    assert matches("file?.txt", "/work/project/file1.txt")
    assert not matches("file?.txt", "/work/project/file10.txt")
    assert matches("src/../docs", "/work/project/docs/x")
    assert matches("*a*b", "/work/project/xaxxb")
    assert not matches("*a*b", "/work/project/xaxxbc")


def test_paths_within_and_parents() -> None:
    assert is_within_directory("/work/project/a", "/work/project")
    assert is_within_directory("/work/project", "/work/project/")
    assert not is_within_directory("/work/projects", "/work/project")
    assert parent_rule_path("/work/project/a.txt") == "//work/project"
    assert parent_rule_path("/a.txt") == "//"
    assert is_within("/work/a", "/work", None)
    assert is_within("/work/docs/a", "docs", "/work")
    assert not is_within("/work/docs/a", "docs", None)


def command(text: str) -> PermissionSubject:
    return {"type": "command", "command": text}


def path(at: str, access: str) -> PermissionSubject:
    return {"type": "path", "path": at, "access": access}  # type: ignore[typeddict-item]


def call(tool: str, subject: PermissionSubject | None) -> RuleCall:
    return RuleCall(tool, subject, Anchors(cwd="/work", home="/home/me"))


def test_rules_parse_and_print_as_written() -> None:
    assert str(Rule.parse("Bash(cargo test:*)")) == "Bash(cargo test:*)"
    assert Rule.parse("  Read(./.env) ").text == "Read(./.env)"
    assert Rule.parse("Bash(*)").restricts(call("bash", command("anything")))
    assert Rule.parse("bash(echo (a))").restricts(call("bash", command("echo (a)")))
    assert Rule.parse("my_tool").restricts(call("my_tool", None))
    assert Rule.parse("Edit(/src/**)").needs_root
    for text, code in [
        ("", "empty"),
        ("Bash(ls", "unbalanced"),
        ("ls)", "unbalanced"),
        ("Bash()", "empty_specifier"),
        ("fetch(x)", "unsupported_specifier"),
        ("(x)", "empty"),
    ]:
        with pytest.raises(RuleError) as raised:
            Rule.parse(text)
        assert raised.value.code == code


def test_deny_and_ask_rules_match_any_sub_command() -> None:
    push = command("cargo fmt && git push origin")
    assert Rule.parse("Bash(git push:*)").restricts(call("bash", push))
    assert not Rule.parse("Bash(git pull:*)").restricts(call("bash", push))
    # Unsplittable: matched against the whole text.
    assert Rule.parse("Bash(echo *)").restricts(call("bash", command("echo 'open")))
    assert Rule.parse("Bash").restricts(call("bash", push))
    # Subject kinds, not tool names.
    assert Rule.parse("bash").restricts(call("other", push))


def test_allow_rules_need_every_sub_command() -> None:
    rules = [Rule.parse("Bash(cargo test:*)"), Rule.parse("Bash(cargo fmt)")]
    assert allowed_by(rules, call("bash", command("cargo fmt && cargo test --all")))
    assert not allowed_by(rules, call("bash", command("cargo fmt && rm -rf target")))
    # An unsplittable command is never allowed.
    assert not allowed_by([Rule.parse("Bash")], call("bash", command("cargo test 'x")))
    assert allowed_by([Rule.parse("bash")], call("bash", command("ls; pwd")))
    assert allowed_by([Rule.parse("other")], call("other", command("anything")))


def test_path_rules_follow_access() -> None:
    read_env = path("/work/.env", "read")
    write_env = path("/work/.env", "write")
    assert Rule.parse("Read(./.env)").restricts(call("read", read_env))
    # A Read deny covers writes.
    assert Rule.parse("Read(./.env)").restricts(call("write", write_env))
    assert not Rule.parse("Edit(./.env)").restricts(call("read", read_env))
    assert allowed_by([Rule.parse("Edit(src/**)")], call("write", path("/work/src/a.rs", "write")))
    assert not allowed_by([Rule.parse("Read(src/**)")], call("write", path("/work/src/a.rs", "write")))
    assert allowed_by([Rule.parse("Write")], call("edit", write_env))


def test_proposals_parse_and_allow_the_call() -> None:
    assert propose_rule("bash", command("git push origin main")) == "Bash(git push:*)"
    assert propose_rule("bash", command("ls -la")) == "Bash(ls:*)"
    assert propose_rule("bash", command("a && b")) is None
    assert propose_rule("write", path("/work/src/a.rs", "write")) == "Edit(//work/src/**)"
    assert propose_rule("read", path("/a.txt", "read")) == "Read(//**)"
    assert propose_rule("fetch", None) == "fetch"
    for proposed in ["Bash(git push:*)", "Edit(//work/src/**)", "Read(//**)"]:
        Rule.parse(proposed)
    assert allowed_by([Rule.parse("Edit(//work/src/**)")], call("write", path("/work/src/a.rs", "write")))
    assert allowed_by([Rule.parse("Bash(git push:*)")], call("bash", command("git push origin main")))


def test_permission_changes_apply_removals_then_additions() -> None:
    state: dict = {"mode": "plan", "rules": {"allow": ["Bash(ls)", "Read"]}, "directories": ["/tmp"]}
    change = {
        "mode": None,
        "add": {"allow": [" Bash(pwd) ", "Read"], "deny": ["Read(./.env)"]},
        "remove": {"allow": ["Bash(ls)"]},
        "directories": ["/data"],
    }
    validate_rules(change["add"])  # type: ignore[arg-type]
    apply_permissions_change(state, change)  # type: ignore[arg-type]
    assert state == {"rules": {"allow": ["Read", "Bash(pwd)"], "deny": ["Read(./.env)"]}, "directories": ["/data"]}
    apply_permissions_change(state, {"remove": {"allow": ["Read", "Bash(pwd)"], "deny": ["Read(./.env)"]}})
    assert state == {"directories": ["/data"]}
    apply_permissions_change(state, {"directories": None, "mode": "acceptEdits"})
    assert state == {"mode": "acceptEdits"}
    with pytest.raises(RuleError):
        validate_rules({"ask": ["Bash("]})
