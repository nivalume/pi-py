"""Permission modes and rules: the ``pi.permissions`` and ``pi.permission-rules`` documents, and :class:`RulesPolicy`,
the approval policy over them."""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, Literal, TypeAlias, TypedDict, cast

from pi_chord.json import copy_json

from ..documents import define_doc
from ..types import ConversationRecord, RewindableConversationDocToken, SessionDocToken, Tx
from .paths import Anchors
from .policy import ApprovalCall, PolicyDecision
from .rules import Rule, RuleCall, RuleError, allowed_by, is_within, propose_rule

if TYPE_CHECKING:
    from pi_chord.context import Context

    from ..harness.types import HookApi

__all__ = [
    "PERMISSION_MODES",
    "PermissionDefaults",
    "PermissionMode",
    "PermissionRules",
    "PermissionRulesDoc",
    "PermissionsChange",
    "PermissionsDoc",
    "PermissionsState",
    "RulesPolicy",
    "add_session_rule",
    "apply_permissions_change",
    "copy_owner_permissions",
    "validate_rules",
]

#: How much a conversation's calls may do without asking (Claude Code's permission modes):
#:
#: - ``default``: reads inside the working directories run; anything else not allowed by a rule asks.
#: - ``acceptEdits``: also writes inside the working directories run.
#: - ``plan``: commands and writes are denied; reads as in ``default``.
#: - ``dontAsk``: what would ask is denied.
#: - ``bypassPermissions``: everything runs, except what a deny or ask rule names.
PermissionMode: TypeAlias = Literal["default", "acceptEdits", "plan", "dontAsk", "bypassPermissions"]
PERMISSION_MODES: Final[tuple[PermissionMode, ...]] = ("default", "acceptEdits", "plan", "dontAsk", "bypassPermissions")


class PermissionRules(TypedDict, total=False):
    """Rules as written, by what they decide."""

    allow: list[str]
    ask: list[str]
    deny: list[str]


class PermissionsState(TypedDict, total=False):
    """The value of a ``pi.permissions`` document."""

    #: Absent: the policy's default mode.
    mode: PermissionMode
    rules: PermissionRules
    #: More directories reads (and ``acceptEdits`` writes) reach: absolute, or relative to the call's directory.
    directories: list[str]


class PermissionsChange(TypedDict, total=False):
    """A change of a conversation's permissions. A given ``mode`` or ``directories`` replaces the stored one,
    ``None`` clears it, absent keeps it; rules are removed, then added, by their text."""

    mode: PermissionMode | None
    add: PermissionRules
    remove: PermissionRules
    directories: Sequence[str] | None


_LISTS: Final = ("allow", "ask", "deny")

#: Conversation scope, rewindable; a fork copies it as of the fork entry; a complete base on every change.
PermissionsDoc: RewindableConversationDocToken[PermissionsState] = define_doc(
    kind="pi.permissions",
    version=1,
    scope="conversation",
    history="rewindable",
    fork="asOf",
    initial=lambda: {},
    checkpoint_when=lambda *_args: True,
)

#: Session scope: the rules approvals added (``{"type": "approvedWithRule"}``); a complete base on every change.
PermissionRulesDoc: SessionDocToken[PermissionRules] = define_doc(
    kind="pi.permission-rules",
    version=1,
    scope="session",
    initial=lambda: {},
    checkpoint_when=lambda *_args: True,
)


def validate_rules(rules: PermissionRules | None) -> None:
    """Raise :class:`RuleError` for the first rule that does not parse."""
    if rules is None:
        return
    for name in _LISTS:
        for text in rules.get(name, ()):
            Rule.parse(text)


def apply_permissions_change(state: Any, change: PermissionsChange) -> None:
    """Apply ``change`` to a ``pi.permissions`` value (a draft or a dict): removals first, then additions not already
    present; lists that end up empty are dropped."""
    if "mode" in change:
        mode = change["mode"]
        if mode is None:
            state.pop("mode", None)
        else:
            state["mode"] = mode
    if "directories" in change:
        directories = change["directories"]
        if directories:
            state["directories"] = list(directories)
        else:
            state.pop("directories", None)
    add = change.get("add") or {}
    remove = change.get("remove") or {}
    if not any(add.get(name) or remove.get(name) for name in _LISTS):
        return
    rules = state.get("rules")
    if rules is None:
        state["rules"] = {}
        rules = state["rules"]
    for name in _LISTS:
        removed = {text.strip() for text in remove.get(name, ())}
        current = [text for text in rules.get(name, ()) if text.strip() not in removed]
        for text in add.get(name, ()):
            text = text.strip()
            if text not in current:
                current.append(text)
        if current:
            if current != list(rules.get(name, ())):
                rules[name] = current
        else:
            rules.pop(name, None)
    if not rules:
        state.pop("rules", None)


async def add_session_rule(tx: Tx, rule: str) -> None:
    """Add ``rule`` to the session's allow rules (``pi.permission-rules``) unless present."""
    rules = await tx.doc(PermissionRulesDoc)
    allow = rules.get("allow")
    if allow is None:
        rules["allow"] = [rule]
    elif rule not in allow:
        allow.append(rule)


async def copy_owner_permissions(tx: Tx, record: ConversationRecord) -> None:
    """A conversation created with an owner starts with a copy of the owner conversation's ``pi.permissions``; a fork
    gets its copy from the fork policy."""
    owner = record.get("owner")
    if owner is None or "parent" in record:
        return
    source = await tx.doc(PermissionsDoc, owner["conversationId"])
    if not source:
        return
    target = await tx.doc(PermissionsDoc, record["id"])
    target.update(copy_json(dict(source)))  # type: ignore[arg-type]


@dataclass(slots=True)
class PermissionDefaults:
    """What :class:`RulesPolicy` starts from: the mode of a conversation that sets none, rules for every conversation,
    and the anchors of ``/path`` and ``~/path`` rules."""

    mode: PermissionMode = "default"
    rules: PermissionRules = field(default_factory=lambda: cast(PermissionRules, {}))
    directories: Sequence[str] = ()
    #: The directory ``/path`` rules are relative to.
    root: str | None = None
    #: The directory ``~/path`` rules are relative to; default the process's home directory.
    home: str | None = None


#: The memo a decision is kept in, so a rerun before the intent commit decides the same.
_DECISION_MEMO: Final = "pi.permissions.decision"


@dataclass(frozen=True, slots=True)
class _ParsedRules:
    allow: tuple[Rule, ...] = ()
    ask: tuple[Rule, ...] = ()
    deny: tuple[Rule, ...] = ()

    @staticmethod
    def parse(rules: PermissionRules | None, strict: bool) -> _ParsedRules:
        """The rules that parse; ``strict`` raises on the first that does not."""

        def parse_list(texts: Iterable[str]) -> tuple[Rule, ...]:
            parsed: list[Rule] = []
            for text in texts:
                try:
                    parsed.append(Rule.parse(text))
                except RuleError:
                    if strict:
                        raise
            return tuple(parsed)

        rules = rules or {}
        return _ParsedRules(
            parse_list(rules.get("allow", ())), parse_list(rules.get("ask", ())), parse_list(rules.get("deny", ()))
        )


class RulesPolicy:
    """Claude Code's rules and permission modes as an approval policy.

    Deny rules, then ask rules, then the mode, allow rules and the working directories decide; what remains asks with
    a proposed rule (or is denied in ``dontAsk``). Rules come from the defaults, the session's ``pi.permission-rules``
    and the conversation's ``pi.permissions``; the mode from the conversation, else the defaults. A decision is
    memoized in the asking task, so a rerun decides the same.
    """

    def __init__(self, defaults: PermissionDefaults | None = None) -> None:
        """Raise :class:`RuleError` when a default rule does not parse or needs a root that is not set."""
        defaults = defaults if defaults is not None else PermissionDefaults()
        rules = _ParsedRules.parse(defaults.rules, True)
        if defaults.root is None:
            for rule in (*rules.allow, *rules.ask, *rules.deny):
                if rule.needs_root:
                    raise RuleError("no_root", rule.text)
        home = defaults.home
        if home is None:
            expanded = os.path.expanduser("~")
            home = None if expanded == "~" else expanded
        self.defaults = defaults
        self._rules = rules
        self._home = home

    async def decide(self, call: ApprovalCall, api: HookApi, context: Context) -> PolicyDecision:
        decided = await api.memo(_DECISION_MEMO, context)
        if decided is None:
            decided = await api.memo(_DECISION_MEMO, await self._evaluate(call, api, context), context)
        return cast(PolicyDecision, copy_json(decided))

    async def _evaluate(self, call: ApprovalCall, api: HookApi, context: Context) -> PolicyDecision:
        state = await api.snapshot(PermissionsDoc, call.conversation_id, context) or {}
        session = await api.snapshot(PermissionRulesDoc, context)
        sources = (
            self._rules,
            _ParsedRules.parse(session, False),
            _ParsedRules.parse(state.get("rules"), False),
        )
        mode = state.get("mode") or self.defaults.mode
        return self._decide_with(call, mode, sources, state.get("directories", ()))

    def _decide_with(
        self, call: ApprovalCall, mode: PermissionMode, sources: Sequence[_ParsedRules], directories: Sequence[str]
    ) -> PolicyDecision:
        """The decision for ``call`` under ``mode`` with the rule sets ``sources`` and the conversation's extra
        ``directories``."""
        tool = call.call.name
        subject = call.subject
        rule_call = RuleCall(tool, subject, Anchors(cwd=call.cwd, home=self._home, root=self.defaults.root))
        for rule in (rule for rules in sources for rule in rules.deny):
            if rule.restricts(rule_call):
                return {"type": "deny", "reason": f"denied by rule {rule}"}
        for rule in (rule for rules in sources for rule in rules.ask):
            if rule.restricts(rule_call):
                return {"type": "ask", "prompt": f"{rule} requires approval"}
        changes = subject is not None and (subject["type"] == "command" or subject["access"] == "write")
        if mode == "bypassPermissions":
            return {"type": "allow"}
        if mode == "plan" and changes:
            return {"type": "deny", "reason": f"plan mode: {tool} may not run"}
        if call.hook == "allow" or allowed_by((rule for rules in sources for rule in rules.allow), rule_call):
            return {"type": "allow"}
        if subject is not None and subject["type"] == "path":
            path = subject["path"]
            reachable = subject["access"] == "read" or mode == "acceptEdits"
            inside = (call.cwd is not None and is_within(path, call.cwd, None)) or any(
                is_within(path, directory, call.cwd) for directory in (*self.defaults.directories, *directories)
            )
            if reachable and inside:
                return {"type": "allow"}
        if mode == "dontAsk":
            return {"type": "deny", "reason": f"{tool} would need approval (dontAsk mode)"}
        decision: PolicyDecision = {"type": "ask"}
        proposed = propose_rule(tool, subject)
        if proposed is not None:
            decision["proposedRule"] = proposed
        return decision
