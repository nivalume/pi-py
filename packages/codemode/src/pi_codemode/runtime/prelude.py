"""Python evaluated inside the Monty sandbox before the script runs (port of ``prelude-source.ts``).

The prelude is fed to the session as its own snippet, so the script that follows is a
separate snippet whose line numbers are its own. It defines what the script sees:

- ``tools`` (an object with one attribute per tool, ``tools["raw-name"]`` and ``in``),
  ``ALL_TOOLS``, the globals and their namespaces;
- ``text()``, ``image()``, ``print()``, ``exit()``, ``store()`` and ``load()``.

Bridge to the host. Two channels, both plain data:

- *Calls* are calls of names the sandbox does not define, which Monty hands to the host as
  function snapshots: ``__codemode_call__(index, args, kwargs)`` for tool ``index``,
  ``__codemode_global__(index, args, kwargs)`` for a global, ``__codemode_exit__()`` and
  ``__codemode_fail__(name, message)``. The host answers a tool or global call with a
  future at once, so the call starts eagerly, like a JavaScript promise, and the script
  awaits it; it never answers ``exit`` and ``fail``, which end the run.
- *Records* are one JSON array per line written with the builtin ``print``: ``["t", text]``
  a text item, ``["c", text]`` a console item, ``["i", data, mime_type]`` an image,
  ``["s", key, json]`` a store write and ``["d", key]`` a store deletion. Monty flushes
  printed output before every host call, so records arrive in order with the calls.

The helpers are closures of one factory, so their state (output counters, the store) and the
builtins and module functions they use are out of the script's reach: the script shares the
prelude's globals, and rebinding ``len`` or ``print`` there only changes what the script itself
sees. Monty's modules are read-only. Nothing here is a trust boundary, though: the host checks
every record and call it receives.

Inputs bound before the prelude runs: ``__codemode_tools__`` (``[name, js_name,
description]`` lists), ``__codemode_globals__`` (``[name, spread]``) and
``__codemode_store__`` (key to JSON text).
"""

from __future__ import annotations

import json
from typing import Final

__all__ = [
    "MAX_OUTPUT_CHARS",
    "MAX_OUTPUT_ITEMS",
    "MAX_STORE_TOTAL_CHARS",
    "MAX_STORE_VALUE_CHARS",
    "PRELUDE_SOURCE",
]

MAX_STORE_VALUE_CHARS: Final = 256 * 1024
MAX_STORE_TOTAL_CHARS: Final = 1024 * 1024
#: Output one script may produce with ``text()``, ``image()`` and ``print()``: characters of
#: text and base64 image data, and items. The host keeps all output until the script ends, so
#: without a limit a script that prints in a loop grows the host's memory until it crashes.
#: The item limit covers loops that print empty strings.
MAX_OUTPUT_CHARS: Final = 16 * 1024 * 1024
MAX_OUTPUT_ITEMS: Final = 100_000

IMAGE_HELPER_EXPECTS: Final = (
    "image expects a non-empty image URL string, a dict with image_url, "
    'or an image block {"type": "image", "data": ..., "mimeType": ...}'
)
OUTPUT_LIMIT_MESSAGE: Final = (
    f"script output exceeded the limit of {MAX_OUTPUT_CHARS} characters or {MAX_OUTPUT_ITEMS} text(), image(), "
    "and print() calls. Print a summary instead, or write large data to a file with a tool."
)
STORE_HINT: Final = (
    "store() is for small state such as IDs or summaries. Show images with image(), keep large data in "
    "variables, or write it to a file with a tool."
)
TOOLS_HINT: Final = "ALL_TOOLS lists every tool; search_tools(query) finds tools by topic."

_TEMPLATE: Final = r"""
def __codemode_prelude__(tool_entries, global_entries, stored_json):
    import json
    import re

    emit = print
    dumps = json.dumps
    loads = json.loads
    re_sub = re.sub
    re_fullmatch = re.fullmatch
    re_match = re.match
    _len = len
    _str = str
    _isinstance = isinstance
    _dict = dict
    _list = list
    _tuple = tuple
    _sum = sum
    _enumerate = enumerate

    def record(entry):
        emit(dumps(entry, ensure_ascii=False))

    counters = {"chars": 0, "items": 0}

    # Past the output limits the script fails: the host ends it at this call, so catching the error does
    # not resume output.
    def output(kind, data, mime_type=None):
        counters["chars"] += _len(data)
        counters["items"] += 1
        if counters["chars"] > @MAX_OUTPUT_CHARS@ or counters["items"] > @MAX_OUTPUT_ITEMS@:
            __codemode_fail__("ValueError", @OUTPUT_LIMIT_MESSAGE@)
        if mime_type is None:
            record([kind, data])
        else:
            record([kind, data, mime_type])

    def text(value):
        if _isinstance(value, _str):
            rendered = value
        else:
            try:
                rendered = dumps(value, separators=(",", ":"), ensure_ascii=False)
            except Exception as error:
                raise TypeError(_str(error))
        output("t", rendered)

    def print_(*values, sep=" ", end="\n", file=None, flush=False):
        if sep is None:
            sep = " "
        if end is None:
            end = "\n"
        line = sep.join([_str(value) for value in values]) + end
        if line.endswith("\n"):
            line = line[:-1]
        output("c", line)

    def image_url(value):
        if _isinstance(value, _str):
            return value
        if not _isinstance(value, _dict):
            raise TypeError(@IMAGE_HELPER_EXPECTS@)
        if value.get("image_url") is not None:
            url = value["image_url"]
            if not _isinstance(url, _str):
                raise TypeError(@IMAGE_HELPER_EXPECTS@)
            return url
        kind = value.get("type")
        if not _isinstance(kind, _str):
            raise TypeError(@IMAGE_HELPER_EXPECTS@)
        if kind != "image":
            raise TypeError('image only accepts image blocks, got "' + kind + '"')
        data = value.get("data")
        if not _isinstance(data, _str) or data == "":
            raise TypeError("image expected image block data")
        if data.lower().startswith("data:"):
            return data
        return "data:;base64," + data

    # Base64 of the signatures of the formats providers accept inline (PNG, JPEG except JPEG-LS, GIF,
    # "RIFF....WEBP"). Signatures start at byte 0, so their encodings are prefixes.
    signatures = (
        ("image/png", r"iVBORw0KGg"),
        ("image/jpeg", r"/9j/(?!9)"),
        ("image/gif", r"R0lGOD[dl]h"),
        ("image/webp", r"UklG.{8}RUJQ"),
    )

    def image(value):
        url = image_url(value)
        if url == "":
            raise TypeError(@IMAGE_HELPER_EXPECTS@)
        colon = url.find(":")
        scheme = "" if colon == -1 else url[:colon].lower()
        if scheme == "http" or scheme == "https":
            raise TypeError("remote image URLs are not supported in tool outputs. Pass a base64 data URI instead")
        comma = url.find(",")
        header = [] if comma == -1 else url[colon + 1 : comma].split(";")
        if scheme != "data" or comma == -1 or not [part for part in header[1:] if part.lower() == "base64"]:
            raise TypeError("invalid image output. Pass a base64 data URI instead")
        # Providers reject the whole request on a bad image, and a persisted image block would be resent on
        # every later turn. Line breaks from wrapped base64 are dropped. The declared type is ignored in favor
        # of the detected one, as providers also reject mismatches.
        data = re_sub(r"\s+", "", url[comma + 1 :])
        if _len(data) % 4 != 0 or re_fullmatch(r"[A-Za-z0-9+/]+={0,2}", data) is None:
            raise TypeError("invalid image output. The image data is not valid base64 (truncated or corrupted?)")
        head = data[:16]
        detected = None
        for mime_type, pattern in signatures:
            if detected is None and re_match(pattern, head) is not None:
                detected = mime_type
        if detected is None:
            raise TypeError("invalid image output. The image data is not a PNG, JPEG, GIF, or WebP image")
        output("i", data, detected)

    def exit_():
        __codemode_exit__()

    # key -> JSON text. Sizes count key and JSON characters.
    stored = _dict(stored_json)
    store_state = {"chars": _sum([_len(key) + _len(encoded) for key, encoded in stored.items()])}

    def check_key(name, key):
        if not _isinstance(key, _str):
            raise TypeError(name + "() key must be a string")

    def store(key, value):
        check_key("store", key)
        quoted = dumps(key, ensure_ascii=False)
        previous = _len(key) + _len(stored[key]) if key in stored else 0
        if value is None:
            if key in stored:
                stored.pop(key)
            store_state["chars"] -= previous
            record(["d", key])
            return None
        try:
            encoded = dumps(value, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        except Exception as error:
            raise TypeError("store(" + quoted + ") value is not JSON-serializable: " + _str(error))
        if _len(encoded) > @MAX_STORE_VALUE_CHARS@:
            raise ValueError(
                "store(" + quoted + ") value has " + _str(_len(encoded))
                + " characters of JSON, more than the limit of @MAX_STORE_VALUE_CHARS@. " + @STORE_HINT@
            )
        following = store_state["chars"] - previous + _len(key) + _len(encoded)
        if following > @MAX_STORE_TOTAL_CHARS@:
            raise ValueError(
                "store is full: stored values would exceed @MAX_STORE_TOTAL_CHARS@ characters of JSON. "
                + "Delete keys with store(key, None). " + @STORE_HINT@
            )
        stored[key] = encoded
        store_state["chars"] = following
        record(["s", key, encoded])
        return None

    def load(key):
        check_key("load", key)
        encoded = stored.get(key)
        return None if encoded is None else loads(encoded)

    # label -> member -> function, and label -> the names listed in errors
    members = {"tools": {}}
    member_names = {"tools": []}

    # Reading a member that does not exist names the close matches. Monty has no __getattr__, so attribute
    # reads get Monty's own AttributeError, which the host rewrites the same way when the script does not
    # catch it.
    def comparable(name):
        return re_sub(r"[^a-z0-9]", "", name.lower())

    def missing(label, member):
        names = member_names[label]
        wanted = comparable(member)
        close = [name for name in names if comparable(name) == wanted]
        if not close and wanted:
            close = [name for name in names if comparable(name) in wanted or wanted in comparable(name)]
        message = label + "." + member + " does not exist."
        if close:
            message += " Did you mean " + ", ".join([label + "." + name for name in close[:5]]) + "?"
        elif _len(names) <= 20:
            message += " Available: " + ", ".join(names) + "."
        if label == "tools":
            message += " " + @TOOLS_HINT@
        message += ' Check for a member with "' + member + '" in ' + label + "."
        return message

    def lookup(label, member):
        if _isinstance(member, _str) and member in members[label]:
            return members[label][member]
        raise AttributeError(missing(label, _str(member)))

    def has(label, member):
        return _isinstance(member, _str) and member in members[label]

    def tool_function(index):
        def call(args=None, /, **kwargs):
            return __codemode_call__(index, args, kwargs)

        return call

    def global_function(index):
        def call(args=None, /, **kwargs):
            return __codemode_global__(index, args, kwargs)

        return call

    def spread_function(index):
        def call(*args, **kwargs):
            return __codemode_global__(index, _list(args), kwargs)

        return call

    all_tools = []
    for index, entry in _enumerate(tool_entries):
        function = tool_function(index)
        # The first tool wins when two names normalize to the same identifier.
        if entry[1] not in members["tools"]:
            members["tools"][entry[1]] = function
            member_names["tools"].append(entry[1])
            all_tools.append({"name": entry[1], "description": entry[2]})
        if entry[0] not in members["tools"]:
            members["tools"][entry[0]] = function

    plain = []
    for index, entry in _enumerate(global_entries):
        function = spread_function(index) if entry[1] else global_function(index)
        parts = entry[0].split(".")
        if _len(parts) == 1:
            plain.append((parts[0], function))
        else:
            if parts[0] not in members:
                members[parts[0]] = {}
                member_names[parts[0]] = []
            members[parts[0]][parts[1]] = function
            member_names[parts[0]].append(parts[1])

    return {
        "text": text,
        "print": print_,
        "image": image,
        "exit": exit_,
        "store": store,
        "load": load,
        "lookup": lookup,
        "has": has,
        "all_tools": _tuple(all_tools),
        "attributes": {label: [(name, members[label][name]) for name in member_names[label]] for label in members},
        "plain": plain,
    }


__codemode__ = __codemode_prelude__(__codemode_tools__, __codemode_globals__, __codemode_store__)
__codemode_lookup__ = __codemode__["lookup"]
__codemode_has__ = __codemode__["has"]
text = __codemode__["text"]
print = __codemode__["print"]
image = __codemode__["image"]
exit = __codemode__["exit"]
store = __codemode__["store"]
load = __codemode__["load"]
ALL_TOOLS = __codemode__["all_tools"]

# A namespace is an instance of a class named like it, so Monty's own AttributeError names it. exec() runs at
# the top level, where its assignments bind globals.
for __codemode_label__ in __codemode__["attributes"]:
    exec(
        (
            "class @@:\n"
            "    def __getitem__(self, member):\n"
            "        return __codemode_lookup__('@@', member)\n"
            "    def __contains__(self, member):\n"
            "        return __codemode_has__('@@', member)\n"
            "    def __repr__(self):\n"
            "        return '<codemode namespace @@>'\n"
            "@@ = @@()\n"
            "for __codemode_member__ in __codemode__['attributes']['@@']:\n"
            "    setattr(@@, __codemode_member__[0], __codemode_member__[1])\n"
        ).replace("@@", __codemode_label__)
    )
for __codemode_index__, __codemode_entry__ in enumerate(__codemode__["plain"]):
    exec(__codemode_entry__[0] + " = __codemode__['plain'][" + str(__codemode_index__) + "][1]")
"""


def _render() -> str:
    source = _TEMPLATE
    for name, value in (
        ("MAX_OUTPUT_CHARS", str(MAX_OUTPUT_CHARS)),
        ("MAX_OUTPUT_ITEMS", str(MAX_OUTPUT_ITEMS)),
        ("MAX_STORE_VALUE_CHARS", str(MAX_STORE_VALUE_CHARS)),
        ("MAX_STORE_TOTAL_CHARS", str(MAX_STORE_TOTAL_CHARS)),
        ("OUTPUT_LIMIT_MESSAGE", json.dumps(OUTPUT_LIMIT_MESSAGE)),
        ("IMAGE_HELPER_EXPECTS", json.dumps(IMAGE_HELPER_EXPECTS)),
        ("STORE_HINT", json.dumps(STORE_HINT)),
        ("TOOLS_HINT", json.dumps(TOOLS_HINT)),
    ):
        source = source.replace(f"@{name}@", value)
    return source.lstrip("\n")


#: The prelude, fixed for a given pi_codemode version; its hash is part of a deterministic run's fingerprint.
PRELUDE_SOURCE: Final = _render()
