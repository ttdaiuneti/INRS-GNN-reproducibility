"""Check a source file against the SHA-256 recorded when a study ran.

After the runs finished, comments and docstrings were shortened in the files
listed in comment_edits.json; their code is unchanged. For these files the
check compares a hash of the code without comments and docstrings with the
same hash of the file as it was when the study ran.
"""
import ast
import hashlib
import io
import json
import tokenize
from pathlib import Path

EDITS_FILE = Path(__file__).resolve().parent / "comment_edits.json"
SCOPES = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def code_sha256(path):
    """SHA-256 of the token stream without comments, blank lines and docstrings."""
    src = Path(path).read_text(encoding="utf-8")
    docstrings = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, SCOPES) and node.body and isinstance(node.body[0], ast.Expr) \
                and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            docstrings.add((node.body[0].lineno, node.body[0].col_offset))
    tokens = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type in (tokenize.COMMENT, tokenize.NL):
            continue
        if tok.type == tokenize.STRING and tok.start in docstrings:
            continue
        tokens.append(tok.string)
    return hashlib.sha256("\0".join(tokens).encode("utf-8")).hexdigest()


def source_ok(root, rel, recorded):
    """True if root/rel matches the recorded SHA-256, directly or through comment_edits.json."""
    path = Path(root) / rel
    if hashlib.sha256(path.read_bytes()).hexdigest() == recorded:
        return True
    edit = json.loads(EDITS_FILE.read_text()).get(rel)
    return bool(edit) and edit["recorded_sha256"] == recorded and code_sha256(path) == edit["code_sha256"]
