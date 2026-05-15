import ast
import os
import re

nodes_dir = "vishustra_core/nodes"
skip = {"__init__.py", "base_node.py"}

BAD_PATTERNS = [
    r"^\[EOF\]",
    r"^```",
    r"^\[instruction\]",
    r"^\[end\]",
    r"^\[/",
]

print("=" * 60)
print("PHASE 1: Syntax Check")
print("=" * 60)
syntax_errors = []
runtime_issues = []

for fname in sorted(os.listdir(nodes_dir)):
    if not fname.endswith(".py") or fname in skip:
        continue
    fpath = os.path.join(nodes_dir, fname)
    with open(fpath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # Syntax check
    src = "".join(lines)
    try:
        ast.parse(src)
    except SyntaxError as e:
        syntax_errors.append((fname, e.lineno, e.msg))
        print(f"  SYNTAX ERR  {fname}  ->  line {e.lineno}: {e.msg}")
        continue
    
    # Pattern check
    for i, line in enumerate(lines, 1):
        for pat in BAD_PATTERNS:
            if re.match(pat, line.strip()):
                runtime_issues.append((fname, i, line.strip()))
                print(f"  BAD PATTERN {fname}  ->  line {i}: {line.strip()[:60]}")
                break
    else:
        print(f"  OK  {fname}")

print()
print("=" * 60)
print(f"SUMMARY: {len(syntax_errors)} syntax errors, {len(runtime_issues)} pattern issues")
print("=" * 60)

# Try importing each node
print()
print("PHASE 2: Import Check")
print("=" * 60)
import importlib.util, sys

for fname in sorted(os.listdir(nodes_dir)):
    if not fname.endswith(".py") or fname in skip:
        continue
    fpath = os.path.join(nodes_dir, fname)
    module_name = f"vishustra_core.nodes.{fname[:-3]}"
    try:
        spec = importlib.util.spec_from_file_location(module_name, fpath)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        print(f"  IMPORT OK  {fname}")
    except Exception as e:
        print(f"  IMPORT ERR {fname}  ->  {e}")
