import ast
import os

nodes_dir = "vishustra_core/nodes"
skip = {"__init__.py", "base_node.py"}

ok = []
bad = []

for fname in sorted(os.listdir(nodes_dir)):
    if not fname.endswith(".py") or fname in skip:
        continue
    fpath = os.path.join(nodes_dir, fname)
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            src = f.read()
        ast.parse(src)
        ok.append(fname)
        print(f"  OK  {fname}")
    except SyntaxError as e:
        bad.append((fname, e))
        print(f"  ERR {fname}  ->  line {e.lineno}: {e.msg}")

print(f"\nSummary: {len(ok)} OK, {len(bad)} ERRORS")
