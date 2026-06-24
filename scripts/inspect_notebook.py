"""Quick script to inspect notebook cell structure."""
import json

with open(r"notebooks\Full_Pipeline.ipynb", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]
print(f"Total cells: {len(cells)}")
print()

for i, c in enumerate(cells[:40]):
    ct = c["cell_type"]
    src = c.get("source", [])
    first_line = src[0].strip() if src else "(empty)"
    print(f"Cell {i:2d} [{ct:8s}]: {first_line[:80]}")
