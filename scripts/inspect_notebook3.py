"""Check for directory setup cell and src imports."""
import json

with open(r"notebooks\Full_Pipeline.ipynb", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

print("=== Looking for directory setup cells ===")
for i, c in enumerate(cells):
    src = "".join(c.get("source", []))
    if "mkdir" in src or "Setup Director" in src or "setup_dir" in src:
        print(f"\nCell {i} [{c['cell_type']}]:")
        print(src[:300])
        print("...")

print("\n\n=== All src.* imports ===")
for i, c in enumerate(cells):
    src = "".join(c.get("source", []))
    if "from src." in src:
        print(f"\nCell {i}: {src[:200]}")

print("\n\n=== Check cell 4 (consolidated imports) ===")
print("".join(cells[4].get("source", [])))
