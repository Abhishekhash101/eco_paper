"""Check for setup_directories cell."""
import json

with open(r"notebooks\Full_Pipeline.ipynb", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

# Look for setup_directories in metadata IDs or source
for i, c in enumerate(cells):
    meta = c.get("metadata", {})
    cell_id = meta.get("id", "")
    src = "".join(c.get("source", []))
    if "setup_dir" in cell_id.lower() or "subdir" in src.lower() or ("data/processed" in src and "mkdir" in src):
        print(f"Cell {i} (id={cell_id}):")
        print(src[:400])
        print("---")

# Also check for any cell that has all four directories mentioned
for i, c in enumerate(cells):
    src = "".join(c.get("source", []))
    if "data/processed" in src and "data/market" in src:
        print(f"\nCell {i} mentions both data/processed and data/market:")
        print(src[:400])
        print("---")
