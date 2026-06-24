"""Show full source of cells 3, and find any directory setup cell."""
import json

with open(r"notebooks\Full_Pipeline.ipynb", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

print("=== Cell 3 (project root) ===")
print("".join(cells[3].get("source", [])))

# Check if there's a cell between 3 and 5 that handles directory setup
print("\n\n=== Cell metadata IDs for cells 0-10 ===")
for i in range(min(10, len(cells))):
    meta = cells[i].get("metadata", {})
    cell_id = meta.get("id", "no-id")
    print(f"Cell {i}: id={cell_id}")
