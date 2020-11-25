import sys

import torch

src = sys.argv[1]   # weights.th
dest = sys.argv[2]  # modified weights.th

weights = torch.load(src, map_location=torch.device("cpu"))
irrelevant = [k for k in weights.keys() if not k.startswith("_text_field_embedder")]
for k in irrelevant:
    weights.pop(k)
with open(dest, "wb") as f:
    torch.save(weights, f)