import json

import torch

# wiki_path = "/mimer/NOBACKUP/groups/snic2022-22-1003/APP/qa-retriever/data/atlas/corpora/wiki/enwiki-dec2018/infobox.jsonl"
# # wiki_path = "/mimer/NOBACKUP/groups/snic2022-22-1003/APP/qa-retriever/data/atlas/corpora/wiki/enwiki-dec2018/text-list-100-sec.jsonl"

# with open(wiki_path, "r", encoding="utf-8") as f:
#     for i, line in enumerate(f):
#         if line.strip():
#             if i > 3:
#                 break

#             data = json.loads(line)
#             print(data)


passage_path = "/mimer/NOBACKUP/groups/snic2022-22-1003/APP/qa-retriever/data/atlas/indices/atlas_nq/wiki/base/passages.0.pt"
x = torch.load(passage_path)
