import json, os
from datasets import Dataset

def load_tiny_instruct(path="data/tiny_instruct.jsonl", split_ratio=0.8):
    rows = [json.loads(line) for line in open(path, "r", encoding="utf-8")]
    # prompt 형식으로 변환
    def to_text(r):
        if r.get("input"):
            return f"### Instruction:\n{r['instruction']}\n### Input:\n{r['input']}\n### Response:\n{r['output']}"
        return f"### Instruction:\n{r['instruction']}\n### Response:\n{r['output']}"
    texts = [{"text": to_text(r)} for r in rows]
    ds = Dataset.from_list(texts)
    n = int(len(ds) * split_ratio)
    return ds.select(range(n)), ds.select(range(n, len(ds)))
