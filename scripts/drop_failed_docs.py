# Drop checkpoint records with failed chunks so --resume re-extracts them.
# Usage: python scripts/drop_failed_docs.py <run_dir>/checkpoints/<name>.extractions.jsonl

import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
kept, dropped = [], []
for line in p.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    rec = json.loads(line)
    meta = rec.get("meta", {})
    if meta.get("chunks_failed"):
        dropped.append(f"{rec['doc_id']} ({meta['chunks_failed']}/{meta.get('n_chunks')} chunks)")
        continue
    kept.append(line)

if not dropped:
    print("no failed docs in checkpoint; nothing to do")
    sys.exit(0)

backup = p.with_suffix(".jsonl.bak")
backup.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
p.write_text("\n".join(kept) + "\n", encoding="utf-8")
print(f"dropped {len(dropped)}: {', '.join(dropped)}")
print(f"kept {len(kept)} records; backup at {backup}")
print("now re-run the same command with --resume to re-extract only those docs")
