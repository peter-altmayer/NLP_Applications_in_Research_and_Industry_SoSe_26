# Clean Re-run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the stale `bert-score` dependency, enable faithfulness scoring, extend the Drive purge utility to include checkpoints, and clear notebook outputs — leaving the repo ready for a clean full Colab re-run.

**Architecture:** Four targeted edits across two files (`requirements.txt` and `notebooks/colab_runner.ipynb`). No new files, no structural changes. All edits are pure text replacements verifiable by reading the file back.

**Tech Stack:** Python 3, Jupyter notebook (JSON), Git

---

### Task 1: Remove `bert-score` and `scipy` from `requirements.txt`

**Files:**
- Modify: `nlp_medical_qa/requirements.txt`

The code switched to `sentence-transformers` for BERTScore (commit `c19c5cc`). `bert-score>=0.3.13` is unused and crashes with `transformers>=4.40`. `scipy` was only listed as its indirect dep.

- [ ] **Step 1: Edit `requirements.txt`**

Open `nlp_medical_qa/requirements.txt`. Replace:

```
# Evaluation
rouge-score>=0.1.2
bert-score>=0.3.13
scipy>=1.11             # indirect dep of bert-score
```

With:

```
# Evaluation
rouge-score>=0.1.2
```

- [ ] **Step 2: Verify**

Read `nlp_medical_qa/requirements.txt`. Confirm:
- `bert-score` is gone
- `scipy` is gone
- `rouge-score>=0.1.2` is still present
- All other lines are unchanged

- [ ] **Step 3: Commit**

```bash
git add nlp_medical_qa/requirements.txt
git commit -m "fix(deps): remove bert-score and scipy — unused since sentence-transformers switch"
```

---

### Task 2: Remove `bert-score` from notebook pip install (cell 0-B)

**Files:**
- Modify: `nlp_medical_qa/notebooks/colab_runner.ipynb` (cell id: `cell-0b-install`)

The Colab install cell still installs `bert-score` even though the code no longer uses it. A fresh device installing it gets the broken `transformers>=4.40` conflict.

- [ ] **Step 1: Edit cell 0-B source in the notebook**

In `nlp_medical_qa/notebooks/colab_runner.ipynb`, find the source of cell `cell-0b-install`.

Replace:
```
    "rouge-score>=0.1.2\" \\\n",
    "    \"bert-score>=0.3.13\" \\\n",
    "    \"accelerate>=0.27\" \\\n",
```

With:
```
    "rouge-score>=0.1.2\" \\\n",
    "    \"accelerate>=0.27\" \\\n",
```

> **Tip:** The `.ipynb` file is JSON. Use the Edit tool with the exact strings above, or load the file, locate `cell-0b-install`, and remove the `bert-score` line from its `source` array.

- [ ] **Step 2: Verify**

Read `nlp_medical_qa/notebooks/colab_runner.ipynb`. Confirm cell `cell-0b-install` source contains `rouge-score` and `accelerate` adjacent to each other, with no `bert-score` line between them.

- [ ] **Step 3: Commit**

```bash
git add nlp_medical_qa/notebooks/colab_runner.ipynb
git commit -m "fix(notebook): remove bert-score from pip install cell (unused, breaks transformers>=4.40)"
```

---

### Task 3: Enable faithfulness in the full eval run (cell 4-B)

**Files:**
- Modify: `nlp_medical_qa/notebooks/colab_runner.ipynb` (cell id: `cell-4b-full`)

The 12-step framework (Step 6) requires the NLI faithfulness metric. Cell 4-B currently sets `SKIP_FAITH = True`, which makes all faithfulness values `None` in the output CSVs.

- [ ] **Step 1: Edit cell 4-B source in the notebook**

In cell `cell-4b-full`, replace:

```
SKIP_FAITH = True  # set False to include NLI faithfulness (slower)
```

With:

```
SKIP_FAITH = False  # True to skip NLI faithfulness and save ~40% time
```

- [ ] **Step 2: Verify**

Read the notebook. Confirm cell `cell-4b-full` source contains `SKIP_FAITH = False`.

- [ ] **Step 3: Commit**

```bash
git add nlp_medical_qa/notebooks/colab_runner.ipynb
git commit -m "fix(notebook): enable faithfulness scoring in full eval run (required by assignment)"
```

---

### Task 4: Extend Drive purge utility to delete checkpoints (cell X-3)

**Files:**
- Modify: `nlp_medical_qa/notebooks/colab_runner.ipynb` (cell id: `cell-x-purge`)

The existing cell X-3 deletes indexes and result CSVs but not V1/V3 checkpoint directories. A clean re-run also needs checkpoints deleted so Phase 3 (training) runs from scratch.

- [ ] **Step 1: Edit cell X-3 source in the notebook**

Replace the entire source of cell `cell-x-purge` with:

```python
# X-3  Purge stale indexes, checkpoints and results (run before re-building after a code update)
# Deletes BM25/FAISS indexes, V1/V3 checkpoints, and all result CSVs from Drive.
# Run Phase 2, Phase 3, and Phase 4 again afterwards.
import glob, shutil

confirm = input("Type YES to delete all indexes, checkpoints and result CSVs: ")
if confirm.strip().upper() == "YES":
    idx_files = (
        glob.glob(f"{DRIVE_BASE}/data/processed/bm25_*.pkl")
        + glob.glob(f"{DRIVE_BASE}/data/processed/dense_*.faiss")
        + glob.glob(f"{DRIVE_BASE}/data/processed/dense_*.pkl")
    )
    csv_files = glob.glob(f"{DRIVE_BASE}/results/*.csv")
    for p in idx_files + csv_files:
        os.remove(p)
        print(f"Deleted: {p}")

    ckpt_dir = f"{DRIVE_BASE}/data/processed/checkpoints"
    for variant in ["v1", "v3"]:
        ckpt_path = f"{ckpt_dir}/{variant}"
        if os.path.exists(ckpt_path):
            shutil.rmtree(ckpt_path)
            print(f"Deleted checkpoint: {ckpt_path}")

    print(f"Removed {len(idx_files)} index files, {len(csv_files)} result CSVs, and checkpoints.")
    print("Now re-run Phase 2 (build indexes), Phase 3 (train V1/V3), and Phase 4 (evaluate).")
else:
    print("Aborted — nothing deleted.")
```

- [ ] **Step 2: Verify**

Read the notebook. Confirm cell `cell-x-purge` source contains `shutil.rmtree` and references both `"v1"` and `"v3"` in the checkpoint loop.

- [ ] **Step 3: Commit**

```bash
git add nlp_medical_qa/notebooks/colab_runner.ipynb
git commit -m "fix(notebook): extend purge utility to also delete V1/V3 checkpoints"
```

---

### Task 5: Clear all notebook cell outputs

**Files:**
- Modify: `nlp_medical_qa/notebooks/colab_runner.ipynb`

The committed notebook contains outputs from prior partial runs. These bloat the diff, confuse readers, and may show stale checkpoint/index paths. Clear them all.

- [ ] **Step 1: Clear outputs via Python**

Run from the `nlp_medical_qa/` directory:

```bash
python -c "
import json
path = 'notebooks/colab_runner.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)
for cell in nb.get('cells', []):
    cell['outputs'] = []
    cell['execution_count'] = None
with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write('\n')
print('Outputs cleared.')
"
```

Expected output:
```
Outputs cleared.
```

- [ ] **Step 2: Verify**

```bash
python -c "
import json
with open('notebooks/colab_runner.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
has_output = any(cell.get('outputs') for cell in nb.get('cells', []))
print('Has outputs:', has_output)
"
```

Expected output:
```
Has outputs: False
```

- [ ] **Step 3: Commit**

```bash
git add nlp_medical_qa/notebooks/colab_runner.ipynb
git commit -m "chore(notebook): clear stale cell outputs before re-run"
```

---

## Drive Cleanup Instructions (you do this manually)

Before the next Colab session, delete these from [drive.google.com](https://drive.google.com):

```
MyDrive/nlp_medical_qa_data/data/processed/    ← right-click → Remove to Trash
MyDrive/nlp_medical_qa_data/results/           ← right-click → Remove to Trash
```

**Keep untouched:**
```
MyDrive/nlp_medical_qa_data/data/raw/BioASQ-training13b/training13b.json   ← 45 MB, keep!
```

Alternatively: in Colab, run cell X-3 (after the code changes above are pulled) and type `YES`.

---

## Colab Execution Order (after Drive cleanup)

| Cell | Action | Est. time |
|---|---|---|
| 0-A → 0-D | Setup, mount, install (no bert-score), git pull | ~5 min |
| 1-B | Verify BioASQ JSON | <1 min |
| 2 | Build indexes (fresh) | ~10 min |
| 3-A | Train V1 from scratch | ~20 min |
| 3-B | Fine-tune V3 | ~30 min |
| 4-B | Full eval, 24 combos, faithfulness ON | ~100 min |
| 5-A → 5-D | Results table + failure analysis + plots | ~5 min |

**Total: ~2.8 hours.** All cells are idempotent — safe to re-run if the session resets.
