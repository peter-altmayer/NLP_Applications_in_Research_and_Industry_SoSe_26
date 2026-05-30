import json
from pathlib import Path
from tabulate import tabulate


def save_cache(path, data):
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_cache(path):
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def print_table(rows, headers):
    print(tabulate(rows, headers=headers, tablefmt="github"))
