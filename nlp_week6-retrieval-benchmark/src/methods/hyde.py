import hashlib
import json
import os
import time
from pathlib import Path

from src.methods.base import Retriever

_SYSTEM = "You write short, factual-sounding passages that directly answer questions."
_USER_TMPL = (
    "Write a short passage (3-5 sentences) that directly answers this question. "
    "Do not preface or explain.\nQuestion: {query}\nPassage:"
)


class HyDERetriever(Retriever):
    def __init__(
        self,
        first_stage: Retriever,
        dataset_name: str,
        cache_dir: Path,
        model: str = "GPT OSS 120B",
        max_tokens: int = 150,
        sleep: float = 1.1,
    ):
        self.first_stage = first_stage
        self.dataset_name = dataset_name
        self.cache_dir = Path(cache_dir)
        self.model = model
        self.max_tokens = max_tokens
        self.sleep = sleep
        self._cache_path = self.cache_dir / "hyde" / f"{dataset_name}__hypotheticals.json"
        self._hypotheticals: dict[str, dict] = {}
        self._client = None

    def _get_client(self):
        from openai import OpenAI

        api_key = os.environ.get("JGU_API_KEY")
        if not api_key:
            raise RuntimeError(
                "JGU_API_KEY environment variable is not set. "
                "Export it before running HyDE: export JGU_API_KEY=<your_key>"
            )
        return OpenAI(base_url="https://ki-chat.uni-mainz.de/api", api_key=api_key)

    def _load_cache(self) -> None:
        if self._cache_path.exists():
            with open(self._cache_path, encoding="utf-8") as f:
                self._hypotheticals = json.load(f)

    def _save_cache(self) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._cache_path, "w", encoding="utf-8") as f:
            json.dump(self._hypotheticals, f, indent=2, ensure_ascii=False)

    def index(self, corpus: dict[str, str]) -> None:
        self._load_cache()
        # first_stage must already be indexed externally

    def _query_key(self, query: str) -> str:
        return hashlib.md5(query.encode()).hexdigest()

    def generate_hypothetical(self, query: str) -> str:
        key = self._query_key(query)
        if key in self._hypotheticals:
            return self._hypotheticals[key]["hypothetical"]

        client = self._get_client()
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": _USER_TMPL.format(query=query)},
            ],
            max_tokens=self.max_tokens,
            reasoning_effort="low",
        )
        time.sleep(self.sleep)
        hyp = resp.choices[0].message.content.strip()

        self._hypotheticals[key] = {
            "query": query,
            "hypothetical": hyp,
            "model": self.model,
            "reasoning_effort": "low",
            "prompt_system": _SYSTEM,
            "prompt_user": _USER_TMPL.format(query=query),
        }
        self._save_cache()
        return hyp

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        hypothetical = self.generate_hypothetical(query)
        return self.first_stage.search(hypothetical, k)
