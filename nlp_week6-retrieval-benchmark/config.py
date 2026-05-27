from pathlib import Path

ROOT = Path(__file__).parent
CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results"

SEED = 42
MSMARCO_SAMPLE_SIZE = 300
HYDE_SAMPLE_SIZE = 10

RRF_K = 60
FIRST_STAGE_K = 100
LATENCY_WARMUP = 3
LATENCY_SAMPLES = 20

MODELS = {
    "dense_general": "sentence-transformers/all-MiniLM-L6-v2",
    "dense_domain": "sentence-transformers/msmarco-distilbert-base-v3",
    "cross_encoder": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "colbert": "colbert-ir/colbertv2.0",
}

HYDE_CONFIG = {
    "base_url": "https://ki-chat.uni-mainz.de/api",
    "model": "GPT OSS 120B",
    "max_tokens": 150,
    "sleep": 1.1,
}
