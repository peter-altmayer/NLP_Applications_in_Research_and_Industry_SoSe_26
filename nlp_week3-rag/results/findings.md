# Findings — Model × Retriever × K

## Full results (mean EM / mean F1)

| model | dataset | retrieval | k | EM | F1 | N |
|---|---|---|---|---|---|---|
| phi3 | natural_questions | bm25 | 1 | 0.000 | 0.071 | 1000 |
| phi3 | natural_questions | bm25 | 5 | 0.000 | 0.097 | 1000 |
| phi3 | natural_questions | bm25 | 10 | 0.000 | 0.198 | 1000 |
| phi3 | natural_questions | dpr_nq | 1 | 0.000 | 0.044 | 1000 |
| phi3 | natural_questions | dpr_nq | 5 | 0.000 | 0.081 | 1000 |
| phi3 | natural_questions | none | 0 | 0.000 | 0.058 | 1000 |
| phi3 | trivia_qa | bm25 | 1 | 0.147 | 0.176 | 1000 |
| phi3 | trivia_qa | bm25 | 5 | 0.100 | 0.118 | 1000 |
| phi3 | trivia_qa | bm25 | 10 | 0.092 | 0.109 | 1000 |
| phi3 | trivia_qa | dpr_multiset | 1 | 0.083 | 0.110 | 1000 |
| phi3 | trivia_qa | dpr_multiset | 5 | 0.054 | 0.073 | 1000 |
| phi3 | trivia_qa | dpr_multiset | 10 | 0.065 | 0.085 | 1000 |
| phi3 | trivia_qa | dpr_nq | 1 | 0.100 | 0.129 | 1000 |
| phi3 | trivia_qa | dpr_nq | 5 | 0.058 | 0.073 | 1000 |
| phi3 | trivia_qa | dpr_nq | 10 | 0.051 | 0.066 | 1000 |
| phi3 | trivia_qa | none | 0 | 0.338 | 0.408 | 1000 |
| qwen | natural_questions | none | 0 | 0.000 | 0.033 | 1000 |
| qwen | trivia_qa | bm25 | 1 | 0.044 | 0.055 | 1000 |
| qwen | trivia_qa | bm25 | 5 | 0.046 | 0.058 | 1000 |
| qwen | trivia_qa | bm25 | 10 | 0.043 | 0.060 | 1000 |
| qwen | trivia_qa | dpr_multiset | 1 | 0.015 | 0.026 | 1000 |
| qwen | trivia_qa | dpr_nq | 1 | 0.008 | 0.019 | 1000 |
| qwen | trivia_qa | dpr_nq | 5 | 0.023 | 0.034 | 1000 |
| qwen | trivia_qa | dpr_nq | 10 | 0.022 | 0.035 | 1000 |
| qwen | trivia_qa | none | 0 | 0.326 | 0.391 | 1000 |
