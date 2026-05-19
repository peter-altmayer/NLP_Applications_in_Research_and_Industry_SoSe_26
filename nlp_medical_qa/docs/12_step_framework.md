# 12-Step Project Framework — Medical QA Pipeline

**Project:** NLP Medical QA — RAG Pipeline over Clinical Text
**Course:** NLP Applications in Research and Industry, Uni Mainz, SoSe 2026
**Author:** Peter Altmayer

---

## 1. Problem Definition

- **What exactly is the task? Is it well-defined?**

  - **The task:** given a natural-language question and a corpus of biomedical text,
    retrieve the relevant passages and produce a correct, source-grounded answer — i.e. a
    Retrieval-Augmented Generation (RAG) pipeline for Medical QA.
  - **QA is not search:** search returns documents; QA returns answers. A QA system must
    understand the question and synthesize a response, not just locate a page.
  - **Is it well-defined?** For this project, yes — the scope is fixed: benchmark
    datasets with gold answers, a defined set of metrics, and four model variations
    (V1–V4) to compare. As a general problem it is only partially well-defined, because
    "the correct answer" is often clinically ambiguous; we adopt the benchmarks' gold
    answers as our operational definition of correctness.

- **Who is the user? What do they actually need?**

  - Primary user: a clinician — doctor, physician, or nurse — who needs to retrieve
    information from biomedical literature quickly, without manually reading long
    documents.
  - They need answers that are (1) fast, (2) accurate, and (3) traceable — backed by a
    citable source span, so the answer can be verified rather than trusted blindly.
  - Secondary "user": the course evaluators — the pipeline must be reproducible and
    well-documented enough for a third party to run and assess.

- **What does success look like to them vs. to us?**

  - **To the user (clinician):** a correct, source-backed answer found in seconds
    instead of minutes — trusted because every answer points to its evidence. Success =
    speed + accuracy + verifiability.
  - **To us:** a working, reproducible RAG pipeline meeting the project's evaluation
    requirements — all four variations compared on the same datasets and metrics,
    retrieval and generation evaluated separately, a faithfulness check, failure
    analysis, and a clear results table. Success = a defensible, well-evaluated system
    that demonstrates understanding of the design tradeoffs.
  - **Tension to keep in mind:** these are not the same. A system can score well on
    Exact Match / F1 (our success) and still be clinically unhelpful or unsafe (their
    failure). Benchmark performance is a *proxy* for real-world usefulness, not a
    guarantee — a gap that recurs throughout the framework.

---

## 2. Prior Work & Existing Solutions

- **Has this been done before? What did they do?**

  - Yes — Medical QA and RAG over biomedical text are well-established.
  - The standard recipe is the one we follow: retrieve relevant passages from a trusted
    corpus, then have a language model generate an answer grounded in those passages.
    Approaches range from extractive QA with domain encoders to RAG pipelines on open
    LLMs, evaluated on benchmarks like PubMedQA and MedMCQA.

- **Are there existing models, datasets, or pipelines we can reuse?**

  - Reusable components exist at every layer: domain encoders for extractive QA, open
    generative LLMs for generation, and pretrained sentence-embedding models for
    retrieval. The model-family choices and their tradeoffs are detailed under
    Architecture; the datasets under Data.
  - **The four canonical experimental variations to position against:**
    - **V1 — Train from scratch:** randomly initialized weights, trained on biomedical
      QA data; establishes the baseline — what does pre-training actually buy you?
      > **→ Implemented as:** 2-layer BERT (`L=2, H=256, A=4, intermediate=1024`, ~13M params),
      > random weights, trained on SQuAD train (10 000 examples, 3 epochs, lr=3e-4).
      > Tokeniser: `bert-base-uncased` (vocabulary reuse only — encoder weights are random).
    - **V2 — Pretrained, no fine-tuning:** off-the-shelf pretrained model evaluated
      directly; tests parametric knowledge and shows the value of pre-training over V1.
      > **→ Implemented as:** `deepset/roberta-base-squad2` (RoBERTa-base, ~125M params),
      > pretrained on SQuAD 2.0, evaluated on all three datasets without any further training.
    - **V3 — Pretrained + fine-tuned:** pretrained model fine-tuned on biomedical QA
      data; compare to V2 to isolate the contribution of fine-tuning.
      > **→ Implemented as:** `bert-base-uncased` (~110M params) fine-tuned by us on BioASQ
      > yesno questions (1 459 examples) + PubMedQA labeled split (train portion), 4 epochs,
      > lr=2e-5. Training data formatted so the answer is always a verbatim span in the context.
    - **V4 — Already fine-tuned:** domain-specific model from HuggingFace evaluated
      directly; the ceiling — compare to V3: was your fine-tuning as good as the
      specialist?
      > **→ Implemented as:** `dmis-lab/biobert-base-cased-v1.1-squad` (~110M params) —
      > BioBERT v1.1 (pre-trained on PubMed + PMC abstracts) further fine-tuned on SQuAD v1.1
      > by the DMIS Lab. Evaluated on all three datasets without any additional training.
      > *(Originally specified as `qiaojin/PubMedBERT-base-uncased-abstract-squads`; that
      > identifier is not available on HuggingFace Hub, so BioBERT v1.1-SQuAD is used as the
      > nearest equivalent domain-specialist.)*

- **Are we in a research setting or an industry setting?**

  - Closer to the industry end — but neither, strictly. This is a course project: no
    novel contribution is required, since Medical QA systems have been built many times.
  - The goal is a solution that *verifiably* performs well — measured rigorously against
    benchmarks — rather than one that advances the state of the art.
  - The deliverable is therefore the **evaluation and comparison itself**: a sound,
    reproducible study of how V1–V4 trade off, not a new model.

- **What are the limitations of prior work we need to address?**

  - **Retrieval and generation errors get conflated.** A good end-to-end score can hide
    a weak retriever. We evaluate retrieval (Precision@k, Recall@k) separately from
    generation.
  - **Benchmarks are often too narrow.** Much prior work evaluates only on simple
    multiple-choice QA. We use datasets that test *different* question types (factoid,
    yes/no, multi-hop) plus a general RAG dataset.
  - **The "accuracy fallacy" — fluent but ungrounded answers.** A model can score well
    while its answer is not actually supported by the retrieved evidence. We add an
    explicit faithfulness/NLI-based metric.
  - **Hallucination remains unsolved.** RAG reduces but does not eliminate it. We treat
    grounded-generation prompting as a mitigation, not a fix.
  - **Reproducibility gaps.** Prior pipelines are often hard to rerun. We address this
    with a documented environment and exact setup steps.

---

## 3. Data

- **Does training/evaluation data exist?**

  - Yes — established biomedical QA benchmarks exist and are the right fit, since our
    four setups (V1–V4) all require labeled QA data on the *same* datasets to be
    comparable.
  - **Datasets we will actually use** (2 biomedical + 1 general RAG, per project spec):
    - **PubMedQA** — biomedical. Yes/no/maybe questions over PubMed abstracts; ~1k
      expert-labeled instances plus a large artificially-generated set (~211k). Each
      instance pairs a question with a context (the abstract) and a gold answer — so it
      supports retrieval + QA cleanly. The large artificial split makes V1 (train from
      scratch) feasible; HuggingFace hosts models already fine-tuned on it for V4.
      > **→ Implemented as:** HuggingFace `pubmed_qa` / config `pqa_labeled` (1 000 expert-labeled
      > instances). 80/20 train–test split (seed 42). Gold answer = `final_decision` (yes/no/maybe)
      > for EM; `long_answer` as reference for ROUGE/BERTScore. Evaluation: 200 samples.
    - **BioASQ (Task B)** — biomedical. Mixed question types (factoid, list, yes/no,
      summary), expert-annotated, but small (~3k training instances). Chosen
      deliberately as a *contrast* to PubMedQA: different question formats.
      > **→ Implemented as:** BioASQ Task B version 13 — local file
      > `data/raw/BioASQ-training13b/training13b.json` (5 389 questions total, registered
      > download from bioasq.org). Question types used: **factoid** and **yesno** only
      > (list and summary excluded — answer types incompatible with extractive span evaluation).
      > 80/20 train–test split (seed 42). Gold answer = `exact_answer` (normalised to string).
      > Evaluation: 200 samples.
    - **SQuAD** (or Natural Questions) — general-domain RAG dataset. Extractive QA with
      question/context/answer-span structure; large, widely used, and abundant
      pretrained + fine-tuned checkpoints exist for V2/V4. Acts as the
      out-of-medical-domain reference point.
      > **→ Implemented as:** HuggingFace `squad` v1.1, validation split (randomly shuffled,
      > seed 42). Gold answer = first entry in `answers["text"]`. Evaluation: 200 samples.
      > SQuAD train split (10 000 examples) used for V1 fine-tuning.
  - _Selection rationale: every dataset must (a) have gold answers, (b) be large enough
    or paired with a HF checkpoint so all of V1–V4 can run, and (c) test something
    different from the others._

- **Can we use it? (licensing, copyright, consent, privacy)**

  - Yes — the chosen datasets are all publicly released for research and distributed via
    HuggingFace / official repositories.
  - PubMedQA and BioASQ are built from published research abstracts, and SQuAD from
    Wikipedia — public text, not patient records. This keeps the project clear of
    consent and protected-health-information issues.
  
- **What does it look like? (domain, language, format, noise)**

  - **Format:** all three follow a (question, context, gold answer) structure — clean,
    English, already tokenizable. PubMedQA answers are yes/no/maybe; BioASQ spans
    several answer types; SQuAD answers are extracted text spans.
  - **Domain & language:** PubMedQA/BioASQ are dense biomedical scientific English
    (technical, but edited and well-formed — *not* messy clinical shorthand). SQuAD is
    general-domain English.
  - **Noise profile:** much lower than raw clinical notes — these are curated
    benchmarks. But the underlying biomedical-language difficulties below still apply.
  - **Data challenges in the biomedical domain:**
    - Abbreviations are ambiguous: "SOB" = shortness of breath; "MS" = multiple sclerosis or morphine sulfate
    - Negation is critical: "no evidence of pneumonia" ≠ "pneumonia" — models must detect scope of negation
    - Implicit knowledge required: "troponin elevated" implies cardiac event — not stated, must be inferred
    - Specialized vocabulary: Latin roots, drug names, procedure codes — far from the web text models were pre-trained on
    - Class imbalance: rare diseases are underrepresented — models fail at the cases that matter most

- **How much do we have? Is it enough?**

  - **PubMedQA:** ~1k expert-labeled is small for evaluation but workable; the ~211k
    artificial split gives enough volume for V1 (train from scratch) and V3 (fine-tune).
    > **→ Actual split used:** 800 train / 200 test (seed 42, `pqa_labeled` only).
  - **BioASQ:** ~3k training instances — small. Enough for fine-tuning (V3), marginal
    for training from scratch (V1).
    > **→ Actual split used:** 4 311 train / 1 078 test (seed 42, factoid+yesno from 5 389
    > total; list+summary excluded). V3 uses only yesno type (1 459 examples) for fine-tuning
    > because factoid exact_answers are frequently not verbatim spans in the retrieved snippets.
  - **SQuAD:** large (~100k+ QA pairs) — comfortably enough for all setups.
    > **→ Actual split used:** validation split shuffled (seed 42), 200 test samples;
    > 10 000 training examples for V1 fine-tuning.
  - **Verdict:** sufficient for a sound comparative study.

- **If not — how do we get more?**

  - We do not need to collect new data — the benchmarks are adequate for the project's
    scope.

- **Is the data representative of our real users?**

  - **Reasonably so for the task as scoped.** Our task is QA over biomedical literature,
    and our datasets *are* biomedical literature (plus a general-domain set) — so the
    evaluation data matches the kind of text and questions the system is actually built
    to handle.
  - The honest caveat: this is representative of literature-based QA, not of a
    hypothetical clinical deployment over real EHR data, where the language is messier
    and the query distribution differs.

- **Are there biases in the data we need to worry about?**

  - Historical underrepresentation: clinical research has historically excluded women, minorities, and the elderly — their data is scarcer
  - Geographic bias: most publicly available data comes from Western, high-income healthcare systems — not representative globally
  - Label bias: diagnostic labels can reflect existing medical biases (e.g. pain undertreated in certain demographic groups)
  - Feedback loops: biased models → biased decisions → biased future data — the problem compounds over time
  - _Project-specific: PubMed-derived data inherits research-publication bias —
    well-studied diseases and populations are overrepresented. We probe at least one
    demographic axis in evaluation._

---

## 4. Constraints

- **How much compute do we have?**

  - Limited. Two options: a personal laptop or Google Colab — Colab is the better choice
    (free GPU access, no local hardware strain, reproducible environment).
  - Colab's free tier provides a single mid-range GPU (typically a T4, ~16 GB VRAM) with
    capped, interruptible sessions — adequate for fine-tuning small models, but not for
    large-scale training.
  - **Direct consequence:** compute is the binding constraint on *dataset sample size*.
    We do not train/evaluate on the full datasets — we draw fixed, reproducible subsets
    (with a set random seed) sized to fit Colab sessions, documented in the README.
    > **→ Fixed settings:** random seed = 42; evaluation subset = 200 samples per dataset;
    > V1 training = 10 000 SQuAD examples (~20 min on T4); V3 fine-tuning = ~2 000 biomedical
    > examples (~30 min on T4).

- **Should we use open or closed models?**

  - **Open models** (HuggingFace) — required, not just preferred: V1/V3 need weight
    access for training/fine-tuning; V4 explicitly calls for an off-the-shelf fine-tuned
    model from HuggingFace; and open models run inside Colab with no per-call cost.
  
- **How big a model can we actually use?**

  - Small. The ~16 GB Colab GPU caps us at compact encoder models (BERT-base /
    ClinicalBERT-scale, ~110M params) for extractive QA, and small generative models for
    the RAG generator.
  - **V1 (train from scratch) is the hardest hit** — a from-scratch transformer on
    limited data and compute will be weak. V1 therefore uses a *small* architecture, and
    we treat its low scores as an expected, informative baseline rather than a failure.
    > **→ V1 architecture:** `BertConfig(num_hidden_layers=2, hidden_size=256,
    > num_attention_heads=4, intermediate_size=1024)` + `BertForQuestionAnswering` QA head,
    > ~13M parameters total. Fits comfortably in 16 GB VRAM.

- **Are there latency requirements?**

  - No real-time requirement. This is an offline research/evaluation project — all runs
    are **batch**: process a dataset subset, compute metrics, write a results table.
    Latency would matter in a real clinical deployment but is out of scope here.

- **On-premise or cloud?**

  - Cloud (Colab) — acceptable *only because* our datasets are public, literature-based
    benchmarks, not patient records. No protected health information ever touches the
    cloud, so data-residency / HIPAA / GDPR concerns do not bind here. A real clinical
    system handling EHR data could not use free cloud compute this way.

- **How long do we have to build this?**

  - **One week.** A hard constraint that shapes scope heavily: favor small models and
    dataset subsets over full-scale runs; reuse existing HuggingFace models/tokenizers
    rather than building from scratch (except V1, which must be from-scratch); prioritize
    a complete, correct, reproducible V1–V4 comparison over peak scores on any one setup.

- **How many people are working on it?**

  - **One person — solo project.** Combined with the one-week limit: no parallel
    workstreams, so the plan is deliberately lean — a working end-to-end pipeline first,
    refinements (extra metrics, failure analysis, bias check) second.

---

## 5. Architecture & Techniques

- **What's the right approach? (prompting, fine-tuning, RAG, agents...)**

  - **Choosing between extractive and generative** is an architectural decision with
    downstream consequences for evaluation, reliability, and legal defensibility.
    - *Extractive is appropriate when:* you need verifiable, traceable answers;
      questions are factoid (who/what/when/where); vocabulary is consistent.
    - *Generative is appropriate when:* questions require synthesis across passages; you
      need fluent responses; occasional errors are acceptable given mitigations.
  - **Our choice: a RAG pipeline (hybrid).** Retrieve relevant passages, then generate a
    grounded answer. Retrieval supplies traceable source passages — addressing the
    medical need for verifiable answers — while the generator handles synthesis. This
    also matches the project requirement to evaluate retrieval and generation separately.
    Prompting (not agents) is the interaction method: agents add complexity and failure
    modes a one-week solo project does not need.

- **Do we need retrieval? A knowledge base? External tools?**

  - **Yes — retrieval is the core of the pipeline**, for two reasons: *knowledge* (the
    answer lives in the corpus, not reliably in the model's parameters) and
    *traceability* (retrieval gives a citable source span — a pure generative model
    cannot point to its evidence).
  - **Knowledge base:** the document corpus itself (PubMedQA abstracts, BioASQ snippets,
    SQuAD contexts), embedded and indexed for similarity search.
  - **External tools:** none.

- **Do we need to train or fine-tune, or can we use off-the-shelf?**

  - **Both — that is exactly what the V1–V4 comparison measures** (V1 from scratch, V2
    pretrained as-is, V3 fine-tuned by us, V4 already fine-tuned). The point is to
    isolate what each step contributes: pre-training (V1→V2), our fine-tuning (V2→V3),
    and a purpose-built specialist (V3→V4).
  - **The retrieval component is off-the-shelf and held constant throughout.** Only the
    QA/generation model varies across V1–V4; the retriever (a pretrained
    sentence-embedding model) is fixed — otherwise we could not tell whether a score
    change came from the retriever or the generator.
    > **→ Implemented as two fixed retrievers (both constant across V1–V4):**
    > - **Dense:** `sentence-transformers/all-MiniLM-L6-v2` (~22M params) → FAISS `IndexFlatIP`
    >   (inner-product with L2-normalised embeddings); index built once and saved to disk.
    > - **Sparse:** BM25 (Okapi BM25 via `rank-bm25`); comparison axis for retrieval ablation.
    > Both are evaluated side-by-side; BM25 vs. dense retrieval is the *retrieval experiment axis*
    > (see Step 6 — replaces the temperature/decoding experiment for extractive QA).

- **How do we handle context length limitations?**

  - A real constraint: encoder QA models have a hard ~512-token input limit, and small
    generative models have modest context windows. PubMed abstracts can exceed this.
  - **Chunking:** the corpus is split into passage-sized chunks before indexing, so each
    retrievable unit fits the model's limit. Chunk size is a tradeoff — too large
    exceeds the limit and dilutes relevance; too small splits an answer across chunks so
    the full answer never reaches the model.
  - **Top-k selection:** only the top-k retrieved chunks enter the context. Larger k
    improves recall but adds noise and eats context budget.
  - **For long single contexts** (extractive setup): a sliding window over the passage,
    taking the best-scoring span.
  - _All chunk-size and k settings are fixed and documented in the README._
    > **→ Fixed settings:** corpus unit = one BioASQ snippet / one SQuAD context paragraph /
    > one PubMedQA abstract section (no further splitting). Model input = top-k passages
    > joined, truncated to **max_length = 384 tokens** (`truncation="only_second"`).
    > **k = 5** passages retrieved per question (fixed across all experiments).

- **What are the tradeoffs of our architectural choices?**

  - **Model family:** extractive QA → encoder model (BERT, RoBERTa, ClinicalBERT);
    generative QA → decoder or encoder-decoder; RAG combines both — encoder for
    retrieval, decoder for generation.
  - Medical-specific tokenizers (or domain-adapted ones) can help — but aren't always
    available.
  - For medical QA: use **low decoding temperature** — consistent, conservative answers
    (this is also our decoding experiment).
    > **→ Extractive QA note:** V1–V4 are all BERT-family extractive models; they predict
    > start/end token positions and have no temperature parameter. The equivalent experiment
    > axis is **BM25 vs. dense retrieval**: both are run for every model×dataset combination,
    > allowing us to hold the QA model fixed and vary the retrieval strategy — the same
    > "second axis of improvement" role the temperature experiment plays for generative models.
  - **Tradeoffs specific to our choices:**
    - *RAG vs. pure generative:* RAG adds traceability and reduces hallucination, but
      adds a retriever — a second component that can fail independently and must be
      evaluated separately.
    - *Small models (compute-forced):* lower accuracy, but feasible to run and fine-tune
      in a week. V1 especially is expected to be weak.
    - *Constant retriever across V1–V4:* keeps the comparison clean, but means we are not
      optimizing retrieval.
    - *Extractive vs. generative within the study:* extractive gives exact, verifiable
      spans but cannot synthesize; generative synthesizes but can hallucinate. The
      project lets us observe this tradeoff empirically rather than just asserting it.

- **Prompt structure in RAG:**

  - System prompt: defines the model's role and constraints (e.g. "You are a clinical
    assistant. Answer only from the provided context. Do not use external knowledge.")
  - Retrieved context: the top-k chunks inserted as the "knowledge" the model should use.
  - User question: the original query.

---

## 6. Evaluation

- **What metrics do we use? (automatic vs. human)**

  - **Generation quality.** Exact Match (EM) — knowing it is likely too strict — used
    alongside **token-level F1** (bag-of-tokens overlap; partial credit), **BERTScore**
    (semantic similarity — credits a correct answer phrased differently, which EM
    unfairly penalizes), and **ROUGE** (overlap for longer-form answers).
    > **→ Implemented as:** SQuAD-style EM and token-F1 (lowercase, strip punctuation);
    > **ROUGE-L** F-measure (`rouge_score` library, stemmed); **BERTScore** F1 (`bert_score`
    > library, `bert-base-uncased` backbone, computed in batch on CPU).
  - **Retrieval quality — Precision@k & Recall@k**, reported separately from generation:
    - Precision@k = |{relevant} ∩ {top-k retrieved}| / k — of the top-k returned, what
      fraction are relevant? (retrieval quality)
    - Recall@k = |{relevant} ∩ {top-k retrieved}| / |{all relevant}| — of all relevant
      documents, what fraction appear in the top-k? (coverage)
    - These trade off: larger k improves recall but hurts precision (more noise). Low
      Recall@k means the relevant chunk never reaches the generator — the answer will be
      wrong regardless of how good the LLM is. A good end-to-end score can mask a bad
      retriever, so retrieval is always evaluated on its own.
    > **→ Implemented as:** exact-string match against gold snippets (BioASQ: provided in JSON;
    > PubMedQA: abstract sections; SQuAD: the gold context paragraph). k = 5 in all runs.
  - **Faithfulness:** one NLI-based metric — does the retrieved context actually entail
    the generated answer? Catches fluent-but-ungrounded answers (the "accuracy fallacy").
    > **→ Implemented as:** `cross-encoder/nli-MiniLM2-L6-H768` (~22M params) via
    > `transformers pipeline("text-classification")`. Input = `"{context[:400]} [SEP] {answer}"`;
    > output = P(entailment). Runs on CPU to preserve VRAM for QA models.
  - In medicine, a missed finding (FN) is often worse than a false alarm (FP) — metric
    choice should reflect that asymmetry.

- **Do the metrics actually capture what the user cares about?**

  - **Why evaluation is hard, especially in medicine:** there is often no single
    "correct" answer (clinical judgment); expert disagreement is common, so
    inter-annotator agreement is low; automated metrics measure proxies, not clinical
    correctness — a fluent wrong answer scores well; evaluation data may not reflect the
    real clinical distribution; false negatives and false positives carry very different
    stakes. This makes metric choice a design decision, not a technical detail.
  - **Honest answer: only partially.** Our metrics capture *answer overlap* and
    *groundedness* — useful proxies — but not *clinical correctness*. This is why we
    report a *spread* of metrics: EM (strict), F1 (partial credit), BERTScore
    (semantic), faithfulness (grounded), each covering a different blind spot. The gap
    to true clinical validity remains.

- **What's our baseline?**

  - **The baseline is internal — the V1–V4 design itself.** Each setup is the reference
    point for the next: V1 (from scratch) is the absolute floor; V2 (pretrained) is the
    main baseline for the RAG system; V3 (our fine-tuning) is measured against V2; V4
    (specialist) is the ceiling, measured against V3.
  - A trivial reference point also exists: the **majority-class baseline** (e.g. always
    answer "yes" on PubMedQA). Any setup that cannot beat it has learned nothing — a
    useful sanity check.
    > **→ Expected majority-class EM for PubMedQA:** PubMedQA `pqa_labeled` distribution is
    > roughly 55% "yes", 34% "no", 11% "maybe" — always predicting "yes" yields EM ≈ 0.55.
    > Any model below this floor on PubMedQA has learned nothing useful.

- **How do we know if we're improving?**

  - "Improvement" is not one number going up over time — it is the **ordered comparison
    across V1–V4 on the same datasets and metrics.** Expected ordering: V1 < V2 ≤ V3 ≲ V4.
    Each pairwise gap answers a specific question: V1→V2 = value of pre-training, V2→V3 =
    value of our fine-tuning, V3→V4 = gap to a specialist.
  - A result is only meaningful if the comparison is controlled: same dataset subsets,
    metrics, retriever, and random seed. An unexpected ordering (e.g. V3 < V2) is a
    finding to *investigate* via failure analysis, not a number to fix.
  - The decoding experiment (temperature / top-p) is a second axis of "improvement":
    holding the model fixed and seeing whether more conservative decoding raises
    faithfulness.
    > **→ Adapted to extractive QA:** BM25 vs. dense retrieval serves as the second
    > experiment axis — holding the QA model fixed (e.g. V2) and switching retriever.
    > Results are reported for both retrievers in every run; `merge_results.py` prints
    > the side-by-side table automatically.

- **How do we evaluate things that are hard to measure? (helpfulness, tone, safety)**

  - With no human raters, hard-to-measure qualities are approached through *proxies and
    targeted probes*: safety/hallucination → the NLI faithfulness metric plus the
    decoding experiment; helpfulness → partial-credit metrics (F1, BERTScore) plus the
    qualitative failure analysis.

- **Do we have a test set that reflects real usage?**

  - **No** Our test sets are held-out splits of PubMedQA,
    BioASQ, and SQuAD: published-literature QA, not real clinician queries over patient
    records. They *do* give us gold answers, reproducibility, and coverage of different
    question types — which is what makes the V1–V4 comparison informative. They *miss*
    real clinical language, query distribution, and EHR noise; a real deployment's
    performance would likely be lower.

---

## 7. Reliability & Safety

- **Can the model hallucinate? How badly does that matter here?**

  - **It depends on the setup:**
    - *Extractive* setups copy a span verbatim — they cannot fabricate content, only
      select the wrong span. Generative-style hallucination is near-impossible; the
      failure mode is wrong-but-real text.
    - *Generative* setups can hallucinate — fluent text unsupported by the retrieved
      context. This is the dangerous case our safety measures target.
  - Given the high stakes of the medical domain — errors can mean misdiagnosis, wrong
    treatment, patient harm — and the elevated hallucination risk on rare diseases,
    unusual drug interactions, and obscure details, this matters a great deal in
    principle. For the project itself the *consequences* are limited — offline
    evaluation, no real patients — but the project is a proxy for a setting where
    hallucination is a serious safety issue, so we measure it seriously.

- **How do we detect and mitigate hallucinations?**

  - **Mitigation:** grounded-generation prompting ("Answer only from the provided
    context. If the context does not contain the answer, say so explicitly") — the
    baseline that reduces but does not eliminate hallucination; low decoding temperature;
    and retrieval itself, which grounds answers in real passages rather than unreliable
    parametric memory.
  - **Detection:** the NLI-based faithfulness metric is the detector — an answer not
    entailed by its retrieved context is flagged as likely hallucinated. Retrieval
    metrics give an early warning (low Recall@k → ungrounded by construction), and the
    ≥3-case failure analysis categorizes *why* a hallucination occurred.
    > **→ Implemented as:** `cross-encoder/nli-MiniLM2-L6-H768` faithfulness score per
    > sample (see Step 6). For extractive models the predicted span is always real source
    > text, so the failure mode is *wrong-span selection* rather than fabrication —
    > faithfulness detects this as a low entailment score when the span is from an
    > irrelevant retrieved passage.
  - **PHI de-identification:** all context passages pass through a regex-based PHI masker
    (`src/privacy.py`) before being fed to any model — an illustrative safeguard, since
    our datasets contain no real patient data.
    > **→ Implemented patterns:** SSN, phone numbers, email addresses, dates (numeric and
    > written), patient/MRN IDs. Would be replaced by a clinical NER system (e.g. Presidio)
    > in a real deployment over EHR data.

- **Is the model robust to noisy or unexpected inputs?**

  - **Partially** Benchmark inputs are relatively clean, so
    robustness is not stress-tested by the datasets themselves. Known weak points:
    *retrieval noise* (irrelevant top-k chunks force the generator to answer from poor
    context); *out-of-distribution input* (a question whose answer is not in the corpus —
    intended behavior is an explicit "not found"); and *domain/vocabulary shift*
    (biomedical terms a small or off-the-shelf model never saw). We do not build a
    separate adversarial test, but the faithfulness metric indirectly measures robustness.

- **What happens when it fails? Gracefully?**

  - **Designed failure mode = abstention.** The system prompt instructs the model to say
    the answer is not in the context rather than fabricate one — a correct "I don't
    know" is a graceful failure; a confident wrong answer is not. In the extractive
    setup, failure is bounded: a wrong span is still real source text a user can read and
    reject. At evaluation time, failures are not silent — they show up as low EM/F1, low
    faithfulness, or retrieval misses, and feed the failure analysis.

- **Is there a human in the loop? Should there be?**

  - **In this project: no** — it is an automated, offline evaluation pipeline by design,
    so results stay reproducible and comparable.

---

## 8. Privacy & Legal

- **Does the system handle sensitive data?**

  - **This project, specifically: no.** We use only public, literature-based datasets
    (PubMedQA, BioASQ, SQuAD) — no patient records, no protected health information.

- **What regulations apply? (GDPR, HIPAA, etc.)**

  - *For the project:* effectively none bind directly — no personal or health data is
    processed.

- **Can we send data to an external API, or must it stay on-premise?**

  - Cloud (Colab) is fine here because the data is public. A real system handling
    patient data could not send it to an external API.

- **Who owns the outputs? Any IP concerns?**

  - Project code and results are our own work. Model weights carry their HuggingFace
    licenses — fine-tuned V3 inherits its base model's terms. Datasets are
    research-licensed; in extractive setups an output is verbatim source text, so that
    copyright stays with the original.

- **Are there terms of service issues?**

  - Licenses differ per model and dataset (research-only, attribution, commercial
    limits). Usage is research/non-commercial; no scraping, no redistribution.
  > **→ Exact licenses for all components used:**
  >
  > | Component | License |
  > |---|---|
  > | BioASQ 13b data | Research use only (bioasq.org; requires registration) |
  > | PubMedQA (`pqa_labeled`) | MIT |
  > | SQuAD v1.1 | CC BY-SA 4.0 |
  > | `deepset/roberta-base-squad2` (V2) | CC BY 4.0 |
  > | `bert-base-uncased` (V3 base) | Apache 2.0 |
  > | `dmis-lab/biobert-base-cased-v1.1-squad` (V4) | Apache 2.0 |
  > | `all-MiniLM-L6-v2` (retriever) | Apache 2.0 |
  > | `cross-encoder/nli-MiniLM2-L6-H768` (faithfulness) | Apache 2.0 |

---

## 9. Fairness & Bias

- **Does the system perform equally well across user groups?**

  - Models trained on biased data replicate and amplify it; in medicine a performance
    gap means **differential quality of care** — not an acceptable tradeoff. The
    underlying data biases are catalogued under Data.

- **Are there populations that could be harmed?**

  - The at-risk groups are the historically under-represented and under-published ones.
    Any model trained or fine-tuned on PubMed-derived data (V1, V3) inherits
    research-publication bias; V2/V4 inherit their original pre-training's biases.

- **How do we test for this?**

  - **For this project: We don't.**

---

## 10. Interpretability

- **Do we need to explain the model's decisions?**

  - Yes — for clinicians (trust and verify before acting) and end users (why this answer). An unexplainable answer in medicine is, in
    practice, an unusable one.

- **What level of interpretability is required?**

  - Not mechanistic interpretability — the required level is **source-level
    traceability**: every answer must be linkable to the passage it came from.

- **How do we provide it?**

  - The retrieval component already does it: the pipeline returns the top-k retrieved
    chunks alongside every answer.
---

## 11. Deployment

- **Serving, monitoring, updating** *(out of scope to build)*

- **How do we version and document it?** *(in scope — a graded reproducibility
  requirement)*

  - The bar: a stranger clones the repo, follows the README, and reproduces the results
    in under 15 minutes. No "it works on my machine."
  - Everything must be reproducible: data download, environment setup, model download,
    running the code.
  - **The README must record:** the exact dataset subsets and random seed; the chosen
    models and *why*; all model/dataset licenses; and the chunk-size / top-k /
    temperature settings — so the full V1–V4 comparison is reproducible by a third party.

---

## 12. Impact

- **What's the potential positive impact?**

  - *For clinicians:* faster access to answers buried in long documents — less reading,
    more time on care.
  - *For care quality:* traceable, source-grounded answers reduce reliance on memory and
    surface relevant evidence that might otherwise be missed.
  - *As a method study:* the V1–V4 comparison gives a clear, reproducible picture of what
    pre-training and fine-tuning actually buy in biomedical QA. The project's direct
    impact is educational/methodological; the clinical benefits describe what a matured
    version could offer.

- **What could go wrong at scale?**

  - The core risks — patient harm from errors, eroded patient trust, and differential
    quality of care across groups — are established earlier in the framework.

- **Who benefits and who might be harmed?**

  - *Benefits:* clinicians (time saved), patients indirectly (faster, evidence-backed
    answers), and healthcare systems (efficiency).
  - *Could be harmed:* patients from under-represented or under-published groups, who
    face the largest performance gaps.