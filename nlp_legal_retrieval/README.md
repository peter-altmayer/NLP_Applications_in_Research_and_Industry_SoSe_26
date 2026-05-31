# NLP Pipeline Tasks — OCR, Translation, Summarization

Three NLP evaluation pipelines on real HuggingFace datasets.

## Setup

Install PyTorch with CUDA first:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Then install remaining dependencies:
```bash
pip install -r requirements.txt
pip install sentencepiece
```

## Running

Each script is self-contained. Results are cached to `results/` after the first run.

```bash
python part1_ocr.py           # OCR: CORD-v2 receipts with TrOCR
python part2_translation.py   # Translation: WMT19 EN→DE with MarianMT
python part3_summarization.py # Summarization: CNN/DailyMail with BART-large
```

---

## Part 1 — OCR Results

**Model:** `microsoft/trocr-base-printed`
**Dataset:** `naver-clova-ix/cord-v2` (100 test examples)
**Approach:** CORD-v2 provides per-word bounding box coordinates in its `valid_line` annotations. Each receipt image is cropped into individual text-line regions before being passed to TrOCR, which is designed for single-line input. Line OCR outputs are concatenated for document-level evaluation.
**Metrics:** CER (lower = better), WER (lower = better), F1 token overlap (higher = better)

| Metric | Mean Score |
|--------|-----------|
| CER | 0.2806 |
| WER | 0.4614 |
| F1  | 0.8131 |

*Note: TrOCR-base-printed defaults to all-caps output; CER/WER are case-sensitive and therefore overestimate errors from capitalisation differences — F1 (case-insensitive) is the more informative metric here.*

### Sample Results (first 5 examples)

| # | CER | WER | F1 | OCR (40 chars) | Gold (40 chars) |
|---|-----|-----|----|----------------|-----------------|
| 0 | 0.1765 | 0.3333 | 0.9231 | 901016 -TICKET CP 2 60.000 60.000 TOTA | 901016 -TICKET CP 2 60.000 60.000 TOTA |
| 1 | 0.0000 | 0.0000 | 1.0000 | J.STB PROMO 17500 Y.B.BAT 46000 Y.BASO | J.STB PROMO 17500 Y.B.BAT 46000 Y.BASO |
| 2 | 0.2891 | 0.6296 | 0.5000 | 1 JASMINE MT (L) 24.000 DOCONUT JELLY ( | 1 JASMINE MT ( L ) 24,000 COCONUT JELLY |
| 3 | 0.1392 | 0.3333 | 0.8571 | 1X @11000 DONAT GULA 11,000 1.00XITEMS | 1X @11000 DONAT GULA 11,000 1.00xITEMs |
| 4 | 0.1224 | 0.4348 | 0.6809 | ICE BLACKCOFFE 2 82,000 AVOCADO COFFEE | ICE BLACKCOFFE 2 82,000 AVOCADO COFFEE |

### Failure Analysis

**Example #30 (CER=0.82):**
- Gold: `Kupon 3 28,636 Subtotal 28,636 PB1 (10%) 2,864 Dine In Total 31,500 Cash 50,000 Change 18,500`
- OCR:  `KUPON 3 28,636 SUBTOTAL @6% P81 (10%) (RM) DINE IN TOTAL : 1/03 CASH REGULAR, COME AGAIN 5,000 CHANGE *** ********`
- The model hallucinates `@6%`, reads `P81` instead of `PB1`, and replaces all numeric totals with garbage (`1/03`, `***`). The subtotal, total, cash, and change amounts are entirely wrong.

**Example #80 (CER=0.72):**
- Gold: `1 Viet Milk Coffee 25,000 +Ice +M Subtotal 25.000 Total 25.000 CASH 25.000 Kembalian 0`
- OCR:  `1 VIFT MILK COFFEE 25.000 +ICE +M SUBTOTAL : 1 SUBTOTAL !!!: CASH CHANGE 25.000 KEMBALIAN PERMAS NO`
- The model duplicates `SUBTOTAL` with hallucinated punctuation (`!!!:`), invents `PERMAS NO` out of nothing, and drops `Total` and the Kembalian amount. Short text crops with mixed-language content (Indonesian `Kembalian`) confuse the model, which fills uncertain regions with repetitions or spurious tokens.

**Example #71 (CER=0.65):**
- Gold: `Cuka Apel Tetes 1 198000 198000 Sub Total 198000 Tunai 200200 Kembalian 22000`
- OCR:  `CULKA APEL TEFES 1 198000 198000 SUB TOTAL !!! !!! !!! !!! !!! TUNAI : 200200 KEMBALIAN 22000`
- The model correctly reads the item name (with character substitutions: C→C, u→U, k→K, a→A, Tetes→TEFES) and the amounts, but replaces the subtotal amount `198000` with five repeated `!!!` tokens. This is a classic case of the model failing on a crop where the number region has low contrast or is partially obscured by a fold or stain.

---

## Part 2 — Translation Results

**Model:** `Helsinki-NLP/opus-mt-en-de`
**Dataset:** `wmt/wmt19 de-en` validation (100 examples, EN→DE)
**Metrics:** BLEU (higher = better), chrF (higher = better), BERTScore F1 (higher = better)

| Metric | Score |
|--------|-------|
| BLEU | 41.24 |
| chrF | 67.86 |
| BERTScore F1 | 0.8942 |

### Sample Results (first 5 examples)

| # | chrF | Hypothesis (60 chars) | Reference (60 chars) |
|---|------|-----------------------|----------------------|
| 0 | 97.80 | München 1856: Vier Karten, die Ihren Blick auf die Stadt ver | München 1856: Vier Karten, die Ihren Blick auf die Stadt ver |
| 1 | 41.90 | Ein geistiges Asyl, wo sich heute junge Menschen treffen sol | Eine Irren-Anstalt, wo sich heute Jugendliche begegnen solle |
| 2 | 45.40 | Eine Kryptakapelle, wo sie nun Tunnel für die S-Bahn graben. | Eine Gruftkapelle, wo nun für den S-Bahn-Tunnel gegraben wir |
| 3 | 27.61 | Die Zuteilungsinhaber kultivieren den Boden ehemaliger Bauer | Kleingärtner bewirtschaften den einstigen Grund von Bauern.  |
| 4 | 62.43 | Die älteste offizielle Karte von München bringt fesselnde Ge | Die älteste offizielle Karte Münchens fördert spannende Gesc |

### Failure Analysis

**Example #72 (chrF=21.07) — Idiomatic expression:**
- Source: `Only regional produce is served to the holiday-makers.`
- Hypothesis: `Nur regionale Produkte werden den Urlaubern serviert.`
- Reference: `Auf den Tisch der Feriengäste kommt ausschließlich Regionales.`
- MarianMT produces a grammatically correct, literal translation. The reference uses an idiomatic German construction ("comes to the table of the holiday guests") with completely different vocabulary. No shared tokens → chrF collapses. This reveals the gap between semantic equivalence and surface-form metrics.

**Example #3 (chrF=27.61) — Domain terminology:**
- Source: `Allotment holders cultivate the soil of former farmers.`
- Hypothesis: `Die Zuteilungsinhaber kultivieren den Boden ehemaliger Bauern.`
- Reference: `Kleingärtner bewirtschaften den einstigen Grund von Bauern.`
- The model translates "allotment holders" literally as *Zuteilungsinhaber* (a non-existent compound) instead of the standard German term *Kleingärtner*. Similarly, *kultivieren* vs *bewirtschaften* and *Boden* vs *Grund* are both correct alternatives, but the reference consistently uses more idiomatic choices. MarianMT lacks coverage of domain-specific terms that differ significantly from their literal readings.

**Example #74 (chrF=28.19) — Lexical paraphrase:**
- Source: `I can't make a profit on that kind of steep-sloping ground.`
- Hypothesis: `Auf so einem steilen Boden kann ich keinen Gewinn machen.`
- Reference: `Ich kann auf derart steilen Flächen nicht gewinnbringend wirtschaften.`
- The model renders "make a profit" as *Gewinn machen* (correct but informal) while the reference uses the compound adjective *gewinnbringend wirtschaften* (to operate profitably). Additionally *Boden* (floor/ground) vs *Flächen* (areas/surfaces) reflects a register difference. These are valid paraphrases semantically but score poorly on surface-level chrF.

---

## Part 3 — Summarization Results

**Model:** `facebook/bart-large-cnn`
**Dataset:** `abisee/cnn_dailymail 3.0.0` test (100 examples)
**Metrics:** ROUGE-1, ROUGE-2, ROUGE-L (all higher = better)

| Metric | Mean Score |
|--------|-----------|
| ROUGE-1 | 0.3689 |
| ROUGE-2 | 0.1643 |
| ROUGE-L | 0.2776 |

### Sample Results (first 5 examples)

| # | ROUGE-1 | ROUGE-2 | ROUGE-L | Summary (80 chars) |
|---|---------|---------|---------|---------------------|
| 0 | 0.5352 | 0.3768 | 0.4789 | The Palestinian Authority becomes the 123rd member of the International Criminal |
| 1 | 0.4742 | 0.2316 | 0.3918 | Theia, a one-year-old bully breed mix, was hit by a car and buried in a field. F |
| 2 | 0.4054 | 0.1944 | 0.2703 | Mohammad Javad Zarif is the Iranian foreign minister. He has been John Kerry's o |
| 3 | 0.4524 | 0.2439 | 0.3333 | The five were exposed to Ebola in Sierra Leone in March, but none developed the  |
| 4 | 0.4750 | 0.2564 | 0.3000 | Duke student admits to hanging a noose from a tree near a student union. The stu |

### Hallucination Analysis

**Omission hallucination — Example #41 (ROUGE-1=0.1013):**
- Article: Indiana's Religious Freedom Restoration Act controversy — the article discusses multiple specific legal cases where religious freedom conflicted with anti-discrimination law, including a Native American eagle feather case, a Sikh IRS employee, and Chicago church permits.
- Reference highlights: `A Native American from a tribe not recognized by the feds wins the return of his eagle feathers. An IRS accountant is fired for insisting on carrying a symbolic Sikh knife to work. A group of Chicago pastors takes on City Hall…`
- Summary: `The U.S. Religious Freedom Restoration Act became law in 1993. So far, 20 states have some version of the religious liberty law, and the legal controversies have grown.`
- BART produces a factually plausible but entirely generic summary about the RFRA, omitting all three concrete cases the reference highlights. The specific incidents that give the article its news value are completely absent — a classic omission hallucination where the model summarises the framing instead of the content.

**Intrinsic hallucination — Example #61 (ROUGE-1=0.1333):**
- Article: DOJ indicts FedEx for drug shipments. Two legal commentators are quoted — Danny Cevallos argues FedEx has a strong defence; David Perry argues corporations are easier to prosecute.
- Reference highlights: `Justice Department prosecuting FedEx over unauthorized shipment of drugs. Danny Cevallos: FedEx has a strong argument that it shouldn't be held responsible.`
- Summary: `A 15-count indictment filed in federal court in California names FedEx as a defendant. David Perry: Corporations can indeed be prosecuted like a person. In many ways they are easier to prosecute than people, Perry says.`
- The summary selects David Perry's pro-prosecution framing and presents it as the main analytical takeaway, while the reference highlights Cevallos' opposing view (FedEx's defence). Both commentators appear in the article, but BART's selection inverts the editorial emphasis of the reference highlights — an intrinsic hallucination by omission of the counter-argument the editors deemed most salient.
