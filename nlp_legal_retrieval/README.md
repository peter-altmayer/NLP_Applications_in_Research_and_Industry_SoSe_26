# NLP Pipeline Tasks: OCR, Translation, Summarization

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
python part2_translation.py   # Translation: WMT19 EN->DE with MarianMT
python part3_summarization.py # Summarization: CNN/DailyMail with BART-large
```

---

## Part 1: OCR Results

**Model:** `microsoft/trocr-base-printed`

**Dataset:** `naver-clova-ix/cord-v2` (100 test examples)

**Approach:** CORD-v2 provides per-word bounding box coordinates in its `valid_line` annotations. Each receipt image is cropped into individual text-line regions before being passed to TrOCR, which is designed for single-line input. Line OCR outputs are concatenated for document-level evaluation.

**Metrics:** CER (lower = better), WER (lower = better), F1 token overlap (higher = better)

| Metric | Mean Score |
|--------|-----------|
| CER | 0.2806 |
| WER | 0.4614 |
| F1  | 0.8131 |

*Note: TrOCR-base-printed defaults to all-caps output; CER/WER are case-sensitive and therefore overestimate errors from capitalisation differences.*

### Sample Results (first 5 examples)

| # | CER | WER | F1 | OCR (40 chars) | Gold (40 chars) |
|---|-----|-----|----|----------------|-----------------|
| 0 | 0.1765 | 0.3333 | 0.9231 | 901016 -TICKET CP 2 60.000 60.000 TOTA | 901016 -TICKET CP 2 60.000 60.000 TOTA |
| 1 | 0.0000 | 0.0000 | 1.0000 | J.STB PROMO 17500 Y.B.BAT 46000 Y.BASO | J.STB PROMO 17500 Y.B.BAT 46000 Y.BASO |
| 2 | 0.2891 | 0.6296 | 0.5000 | 1 JASMINE MT (L) 24.000 DOCONUT JELLY ( | 1 JASMINE MT ( L ) 24,000 COCONUT JELLY |
| 3 | 0.1392 | 0.3333 | 0.8571 | 1X @11000 DONAT GULA 11,000 1.00XITEMS | 1X @11000 DONAT GULA 11,000 1.00xITEMs |
| 4 | 0.1224 | 0.4348 | 0.6809 | ICE BLACKCOFFE 2 82,000 AVOCADO COFFEE | ICE BLACKCOFFE 2 82,000 AVOCADO COFFEE |

### Failure Analysis

**Example #30 (CER=0.82, WER=1.13, F1=0.56):**
- Gold:  `Kupon 3 28,636 Subtotal 28,636 PB1 (10%) 2,864 Dine In Total 31,500 Cash 50,000 Change 18,500`
- OCR:   `KUPON 3 28,636 SUBTOTAL @6% P81 (10%) (RM) DINE IN TOTAL : 1/03 CASH REGULAR, COME AGAIN 5,000 CHANGE *** ********`
- **Gold vs OCR:** PB1 is read as P81 (B/8 confusion); all four amounts (Subtotal, Total, Cash, Change) are replaced with garbage tokens (`1/03`, `***`, `REGULAR, COME AGAIN`).
- **Error type:** Both character-level (B->8 substitution in `PB1`) and word-level (entire numeric fields replaced or hallucinated).
- **Cause:** Table layout. The totals section of this receipt prints amounts in a right-aligned column. When cropped as individual lines, the amount field sits at the far edge of the crop and may be partially cut off or low-contrast, causing the model to hallucinate plausible-sounding but wrong tokens.
- **Best metric:** WER (1.13 - exceeds 1.0 because the model outputs more words than the gold), which captures the word-level substitutions. CER (0.82) also catches the character confusions. F1 (0.56) is less sensitive because it only measures set overlap.

**Example #80 (CER=0.72, WER=0.93, F1=0.64):**
- Gold:  `1 Viet Milk Coffee 25,000 +Ice +M Subtotal 25.000 Total 25.000 CASH 25.000 Kembalian 0`
- OCR:   `1 VIFT MILK COFFEE 25.000 +ICE +M SUBTOTAL : 1 SUBTOTAL !!!: CASH CHANGE 25.000 KEMBALIAN PERMAS NO`
- **Gold vs OCR:** "Viet" is read as "VIFT" (t/ft confusion); "Subtotal" appears twice with hallucinated punctuation (`!!!:`); "Total" is dropped; "Kembalian 0" becomes "KEMBALIAN PERMAS NO".
- **Error type:** Both character-level (VIFT/Viet - condensed font collapses `e` into `f`) and word-level (repeated token, hallucinated "PERMAS NO").
- **Cause:** Unusual font and mixed-language content. The receipt mixes English labels with Indonesian amounts ("Kembalian" = change in Indonesian). The model has poor coverage of Indonesian tokens and fills low-confidence regions with repeated or invented tokens.
- **Best metric:** CER (0.72) captures the character-level substitution in "VIFT" that WER (0.93) would count only as a single wrong word. F1 (0.64) partially reflects the overlap of correct tokens that were read right.

**Example #71 (CER=0.65, WER=1.0, F1=0.75):**
- Gold:  `Cuka Apel Tetes 1 198000 198000 Sub Total 198000 Tunai 200200 Kembalian 22000`
- OCR:   `CULKA APEL TEFES 1 198000 198000 SUB TOTAL !!! !!! !!! !!! !!! TUNAI : 200200 KEMBALIAN 22000`
- **Gold vs OCR:** "Cuka" -> "CULKA" (inserted L), "Tetes" -> "TEFES" (E->EF, s->S); the subtotal amount `198000` is replaced with five `!!!` tokens.
- **Error type:** Character-level (l/k insertion in CULKA, E->EF in TEFES) and word-level (`198000` replaced by five punctuation tokens).
- **Cause:** Similar characters and low-contrast print. The character insertions (CULK vs Cuk) are classic OCR confusion in condensed receipt fonts where ink bleed merges strokes. The `!!!` tokens suggest the subtotal line was printed with very low ink density or is obscured by a fold, so the model generates repetitive punctuation for the ambiguous region.
- **Best metric:** CER (0.65) best captures the character-level substitutions. WER = 1.0 because the five-token `!!! !!! !!! !!! !!!` replaces the single token `198000`, changing the word count. F1 (0.75) remains high because most content words are still correct.

---

## Part 2: Translation Results

**Model:** `Helsinki-NLP/opus-mt-en-de`

**Dataset:** `wmt/wmt19 de-en` validation (100 examples, EN->DE)

**Metrics:** BLEU (higher = better), chrF (higher = better), BERTScore F1 (higher = better)

| Metric | Score |
|--------|-------|
| BLEU | 41.24 |
| chrF | 67.86 |
| BERTScore F1 | 0.8942 |

### Sample Results (first 5 examples)

| # | chrF | Hypothesis (60 chars) | Reference (60 chars) |
|---|------|-----------------------|----------------------|
| 0 | 97.80 | Munchen 1856: Vier Karten, die Ihren Blick auf die Stadt ver | Munchen 1856: Vier Karten, die Ihren Blick auf die Stadt ver |
| 1 | 41.90 | Ein geistiges Asyl, wo sich heute junge Menschen treffen sol | Eine Irren-Anstalt, wo sich heute Jugendliche begegnen solle |
| 2 | 45.40 | Eine Kryptakapelle, wo sie nun Tunnel fur die S-Bahn graben. | Eine Gruftkapelle, wo nun fur den S-Bahn-Tunnel gegraben wir |
| 3 | 27.61 | Die Zuteilungsinhaber kultivieren den Boden ehemaliger Bauer | Kleingartner bewirtschaften den einstigen Grund von Bauern.  |
| 4 | 62.43 | Die alteste offizielle Karte von Munchen bringt fesselnde Ge | Die alteste offizielle Karte Munchens fordert spannende Gesc |

### Failure Analysis

**Example #72 (chrF=21.07): Idiom mistranslation**
- Source:     `Only regional produce is served to the holiday-makers.`
- Hypothesis: `Nur regionale Produkte werden den Urlaubern serviert.`
- Reference:  `Auf den Tisch der Feriengaste kommt ausschliesslich Regionales.`
- **Why it failed:** The reference uses a German idiom ("comes to the table of the holiday guests") that shares zero surface tokens with the literal translation. MarianMT produces a grammatically correct, semantically equivalent sentence, but chrF collapses to 21 because the metric is purely surface-form. This example shows the fundamental tension between translation correctness and n-gram-based evaluation: a perfect paraphrase can score as low as a wrong translation.

**Example #3 (chrF=27.61): Cultural concept**
- Source:     `Allotment holders cultivate the soil of former farmers.`
- Hypothesis: `Die Zuteilungsinhaber kultivieren den Boden ehemaliger Bauern.`
- Reference:  `Kleingartner bewirtschaften den einstigen Grund von Bauern.`
- **Why it failed:** "Allotment holder" refers to a specific British/European cultural institution (a rented garden plot on public land). The standard German term is "Kleingartner" (a culturally embedded word with its own legal framework). MarianMT has no mapping for this concept and falls back to a literal compound "Zuteilungsinhaber" (allocation holder) that does not exist in German. This is a failure of cultural knowledge rather than linguistic ability.

**Example #59 (chrF=53.23): Hallucination / wrong word sense**
- Source:     `The advisory also points to a recent report by the Missouri Attorney General's Office showing that black drivers in the state were 75 percent more likely to be pulled over than whites.`
- Hypothesis: `Die Beratung verweist auch auf einen kurz... dass schwarze Fahrer im Staat waren 75 Prozent wahrscheinlicher, um uberfahren zu werden als Weisse.`
- Reference:  `Der Rat weist auch auf einen kurzlich... schwarze Autofahrer im Staat 75 Prozent haufiger als Weisse angehalten werden.`
- **Why it failed:** "Pulled over" (a police traffic stop) is translated as "uberfahren" (to run over with a vehicle, i.e. to kill). The reference correctly uses "angehalten werden" (to be stopped). This is a word-sense hallucination: both "pull over" and "uberfahr-" involve vehicles but mean opposite things. MarianMT selected the wrong sense, producing a statement that black drivers were more likely to be physically run over rather than stopped by police - a factually dangerous mistranslation.

---

## Part 3: Summarization Results

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

**Omission hallucination: Example #41 (ROUGE-1=0.10)**

- **Type:** Omission hallucination - the model summarises background framing and omits the specific events that give the article its news value.
- **Article:** Indiana's Religious Freedom Restoration Act controversy. The article covers three specific real-world legal cases where religious freedom clashed with anti-discrimination law: a Native American's eagle feathers confiscated by the federal government, a Sikh IRS employee fired for carrying a ceremonial knife, and Chicago pastors denied church permits.
- **Reference highlights:**
  > A Native American from a tribe not recognized by the feds wins the return of his eagle feathers. An IRS accountant is fired for insisting on carrying a symbolic Sikh knife to work. A group of Chicago pastors takes on City Hall over its permits for new churches and loses.
- **BART summary:**
  > The U.S. Religious Freedom Restoration Act became law in 1993. So far, 20 states have some version of the religious liberty law, and the legal controversies have grown.
- **What is wrong:** BART produces a factually plausible but entirely generic overview of the RFRA. None of the three concrete cases are mentioned. The model latched onto the introductory framing of the article and produced a summary of context rather than content. This is omission hallucination: key source material exists in the article but is absent from the summary.

**Intrinsic hallucination: Example #61 (ROUGE-1=0.13)**

- **Type:** Intrinsic hallucination - the model selects a commentator's framing that contradicts the editorial emphasis of the reference, effectively inverting the article's conclusion.
- **Article:** The U.S. Department of Justice filed a 15-count indictment against FedEx for facilitating illegal drug shipments. Two legal analysts are quoted: Danny Cevallos argues FedEx has a strong legal defence (it cannot be expected to inspect every package); David Perry argues that corporations are easier to prosecute than individuals.
- **Reference highlights:**
  > Justice Department prosecuting FedEx over unauthorized shipment of drugs. Danny Cevallos: FedEx has a strong argument that it shouldn't be held responsible.
- **BART summary:**
  > A 15-count indictment filed in federal court in California names FedEx as a defendant. David Perry: Corporations can indeed be prosecuted like a person. In many ways they are easier to prosecute than people, Perry says.
- **What is wrong:** Both commentators appear in the article, so no content is invented. However, BART selects David Perry's pro-prosecution view as the analytical takeaway, while the reference editors chose Cevallos' defence argument as the more newsworthy angle. The summary therefore implies FedEx is likely to lose, while the reference implies the opposite. This is intrinsic hallucination: the model does not fabricate facts, but it misrepresents the article's emphasis in a way that changes the reader's conclusion.
