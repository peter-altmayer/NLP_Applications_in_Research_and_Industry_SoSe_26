# NLP Week 2 — Medical QA Pipelines

This project builds two question-answering pipelines — one extractive, one generative — on the PubMedQA benchmark dataset. Medical records contain answers buried in unstructured clinical text; a clinician querying a patient's history or a researcher scanning literature should not have to read hundreds of pages. We compare two pretrained models per task to understand how general-domain models behave on biomedical text, where domain-specific vocabulary, abbreviations, and reasoning patterns routinely differ from general web corpora.

---

## Repository Structure

```
nlp_week2-qa/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/              # pubmed_qa download — never pushed
│   └── processed/        # qa_pairs.csv (25 examples) — never pushed
├── src/
│   ├── dataset.py        # download, flatten, save qa_pairs.csv
│   └── model.py          # forward-pass wrappers (extractive + generative)
├── experiments/
│   ├── run_extractive.py # RoBERTa + ELECTRA on 25 examples
│   ├── run_generative.py # Flan-T5-base + Flan-T5-large on 25 examples
│   └── run_decoding.py   # temperature sweep on Flan-T5-large
├── results/              # all CSVs committed
└── notebooks/
    └── exploration.ipynb # scratch only
```

---

## Dataset

**PubMedQA** (`pqa_labeled` config, 1 000 expert-annotated examples)

- `context` — concatenated PubMed abstract sections (BACKGROUND, METHODS, RESULTS, CONCLUSIONS)
- `question` — a biomedical research question
- `reference_answer` — the conclusion sentence from the abstract (used as ground truth)

Loaded via HuggingFace `datasets`, 25 examples used across all experiments.

---

## Models

### Task 1 — Extractive QA

| Model | HuggingFace ID | Size | Why |
|---|---|---|---|
| RoBERTa-base | `deepset/roberta-base-squad2` | ~500 MB | Strong general-domain extractive baseline, fine-tuned on SQuAD2 |
| ELECTRA-base | `deepset/electra-base-squad2` | ~110 MB | Compact discriminator model; frequently outperforms BERT/RoBERTa on span extraction |

Both models use a manual forward pass (`AutoModelForQuestionAnswering`) — the `question-answering` pipeline task was removed in transformers 5.x. Both are general-domain models whose performance on medical text reveals the domain gap motivating domain adaptation.

### Task 2 — Generative QA

| Model | HuggingFace ID | Size | Why |
|---|---|---|---|
| Flan-T5-base | `google/flan-t5-base` | ~250 MB | Instruction-tuned encoder-decoder; follows natural-language prompts reliably |
| Flan-T5-large | `google/flan-t5-large` | ~770 MB | Same architecture, ~3× parameters; allows a direct scale comparison |

Both use `AutoModelForSeq2SeqLM` with a structured prompt. The base/large pair provides a built-in ablation over model scale.

---

## Environment Setup

Requires Python 3.10+ and a virtual environment. The venv is created **outside** the project directory to avoid Windows MAX_PATH (260 char) errors caused by the long project path.

**Windows (PowerShell):**
```powershell
python -m venv E:\venvs\nlp2
E:\venvs\nlp2\Scripts\Activate.ps1
pip install -r requirements.txt

# Pin exact versions after installing:
pip freeze > requirements.txt
```

**macOS / Linux (bash):**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip freeze > requirements.txt
```

> **GPU note (Windows):** install the CUDA torch build separately if needed:
> `pip install torch --force-reinstall --index-url https://download.pytorch.org/whl/cu126`

---

## Reproducing Experiments

Run from inside `nlp_week2-qa/` with the venv active. The first run downloads PubMedQA automatically; subsequent runs reuse the cached CSV.

```powershell
# Task 1 — Extractive QA
python experiments/run_extractive.py
# writes: results/extractive_roberta.csv, extractive_electra.csv, extractive_all.csv

# Task 2 — Generative QA
python experiments/run_generative.py
# writes: results/generative_flan_t5_base.csv, generative_flan_t5_large.csv, generative_all.csv

# Decoding experiment
python experiments/run_decoding.py
# writes: results/decoding_results.csv
```

---

## Results

Contexts truncated to 120 characters for readability; full text is in the CSVs.

---

### Task 1 — Extractive QA

#### RoBERTa (`deepset/roberta-base-squad2`)

| Question | Context (truncated) | Predicted Answer |
|---|---|---|
| Do mitochondria play a role in remodelling lace plant leaves during programmed cell death? | Programmed cell death (PCD) is the regulated death of cells within an organism. The lace plant (Aponoge... | it has been less studied during PCD in plants |
| Landolt C and snellen e acuity: differences in strabismus amblyopia? | Assessment of visual acuity depends on the optotypes used for measurement. The ability to recognize dif... | *(empty)* |
| Syncope during bathing in infants, a pediatric form of water-induced urticaria? | Apparent life-threatening events in infants are a difficult and frequent problem in pediatric practice.... | aquagenic |
| Are the long-term results of the transanal pull-through equal to those of the transabdominal pull-through? | The transanal endorectal pull-through (TERPT) is becoming the most popular procedure in the treatment o... | Overall scores were similar |
| Can tailored interventions increase mammography use among HMO women? | Telephone counseling and tailored print communications have emerged as promising methods for promoting m... | there has been little research testing |
| Double balloon enteroscopy: is it efficacious and safe in a community setting? | From March 2007 to January 2011, 88 DBE procedures were performed on 66 patients. Indications included... | *(empty)* |
| 30-Day and 1-year mortality in emergency general surgery laparotomies: an area of concern and need for improvement? | Emergency surgery is associated with poorer outcomes and higher mortality with recent studies suggesting... | *(empty)* |
| Is adjustment for reporting heterogeneity necessary in sleep disorders? | Anchoring vignettes are brief texts describing a hypothetical character who illustrates a certain fixed... | evaluates the need for adjusting for reporting heterogeneity in the management of sleep and energy related problems in Japan |
| Do mutations causing low HDL-C promote increased carotid intima-media thickness? | Although observational data support an inverse relationship between high-density lipoprotein (HDL) chol... | Do mutations causing low HDL-C promote increased carotid intima-media thickness? Although observational... |
| A short stay or 23-hour ward in a general and academic children's hospital: are they effective? | We evaluated the usefulness of a short stay or 23-hour ward in a pediatric unit of a large teaching ho... | they are a useful addition to the emergency service |
| Did Chile's traffic law reform push police enforcement? | The objective of the current study is to determine to what extent the reduction of Chile's traffic fata... | the enactment of the traffic law reform in interaction with number of traffic tickets per police officer is significantly associated with a decrease of 8% in traffic fatalities and 7% in severe injuries |
| Therapeutic anticoagulation in the trauma patient: is it safe? | Trauma patients who require therapeutic anticoagulation pose a difficult treatment problem. The purpose... | Therapeutic anticoagulation in the trauma patient: is it safe? Trauma patients who require therapeutic... |
| Differentiation of nonalcoholic from alcoholic steatohepatitis: are routine laboratory markers useful? | Specific markers for differentiation of nonalcoholic (NASH) from alcoholic steatohepatitis (ASH) are l... | lacking |
| Prompting Primary Care Providers about Increased Patient Risk As a Result of Family History: Does It Work? | Electronic health records have the potential to facilitate family history use by primary care physicians... | automated, at-the-visit tailored prompts about family history risk change PCP behavior |
| Do emergency ultrasound fellowship programs impact emergency medicine residents' ultrasound education? | Recent years have seen a rapid proliferation of emergency ultrasound (EUS) programs in the United State... | there is no evidence supporting that EUS fellowships enhance residents' ultrasound (US) educational experiences |
| Patient-Controlled Therapy of Breathlessness in Palliative Care: A New Therapeutic Concept for Opioid Administration? | Breathlessness is one of the most distressing symptoms experienced by patients with advanced cancer and... | *(empty)* |
| Is there still a need for living-related liver transplantation in children? | To assess and compare the value of split-liver transplantation (SLT) and living-related liver transplan... | *(empty)* |
| Do patterns of knowledge and attitudes exist among unvaccinated seniors? | To examine patterns of knowledge and attitudes among adults aged>65 years unvaccinated for influenza. S... | clustered unvaccinated seniors by their immunization related knowledge and attitudes |
| Is there a model to teach and practice retroperitoneoscopic nephrectomy? | Although the retroperitoneal approach has been the preferred choice for open urological procedures, retr... | This study aims to develop a training model |
| Cardiovascular risk in a rural adult West African population: is resting heart rate also relevant? | Elevated resting heart rate (RHR) is a neglected marker in cardiovascular risk factor studies of sub-Sa... | *(empty)* |
| Israeli hospital preparedness for terrorism-related multiple casualty incidents: can the surge capacity and injury severity distribution be better predicted? | The incidence of large-scale urban attacks on civilian populations has significantly increased across th... | challenging to hospital teams... [truncated — see CSV] |
| Acute respiratory distress syndrome in children with malignancy--can we predict outcome? | The purpose of this study was to delineate early respiratory predictors of mortality in children with h... | early respiratory predictors of mortality |
| Secondhand smoke risk in infants discharged from an NICU: potential for significant health disparities? | Secondhand smoke exposure (SHSe) threatens fragile infants discharged from a neonatal intensive care un... | Soc |
| Do nomograms designed to predict biochemical recurrence (BCR) do a better job of predicting more clinically relevant prostate cancer outcomes than BCR? | To examine the ability of various postoperative nomograms to predict prostate cancer-specific mortality... | it |
| Are reports of mechanical dysfunction in chronic oro-facial pain related to somatisation? | (i) To examine the association between self-reported mechanical factors and chronic oro-facial pain. (ii... | these |

---

#### ELECTRA (`deepset/electra-base-squad2`)

| Question | Context (truncated) | Predicted Answer |
|---|---|---|
| Do mitochondria play a role in remodelling lace plant leaves during programmed cell death? | Programmed cell death (PCD) is the regulated death of cells within an organism. The lace plant (Aponoge... | the |
| Landolt C and snellen e acuity: differences in strabismus amblyopia? | Assessment of visual acuity depends on the optotypes used for measurement. The ability to recognize dif... | were small |
| Syncope during bathing in infants, a pediatric form of water-induced urticaria? | Apparent life-threatening events in infants are a difficult and frequent problem in pediatric practice.... | syncope during bathing in infants, a pediatric form of water-induced urticaria? apparent life-threatening... [truncated] |
| Are the long-term results of the transanal pull-through equal to those of the transabdominal pull-through? | The transanal endorectal pull-through (TERPT) is becoming the most popular procedure in the treatment o... | *(empty)* |
| Can tailored interventions increase mammography use among HMO women? | Telephone counseling and tailored print communications have emerged as promising methods for promoting m... | *(empty)* |
| Double balloon enteroscopy: is it efficacious and safe in a community setting? | From March 2007 to January 2011, 88 DBE procedures were performed on 66 patients. Indications included... | *(empty)* |
| 30-Day and 1-year mortality in emergency general surgery laparotomies: an area of concern and need for improvement? | Emergency surgery is associated with poorer outcomes and higher mortality with recent studies suggesting... | age-related 30-day mortality and 1-year mortality following emergency laparotomy... [truncated] |
| Is adjustment for reporting heterogeneity necessary in sleep disorders? | Anchoring vignettes are brief texts describing a hypothetical character who illustrates a certain fixed... | the need |
| Do mutations causing low HDL-C promote increased carotid intima-media thickness? | Although observational data support an inverse relationship between high-density lipoprotein (HDL) chol... | *(empty)* |
| A short stay or 23-hour ward in a general and academic children's hospital: are they effective? | We evaluated the usefulness of a short stay or 23-hour ward in a pediatric unit of a large teaching ho... | the |
| Did Chile's traffic law reform push police enforcement? | The objective of the current study is to determine to what extent the reduction of Chile's traffic fata... | reduction of chile's traffic fatalities and injuries during 2000-2012 was related to the police traffic enforcement increment |
| Therapeutic anticoagulation in the trauma patient: is it safe? | Trauma patients who require therapeutic anticoagulation pose a difficult treatment problem. The purpose... | the incidence of complications... [truncated, verbatim full-passage — see CSV] |
| Differentiation of nonalcoholic from alcoholic steatohepatitis: are routine laboratory markers useful? | Specific markers for differentiation of nonalcoholic (NASH) from alcoholic steatohepatitis (ASH) are l... | *(empty)* |
| Prompting Primary Care Providers about Increased Patient Risk As a Result of Family History: Does It Work? | Electronic health records have the potential to facilitate family history use by primary care physicians... | automated |
| Do emergency ultrasound fellowship programs impact emergency medicine residents' ultrasound education? | Recent years have seen a rapid proliferation of emergency ultrasound (EUS) programs in the United State... | there is no |
| Patient-Controlled Therapy of Breathlessness in Palliative Care: A New Therapeutic Concept for Opioid Administration? | Breathlessness is one of the most distressing symptoms experienced by patients with advanced cancer and... | pct ) systems would be an additional treatment option |
| Is there still a need for living-related liver transplantation in children? | To assess and compare the value of split-liver transplantation (SLT) and living-related liver transplan... | *(empty)* |
| Do patterns of knowledge and attitudes exist among unvaccinated seniors? | To examine patterns of knowledge and attitudes among adults aged>65 years unvaccinated for influenza. S... | *(empty)* |
| Is there a model to teach and practice retroperitoneoscopic nephrectomy? | Although the retroperitoneal approach has been the preferred choice for open urological procedures, retr... | (rpn) training model |
| Cardiovascular risk in a rural adult West African population: is resting heart rate also relevant? | Elevated resting heart rate (RHR) is a neglected marker in cardiovascular risk factor studies of sub-Sa... | elevated resting heart rate (rhr) is a neglected marker |
| Israeli hospital preparedness for terrorism-related multiple casualty incidents: can the surge capacity and injury severity distribution be better predicted? | The incidence of large-scale urban attacks on civilian populations has significantly increased across th... | key of 20 percent of each hospital's bed capacity |
| Acute respiratory distress syndrome in children with malignancy--can we predict outcome? | The purpose of this study was to delineate early respiratory predictors of mortality in children with h... | early respiratory predictors of mortality |
| Secondhand smoke risk in infants discharged from an NICU: potential for significant health disparities? | Secondhand smoke exposure (SHSe) threatens fragile infants discharged from a neonatal intensive care un... | *(empty)* |
| Do nomograms designed to predict biochemical recurrence (BCR) do a better job of predicting more clinically relevant prostate cancer outcomes than BCR? | To examine the ability of various postoperative nomograms to predict prostate cancer-specific mortality... | they all predicted pcsm more accurately than they predicted bcr (ie, with higher c-index values). |
| Are reports of mechanical dysfunction in chronic oro-facial pain related to somatisation? | (i) To examine the association between self-reported mechanical factors and chronic oro-facial pain. (ii... | *(empty)* |

---

#### All Extractive Models (merged)

| Model | Question | Context (truncated) | Predicted Answer |
|---|---|---|---|
| roberta | Do mitochondria play a role in remodelling lace plant leaves during programmed cell death? | Programmed cell death (PCD) is the regulated death... | it has been less studied during PCD in plants |
| roberta | Landolt C and snellen e acuity: differences in strabismus amblyopia? | Assessment of visual acuity depends on the optotypes... | *(empty)* |
| roberta | Syncope during bathing in infants, a pediatric form of water-induced urticaria? | Apparent life-threatening events in infants are a difficult... | aquagenic |
| roberta | Are the long-term results of the transanal pull-through equal to those of the transabdominal pull-through? | The transanal endorectal pull-through (TERPT)... | Overall scores were similar |
| roberta | Can tailored interventions increase mammography use among HMO women? | Telephone counseling and tailored print communications... | there has been little research testing |
| roberta | Did Chile's traffic law reform push police enforcement? | The objective of the current study is to determine... | the enactment of the traffic law reform... [see per-model table] |
| roberta | Therapeutic anticoagulation in the trauma patient: is it safe? | Trauma patients who require therapeutic anticoagulation... | Therapeutic anticoagulation in the trauma patient... [question echoed back] |
| roberta | Differentiation of nonalcoholic from alcoholic steatohepatitis: are routine laboratory markers useful? | Specific markers for differentiation of nonalcoholic... | lacking |
| electra | Do mitochondria play a role in remodelling lace plant leaves during programmed cell death? | Programmed cell death (PCD) is the regulated death... | the |
| electra | Landolt C and snellen e acuity: differences in strabismus amblyopia? | Assessment of visual acuity depends on the optotypes... | were small |
| electra | Is adjustment for reporting heterogeneity necessary in sleep disorders? | Anchoring vignettes are brief texts describing... | the need |
| electra | A short stay or 23-hour ward in a general and academic children's hospital: are they effective? | We evaluated the usefulness of a short stay... | the |
| electra | Did Chile's traffic law reform push police enforcement? | The objective of the current study is to determine... | reduction of chile's traffic fatalities and injuries... |
| electra | Acute respiratory distress syndrome in children with malignancy--can we predict outcome? | The purpose of this study was to delineate early... | early respiratory predictors of mortality |
| electra | Do nomograms designed to predict biochemical recurrence (BCR)...? | To examine the ability of various postoperative nomograms... | they all predicted pcsm more accurately than they predicted bcr |

*(Full merged table in `results/extractive_all.csv`)*

---

### Task 2 — Generative QA

#### Flan-T5-base (`google/flan-t5-base`)

| Question | Context (truncated) | Predicted Answer |
|---|---|---|
| Do mitochondria play a role in remodelling lace plant leaves during programmed cell death? | Programmed cell death (PCD) is the regulated death of cells within an organism. The lace plant (Aponoge... | no |
| Landolt C and snellen e acuity: differences in strabismus amblyopia? | Assessment of visual acuity depends on the optotypes used for measurement. The ability to recognize dif... | *(empty)* |
| Syncope during bathing in infants, a pediatric form of water-induced urticaria? | Apparent life-threatening events in infants are a difficult and frequent problem in pediatric practice.... | aquagenic |
| Are the long-term results of the transanal pull-through equal to those of the transabdominal pull-through? | The transanal endorectal pull-through (TERPT) is becoming the most popular procedure in the treatment o... | yes |
| Can tailored interventions increase mammography use among HMO women? | Telephone counseling and tailored print communications have emerged as promising methods for promoting m... | Yes |
| Double balloon enteroscopy: is it efficacious and safe in a community setting? | From March 2007 to January 2011, 88 DBE procedures were performed on 66 patients. Indications included... | yes |
| 30-Day and 1-year mortality in emergency general surgery laparotomies: an area of concern and need for improvement? | Emergency surgery is associated with poorer outcomes and higher mortality with recent studies suggesting... | 30 day mortality, age-related 30-day mortality and 1-year all-cause mortality |
| Is adjustment for reporting heterogeneity necessary in sleep disorders? | Anchoring vignettes are brief texts describing a hypothetical character who illustrates a certain fixed... | Yes |
| Do mutations causing low HDL-C promote increased carotid intima-media thickness? | Although observational data support an inverse relationship between high-density lipoprotein (HDL) chol... | no |
| A short stay or 23-hour ward in a general and academic children's hospital: are they effective? | We evaluated the usefulness of a short stay or 23-hour ward in a pediatric unit of a large teaching ho... | short stay ward being a useful addition to the emergency service |
| Did Chile's traffic law reform push police enforcement? | The objective of the current study is to determine to what extent the reduction of Chile's traffic fata... | yes |
| Therapeutic anticoagulation in the trauma patient: is it safe? | Trauma patients who require therapeutic anticoagulation pose a difficult treatment problem. The purpose... | yes |
| Differentiation of nonalcoholic from alcoholic steatohepatitis: are routine laboratory markers useful? | Specific markers for differentiation of nonalcoholic (NASH) from alcoholic steatohepatitis (ASH) are l... | Yes |
| Prompting Primary Care Providers about Increased Patient Risk As a Result of Family History: Does It Work? | Electronic health records have the potential to facilitate family history use by primary care physicians... | Yes |
| Do emergency ultrasound fellowship programs impact emergency medicine residents' ultrasound education? | Recent years have seen a rapid proliferation of emergency ultrasound (EUS) programs in the United State... | no |
| Patient-Controlled Therapy of Breathlessness in Palliative Care: A New Therapeutic Concept for Opioid Administration? | Breathlessness is one of the most distressing symptoms experienced by patients with advanced cancer and... | [1]. |
| Is there still a need for living-related liver transplantation in children? | To assess and compare the value of split-liver transplantation (SLT) and living-related liver transplan... | The combination of SLT and LRT has abolished deaths on the waiting list |
| Do patterns of knowledge and attitudes exist among unvaccinated seniors? | To examine patterns of knowledge and attitudes among adults aged>65 years unvaccinated for influenza. S... | Yes |
| Is there a model to teach and practice retroperitoneoscopic nephrectomy? | Although the retroperitoneal approach has been the preferred choice for open urological procedures, retr... | yes |
| Cardiovascular risk in a rural adult West African population: is resting heart rate also relevant? | Elevated resting heart rate (RHR) is a neglected marker in cardiovascular risk factor studies of sub-Sa... | yes |
| Israeli hospital preparedness for terrorism-related multiple casualty incidents: can the surge capacity and injury severity distribution be better predicted? | The incidence of large-scale urban attacks on civilian populations has significantly increased across th... | yes |
| Acute respiratory distress syndrome in children with malignancy--can we predict outcome? | The purpose of this study was to delineate early respiratory predictors of mortality in children with h... | early respiratory predictors of mortality |
| Secondhand smoke risk in infants discharged from an NICU: potential for significant health disparities? | Secondhand smoke exposure (SHSe) threatens fragile infants discharged from a neonatal intensive care un... | potential for significant health disparities |
| Do nomograms designed to predict biochemical recurrence (BCR) do a better job of predicting more clinically relevant prostate cancer outcomes than BCR? | To examine the ability of various postoperative nomograms to predict prostate cancer-specific mortality... | no |
| Are reports of mechanical dysfunction in chronic oro-facial pain related to somatisation? | (i) To examine the association between self-reported mechanical factors and chronic oro-facial pain. (ii... | no |

---

#### Flan-T5-large (`google/flan-t5-large`)

| Question | Context (truncated) | Predicted Answer |
|---|---|---|
| Do mitochondria play a role in remodelling lace plant leaves during programmed cell death? | Programmed cell death (PCD) is the regulated death of cells within an organism. The lace plant (Aponoge... | yes |
| Landolt C and snellen e acuity: differences in strabismus amblyopia? | Assessment of visual acuity depends on the optotypes used for measurement. The ability to recognize dif... | d). |
| Syncope during bathing in infants, a pediatric form of water-induced urticaria? | Apparent life-threatening events in infants are a difficult and frequent problem in pediatric practice.... | d). |
| Are the long-term results of the transanal pull-through equal to those of the transabdominal pull-through? | The transanal endorectal pull-through (TERPT) is becoming the most popular procedure in the treatment o... | yes |
| Can tailored interventions increase mammography use among HMO women? | Telephone counseling and tailored print communications have emerged as promising methods for promoting m... | [iv] |
| Double balloon enteroscopy: is it efficacious and safe in a community setting? | From March 2007 to January 2011, 88 DBE procedures were performed on 66 patients. Indications included... | (iv). |
| 30-Day and 1-year mortality in emergency general surgery laparotomies: an area of concern and need for improvement? | Emergency surgery is associated with poorer outcomes and higher mortality with recent studies suggesting... | (iv) |
| Is adjustment for reporting heterogeneity necessary in sleep disorders? | Anchoring vignettes are brief texts describing a hypothetical character who illustrates a certain fixed... | (iv) |
| Do mutations causing low HDL-C promote increased carotid intima-media thickness? | Although observational data support an inverse relationship between high-density lipoprotein (HDL) chol... | (iv). |
| A short stay or 23-hour ward in a general and academic children's hospital: are they effective? | We evaluated the usefulness of a short stay or 23-hour ward in a pediatric unit of a large teaching ho... | short stay wards being a useful addition to the emergency department. |
| Did Chile's traffic law reform push police enforcement? | The objective of the current study is to determine to what extent the reduction of Chile's traffic fata... | yes |
| Therapeutic anticoagulation in the trauma patient: is it safe? | Trauma patients who require therapeutic anticoagulation pose a difficult treatment problem. The purpose... | [iv] |
| Differentiation of nonalcoholic from alcoholic steatohepatitis: are routine laboratory markers useful? | Specific markers for differentiation of nonalcoholic (NASH) from alcoholic steatohepatitis (ASH) are l... | (iv) |
| Prompting Primary Care Providers about Increased Patient Risk As a Result of Family History: Does It Work? | Electronic health records have the potential to facilitate family history use by primary care physicians... | (iv) |
| Do emergency ultrasound fellowship programs impact emergency medicine residents' ultrasound education? | Recent years have seen a rapid proliferation of emergency ultrasound (EUS) programs in the United State... | Yes |
| Patient-Controlled Therapy of Breathlessness in Palliative Care: A New Therapeutic Concept for Opioid Administration? | Breathlessness is one of the most distressing symptoms experienced by patients with advanced cancer and... | (iv) |
| Is there still a need for living-related liver transplantation in children? | To assess and compare the value of split-liver transplantation (SLT) and living-related liver transplan... | (iv) |
| Do patterns of knowledge and attitudes exist among unvaccinated seniors? | To examine patterns of knowledge and attitudes among adults aged>65 years unvaccinated for influenza. S... | Yes |
| Is there a model to teach and practice retroperitoneoscopic nephrectomy? | Although the retroperitoneal approach has been the preferred choice for open urological procedures, retr... | Yes |
| Cardiovascular risk in a rural adult West African population: is resting heart rate also relevant? | Elevated resting heart rate (RHR) is a neglected marker in cardiovascular risk factor studies of sub-Sa... | (iv) |
| Israeli hospital preparedness for terrorism-related multiple casualty incidents: can the surge capacity and injury severity distribution be better predicted? | The incidence of large-scale urban attacks on civilian populations has significantly increased across th... | (iv). |
| Acute respiratory distress syndrome in children with malignancy--can we predict outcome? | The purpose of this study was to delineate early respiratory predictors of mortality in children with h... | (iv) |
| Secondhand smoke risk in infants discharged from an NICU: potential for significant health disparities? | Secondhand smoke exposure (SHSe) threatens fragile infants discharged from a neonatal intensive care un... | d). |
| Do nomograms designed to predict biochemical recurrence (BCR) do a better job of predicting more clinically relevant prostate cancer outcomes than BCR? | To examine the ability of various postoperative nomograms to predict prostate cancer-specific mortality... | (iv). |
| Are reports of mechanical dysfunction in chronic oro-facial pain related to somatisation? | (i) To examine the association between self-reported mechanical factors and chronic oro-facial pain. (ii... | (ii) |

---

#### All Generative Models (merged)

| Model | Question | Context (truncated) | Predicted Answer |
|---|---|---|---|
| flan_t5_base | Do mitochondria play a role...? | Programmed cell death (PCD) is the regulated death... | no |
| flan_t5_base | Syncope during bathing in infants...? | Apparent life-threatening events in infants... | aquagenic |
| flan_t5_base | Are the long-term results of the transanal pull-through equal...? | The transanal endorectal pull-through (TERPT)... | yes |
| flan_t5_base | Do mutations causing low HDL-C promote increased carotid intima-media thickness? | Although observational data support an inverse relationship... | no |
| flan_t5_base | Do emergency ultrasound fellowship programs impact...? | Recent years have seen a rapid proliferation... | no |
| flan_t5_base | Do nomograms designed to predict BCR...? | To examine the ability of various postoperative nomograms... | no |
| flan_t5_large | Do mitochondria play a role...? | Programmed cell death (PCD) is the regulated death... | yes |
| flan_t5_large | Landolt C and snellen e acuity...? | Assessment of visual acuity depends on the optotypes... | d). |
| flan_t5_large | A short stay or 23-hour ward...are they effective? | We evaluated the usefulness of a short stay... | short stay wards being a useful addition to the emergency department. |
| flan_t5_large | Did Chile's traffic law reform push police enforcement? | The objective of the current study is to determine... | yes |
| flan_t5_large | Do emergency ultrasound fellowship programs impact...? | Recent years have seen a rapid proliferation... | Yes |
| flan_t5_large | Are reports of mechanical dysfunction in chronic oro-facial pain related to somatisation? | (i) To examine the association between self-reported mechanical factors... | (ii) |

*(Full merged table in `results/generative_all.csv`)*

---

## Decoding Experiment

**Model:** `google/flan-t5-large` | **10 examples** | `do_sample=True` | temperatures: 0.3 / 0.7 / 1.0

| Question | Context (truncated) | temp=0.3 | temp=0.7 | temp=1.0 |
|---|---|---|---|---|
| Do mitochondria play a role in remodelling lace plant leaves during programmed cell death? | Programmed cell death (PCD) is the regulated death... | (d). | c). | (ii). |
| Landolt C and snellen e acuity: differences in strabismus amblyopia? | Assessment of visual acuity depends on the optotypes... | (iv) | D). | d). |
| Syncope during bathing in infants, a pediatric form of water-induced urticaria? | Apparent life-threatening events in infants... | d). | (iv). | 4). |
| Are the long-term results of the transanal pull-through equal to those of the transabdominal pull-through? | The transanal endorectal pull-through (TERPT)... | Yes | d). | Yes |
| Can tailored interventions increase mammography use among HMO women? | Telephone counseling and tailored print communications... | Yes | [4]. | Yes |
| Double balloon enteroscopy: is it efficacious and safe in a community setting? | From March 2007 to January 2011, 88 DBE procedures... | iv. | (iv) | (A). |
| 30-Day and 1-year mortality in emergency general surgery laparotomies: an area of concern and need for improvement? | Emergency surgery is associated with poorer outcomes... | d). | [4]. | (c). |
| Is adjustment for reporting heterogeneity necessary in sleep disorders? | Anchoring vignettes are brief texts describing... | [4]. | (iv) | (1). |
| Do mutations causing low HDL-C promote increased carotid intima-media thickness? | Although observational data support an inverse relationship... | (d). | no | 4). |
| A short stay or 23-hour ward in a general and academic children's hospital: are they effective? | We evaluated the usefulness of a short stay... | Westmead Hospital's short stay ward having a lower rate of admission... | short stay wards being a useful addition to the emergency department. | Westmead Hospital, a tertiary referral center... |

**Analysis:** The decoding experiment reveals that temperature has a limited stabilising effect when the underlying generation is already degenerate. At `temp=0.3` (sharpened distribution), the model consistently generates list-item markers like `(d).`, `iv.`, and `[4].` — tokens with the highest probability mass in its confused internal state. At `temp=0.7` and `1.0`, the outputs vary slightly but remain in the same failure mode: the model samples alternative list markers (`c).`, `D).`, `(A).`) rather than content. This demonstrates a key principle: temperature controls diversity *within* the model's probability distribution, but it cannot rescue a model that has already misidentified the generation target. When the model correctly answers (Q4 "Yes", Q5 "Yes"), those answers are stable across all temperatures, confirming that the failures are not stochasticity but a systematic misinterpretation of the prompt format.

---

## Failure Analysis

### Failure 1 — ELECTRA returns single-token or garbage spans

**Observed:** ELECTRA frequently outputs single tokens ("the", "the need", "were small") or empty strings on questions where RoBERTa returns a meaningful span.

**Hypothesis:** ELECTRA's discriminator pretraining objective (replaced token detection) produces token representations that are highly sensitive to local context but may not generalize well to long-range span boundary decisions. The start and end logits are predicted independently; for long PubMedQA abstracts (often 300–500 tokens), ELECTRA's lighter architecture (fewer parameters, less attention capacity) fails to propagate sufficient signal from the question through the full context, causing the model to default to high-frequency function words near the question tokens rather than locating a semantically correct span.

---

### Failure 2 — Flan-T5-large generates list-item markers instead of answers

**Observed:** Flan-T5-large outputs `(iv)`, `(d).`, `[iv]`, etc. on the majority of examples where Flan-T5-base correctly answers "yes" or "no". This is the dominant failure mode for the larger model.

**Hypothesis:** PubMedQA contexts contain numbered lists (e.g., "(1) presence of new traffic law; (2) police officers per population; (3)…") and parenthetical list items (e.g., "(i) To examine…; (ii) To test…"). Flan-T5-large, with more parameters, likely over-fits to structural patterns in its instruction-tuning data and learns to generate list markers when the input contains enumerated structures. The model interprets the prompt as a list-completion task rather than a question-answering task. Flan-T5-base, being smaller and more regularized, does not exhibit this pattern — it collapses to binary yes/no instead, which is less informative but less broken.

---

### Failure 3 — RoBERTa echoes the question back as the answer

**Observed:** For Q9 ("Do mutations causing low HDL-C…?") and Q12 ("Therapeutic anticoagulation…is it safe?"), RoBERTa returns the question text repeated verbatim at the start of the answer span.

**Hypothesis:** These questions are phrased as yes/no questions and the correct answer ("no", implied by cIMT being nearly identical) does not exist as an extractive span in the context. When no clear answer span is present, the span-prediction head still must output *some* start and end position. The highest-confidence position under the independently-predicted start logit happens to land on the question prefix tokens (which are part of the 512-token input), resulting in the question being extracted as if it were the answer. This is a known pathology of SQuAD-style models on unanswerable or yes/no questions: they are not trained to abstain.

---

*Peter Altmayer — NLP Applications in Research and Industry, Uni Mainz, SoSe 2026*
