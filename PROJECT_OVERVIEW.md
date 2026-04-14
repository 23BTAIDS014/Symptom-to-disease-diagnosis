# Symptom-to-Diseases Diagnosis — Project Overview

## 1. What We Have Created (Project Purpose)

This project is a **symptom-to-disease diagnosis support system** that uses **Natural Language Processing (NLP)** and **deep learning** to:

1. **Take free-text symptom descriptions** from a user (e.g. “headache, fever, cough, fatigue”).
2. **Suggest possible diseases** that might explain those symptoms (e.g. Influenza, Common cold, COVID-19, Migraine), **not** just echo the symptoms back.
3. **Rank those diseases with confidence scores** using a neural network, so users see which conditions are more or less likely.
4. **Optionally extract medical entities** (diseases, conditions, symptoms) from longer clinical text (e.g. admission notes) using a BERT-based Named Entity Recognition (NER) model.

So the project combines:
- **Symptom → disease mapping** (rule-based + neural ranking).
- **Medical NER** (BERT/BioBERT) for extracting conditions from narrative text.
- A **web app** where users enter symptoms and see possible diseases with confidence.

---

## 2. Uses (Applications)

| Use case | Description |
|----------|-------------|
| **Patient self-check / triage** | Patients describe symptoms and get a list of possible conditions with confidence to discuss with a doctor. |
| **Clinical note analysis** | Extract diseases/conditions from admission notes or clinical narratives (NER). |
| **Education / demos** | Show how NLP and deep learning can support symptom–disease reasoning and entity extraction. |
| **Research / extension** | Base for adding severity scoring, follow-up questions, or linking to medical knowledge bases. |

**Important:** The system is for **decision support only**. It does not replace a doctor; any result should be validated by a healthcare professional.

---

## 3. Programming Languages & Technologies

### Backend
- **Language:** **Python**
- **Framework:** **Flask** (REST API)
- **Server:** **gevent** (WSGI server for production-like serving)
- **Libraries:**  
  - **PyTorch** — deep learning (BERT, model loading, inference)  
  - **Transformers (Hugging Face)** — BERT tokenizer, `BertForTokenClassification`, zero-shot pipeline  
  - **spaCy** — sentence segmentation (`en_core_web_sm`)  
  - **Flask-CORS** — allow frontend to call the API  
  - **NumPy, Pandas** — data handling  
  - **scikit-learn, NLTK** — used in training/utilities (e.g. `main.py`, data loading)

### Frontend
- **Language:** **TypeScript**
- **Framework:** **Angular 16**
- **Styling:** **Tailwind CSS** (utility-first CSS)
- **HTTP:** **HttpClient** to call the backend API
- **UI:** Single-page app with a form (symptom input), buttons, and a results panel with confidence bars and rankings

### Summary Table

| Layer    | Language / tech      | Role |
|----------|----------------------|------|
| Backend  | Python, Flask        | API, NLP, ML inference |
| ML/NLP   | PyTorch, Transformers| BERT NER, zero-shot ranking |
| NLP utils| spaCy                | Sentence splitting |
| Frontend | TypeScript, Angular 16 | UI and API calls |
| Styling  | Tailwind CSS         | Layout and appearance |

---

## 4. Models Used

### 4.1 BERT-based Medical NER (Named Entity Recognition)

- **Purpose:** Extract medical entities (e.g. diseases, conditions) from free text (e.g. admission notes). Each token is labeled as **B-** (begin), **I-** (inside), or **O** (outside) for a given entity type.
- **Two modes:**
  1. **Base BERT (`BertNER`)**  
     - Model: **bert-base-uncased** (Hugging Face) with a token-classification head.  
     - Entity types: Medical Condition, Symptom, Medication, Vital Statistic, Measurement Value, Negation Cue, Medical Procedure (configurable via `-t`).  
     - Weights: Loaded from a local `.pth` file (e.g. `../models/medcondbert.pth`) if present; otherwise runs in demo mode with untrained head.
  2. **BioBERT disease NER (`BioBertNER`)**  
     - Model: **alvaroalon2/biobert_diseases_ner** (Hugging Face) — BioBERT fine-tuned for disease NER.  
     - Entity type: Disease (B-DISEASE, I-DISEASE, O).  
     - Can run **without** a local `.pth` file; uses the pretrained Hugging Face weights.
- **Used in API:**  
  - `POST /extract_entities` — returns token-level labels.  
  - `POST /extract_entities_structured` — returns tokens, labels, and grouped entity spans.

### 4.2 Zero-shot classification (disease ranking with confidence)

- **Purpose:** Rank possible diseases by relevance to the user’s symptom text and assign a **confidence score** (0–1) to each.
- **Model:** **facebook/bart-large-mnli** (Hugging Face `zero-shot-classification` pipeline).
- **How it’s used:**  
  - Candidate diseases come from the symptom→disease mapping (or a full list if no mapping match).  
  - The pipeline scores how well the symptom text fits each disease label (multi-label).  
  - Results are sorted by score and returned as `ranked_diseases: [{ name, confidence }]`.
- **Used in API:**  
  - `POST /suggest_diseases` — returns `possible_diseases`, `symptoms_considered`, and `ranked_diseases` (with confidence).

### 4.3 Rule-based symptom → disease mapping

- **Purpose:** Map symptom phrases (e.g. “headache”, “chest pain”) to a list of possible diseases so the system suggests **diseases**, not symptoms.
- **Implementation:** In-memory Python dictionary `SYMPTOM_TO_DISEASES` (e.g. headache → Migraine, Tension headache, Hypertension, …).
- **Role:** Feeds candidate diseases into the zero-shot ranker and ensures the UI shows conditions, not echoed symptoms.

### Summary of models

| Model / component              | Type              | Role |
|-------------------------------|-------------------|------|
| bert-base-uncased             | Transformer (NER) | Token-level entity labels (optional; needs trained .pth) |
| alvaroalon2/biobert_diseases_ner | BioBERT (NER)  | Disease NER from text (can run without .pth) |
| facebook/bart-large-mnli     | Zero-shot classifier | Rank diseases and assign confidence from symptom text |
| SYMPTOM_TO_DISEASES (dict)   | Rule-based        | Map symptom phrases → candidate diseases |
| spaCy en_core_web_sm          | NLP pipeline      | Sentence segmentation for NER |

---

## 5. High-level architecture

```
[User] → [Angular UI] → HTTP POST /suggest_diseases (symptom text)
                              ↓
                    [Flask API]
                              ↓
              ┌───────────────┴───────────────┐
              ↓                               ↓
    SYMPTOM_TO_DISEASES              rank_diseases_with_nn()
    (symptom → diseases)              (BART zero-shot)
              ↓                               ↓
              └───────────────┬───────────────┘
                              ↓
                    JSON: possible_diseases,
                          ranked_diseases (name, confidence),
                          symptoms_considered
                              ↓
              [Angular] shows list + confidence bars
```

For NER (extract entities from long text):

```
[User] → [Angular] → POST /extract_entities or /extract_entities_structured
                              ↓
                    [Flask] → spaCy (split sentences)
                              ↓
                    BERT/BioBERT NER (token classification)
                              ↓
                    JSON: tokens, entities, (optional) spans
```

---

## 6. Datasets Used

### 6.1 Labelled NER data (training the BERT/BioBERT NER model)

- **Location:** `datasets/labelled_data/`
- **Format:** CSV files with pipe (`|`) separator. Each row has:
  - **Column 1 (text):** Raw clinical/medical sentence or paragraph.
  - **Column 2 (entity):** Space-separated token-level labels in **BIO** scheme: `B-<TYPE>`, `I-<TYPE>`, `O`.
- **Entity types (subfolders):**
  - **MEDCOND** — Medical conditions (used for disease NER and for BioBERT transfer learning).
  - **SYMPTOM**, **MEDICATION**, **VITALSTAT**, **MEASVAL**, **NEGATION**, **PROCEDURE** — Other annotation types if you train with `-t` (e.g. Symptom, Medication, etc.).
- **Files:** Each type has `all.csv` (and optionally `all.conll`). Training uses `all.csv`; split is 70% train, ~20% val, ~10% test (in code).
- **Used by:** `main.py` (training), `Dataloader` in `utils/dataloader.py`. The API (`api.py`) does **not** load this; it only loads the trained model weights.

### 6.2 TREC Clinical Trials (TREC-CDS) topics — source of medical text

- **Location:** `datasets/xml/` (topics2021.xml, topics2022.xml, topics2023.xml, topics2016) and `datasets/txt/topics2021/`, `topics2022/`, `topics2023/` (plain-text versions).
- **What it is:** **TREC Clinical Decision Support (TREC-CDS)** topics: short clinical case descriptions (e.g. patient history, diagnoses, treatments) in the form of “admission notes” or case summaries. Used in TREC evaluation tasks.
- **Role:** Source material for building or describing the kind of text the NER model is meant to handle. The **labelled_data** CSVs may be derived from or inspired by such topics (or similar clinical text); the exact annotation pipeline (how XML/txt became MEDCOND/SYMPTOM labels) is project-specific.
- **Reference:** TREC-CDS topics are publicly available (e.g. [trec-cds.org](http://trec-cds.org/)).

### 6.3 Symptom → disease mapping (not a dataset)

- **Location:** In code only — Python dict `SYMPTOM_TO_DISEASES` in `src/api.py`.
- **What it is:** Hand-curated mapping from symptom phrases (e.g. "headache", "chest pain") to lists of possible diseases. Not loaded from a file; no external dataset.

### 6.4 Zero-shot and pretrained models

- **facebook/bart-large-mnli:** Pretrained on general NLI data (MNLI). No project-specific dataset; used as-is for ranking disease labels.
- **bert-base-uncased / alvaroalon2/biobert_diseases_ner:** Pretrained on general or biomedical text. Optional fine-tuning in this project uses the **labelled_data** CSVs above.

---

## 7. How to run the project

- **Backend (from project root):**  
  `cd Disease-Detection-NLP/src`  
  `python api.py -t "Medical Condition" -tr`  
  (Use `-tr` to use BioBERT disease NER without a local .pth file.)
- **Frontend:**  
  `cd Disease-Detection-NLP/frontend`  
  `npm install` then `npm start`  
  Open **http://localhost:4200**, enter symptoms, and click “Get possible diseases” to see ranked results with confidence.

This document summarizes what the project is for, its uses, the programming languages and frameworks, and the models used end to end.
