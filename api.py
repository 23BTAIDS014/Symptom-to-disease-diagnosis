import argparse
parser = argparse.ArgumentParser(description='The backend of the specified frontend. Service obtains sentences and predicts entities.')

parser.add_argument('-l', '--length', type=int, default=128,
                    help='Choose the maximum length of the model\'s input layer.')
parser.add_argument('-m', '--model', type=str, default='../models/medcondbert.pth',
                    help='Choose the directory of the model to be used for prediction.')
parser.add_argument('-tr', '--transfer_learning', action='store_true',
                    help='Use BioBERT disease NER from Hugging Face (already trained). No .pth file needed.')
parser.add_argument('-p', '--port', type=int, default=5000,
                    help='The port on which the model is going to run.')
parser.add_argument('-t', '--type', type=str, required=True,
                    help='Specify the type of annotation to process. Type of annotation needs to be one of the following: Medical Condition, Symptom, Medication, Vital Statistic, Measurement Value, Negation Cue, Medical Procedure')

args = parser.parse_args()

max_length = args.length
model_path = args.model
# Resolve to absolute path for existence check and loading
import os
_model_dir = os.path.dirname(os.path.abspath(__file__))
model_path_abs = os.path.normpath(os.path.join(_model_dir, model_path))
transfer_learning = args.transfer_learning
port = args.port
model_loaded = False  # track if we loaded trained weights

print("Preparing model...")

from gevent.pywsgi import WSGIServer # Imports the WSGIServer
from gevent import monkey; monkey.patch_all()
from flask import Flask, request, jsonify
from flask_cors import CORS
from utils.dataloader import Dataloader
from utils.BertArchitecture import BertNER, BioBertNER
from utils.metric_tracking import MetricsTracking
import torch
from torch.optim import SGD
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer, BertForTokenClassification, pipeline
import spacy

# initializing backend
if not args.transfer_learning:
    print("Training base BERT model...")
    model = BertNER(3) #O, B-, I- -> 3 entities

    if args.type == 'Medical Condition':
        type = 'MEDCOND'
    elif args.type == 'Symptom':
        type = 'SYMPTOM'
    elif args.type == 'Medication':
        type = 'MEDICATION'
    elif args.type == 'Vital Statistic':
        type = 'VITALSTAT'
    elif args.type == 'Measurement Value':
        type = 'MEASVAL'
    elif args.type == 'Negation Cue':
        type = 'NEGATION'
    elif args.type == 'Medical Procedure':
        type = 'PROCEDURE'
    else:    
        raise ValueError('Type of annotation needs to be one of the following: Medical Condition, Symptom, Medication, Vital Statistic, Measurement Value, Negation Cue, Medical Procedure')
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_tokens(['B-' + args.type, 'I-' + args.type])
else:
    print("Training BERT model based on BioBERT diseases...")

    if not args.type == 'Medical Condition':
        raise ValueError('Type of annotation needs to be Medical Condition when using BioBERT as baseline.')

    model = BioBertNER(3) #O, B-, I- -> 3 entities
    tokenizer = BertTokenizer.from_pretrained('alvaroalon2/biobert_diseases_ner')
    type = 'DISEASE'

label_to_ids = {
    'B-' + type: 0,
    'I-' + type: 1,
    'O': 2
    }

ids_to_label = {
    0:'B-' + type,
    1:'I-' + type,
    2:'O'
    }

try:
    model.load_state_dict(torch.load(model_path_abs, map_location='cpu'))
    model_loaded = True
    print("Loaded trained weights from:", model_path_abs)
except FileNotFoundError:
    if args.transfer_learning:
        # BioBERT path: HF model is already trained for disease NER — use it as-is
        model_loaded = True
        print("\n*** No local .pth file; using Hugging Face BioBERT disease NER (already trained).")
        print("*** To use your own fine-tuned weights, train with main.py -tr True and pass -m path/to/model.pth\n")
    else:
        model_loaded = False
        print("\n*** WARNING: No trained model found at", model_path_abs)
        print("*** Running in DEMO mode with untrained weights. Predictions will be poor.")
        print("*** To use a trained model now:  python api.py -t \"Medical Condition\" -tr")
        print("*** To train your own (base BERT):  python main.py -t \"Medical Condition\" -o ../models/medcondbert.pth\n")
model.eval()

app = Flask(__name__)
CORS(app)  # Initialize CORS

sentence_detector = spacy.load("en_core_web_sm")

# Symptom → possible diseases (for suggesting diseases from patient symptoms, not echoing symptoms back)
SYMPTOM_TO_DISEASES = {
    "headache": ["Migraine", "Tension headache", "Hypertension", "Meningitis", "Sinusitis", "Dehydration"],
    "fever": ["Influenza (flu)", "Common cold", "COVID-19", "Urinary tract infection", "Pneumonia", "Viral infection"],
    "cough": ["Common cold", "Influenza (flu)", "COVID-19", "Bronchitis", "Pneumonia", "Asthma", "GERD"],
    "fatigue": ["Anemia", "Hypothyroidism", "Chronic fatigue syndrome", "Depression", "Sleep disorders", "Diabetes"],
    "nausea": ["Gastroenteritis", "Migraine", "Pregnancy", "GERD", "Pancreatitis", "Motion sickness"],
    "chest pain": ["Angina", "Heart attack", "GERD", "Pleurisy", "Anxiety", "Costochondritis"],
    "abdominal pain": ["Gastroenteritis", "Appendicitis", "Irritable bowel syndrome", "GERD", "Gallstones", "UTI"],
    "sore throat": ["Pharyngitis", "Common cold", "Strep throat", "Influenza", "COVID-19", "Allergies"],
    "runny nose": ["Common cold", "Allergic rhinitis", "Sinusitis", "Influenza", "COVID-19"],
    "shortness of breath": ["Asthma", "COPD", "Pneumonia", "Heart failure", "Anxiety", "Anemia"],
    "joint pain": ["Osteoarthritis", "Rheumatoid arthritis", "Lupus", "Gout", "Viral infection"],
    "back pain": ["Muscle strain", "Herniated disc", "Osteoarthritis", "Kidney infection", "Sciatica"],
    "dizziness": ["Vertigo", "Low blood pressure", "Anemia", "Dehydration", "Inner ear disorder", "Anxiety"],
    "rash": ["Allergic reaction", "Eczema", "Psoriasis", "Lyme disease", "Viral infection", "Contact dermatitis"],
    "weakness": ["Anemia", "Hypothyroidism", "Chronic fatigue", "Diabetes", "Neurological disorder"],
    "numbness": ["Diabetes", "Multiple sclerosis", "Carpal tunnel", "Stroke", "Peripheral neuropathy"],
    "vomiting": ["Gastroenteritis", "Food poisoning", "Migraine", "Pregnancy", "Appendicitis"],
    "diarrhea": ["Gastroenteritis", "IBS", "Food intolerance", "Inflammatory bowel disease", "Infection"],
    "insomnia": ["Anxiety", "Depression", "Sleep apnea", "Chronic pain", "Restless legs syndrome"],
    "weight loss": ["Hyperthyroidism", "Diabetes", "Cancer", "Depression", "Malabsorption", "Chronic disease"],
    "swelling": ["Edema", "Allergic reaction", "Heart failure", "Kidney disease", "Lymphatic disorder"],
    "pain": ["Various conditions – see specific body area"],
    "stomach pain": ["Gastritis", "GERD", "Appendicitis", "IBS", "Food poisoning"],
    "body ache": ["Influenza", "COVID-19", "Fibromyalgia", "Viral infection", "Chronic fatigue"],
    "loss of appetite": ["Depression", "Infection", "Liver disease", "Cancer", "Hypothyroidism"],
    "high blood pressure": ["Hypertension", "Kidney disease", "Thyroid disorder", "Anxiety"],
    "coughing": ["Bronchitis", "Pneumonia", "Asthma", "Common cold", "COVID-19"],
    "sneezing": ["Allergic rhinitis", "Common cold", "Influenza", "COVID-19"],
}

# Build a unique candidate disease list for neural ranking
_all_diseases = []
for _sym, _ds in SYMPTOM_TO_DISEASES.items():
    _all_diseases.extend(_ds)
CANDIDATE_DISEASES = sorted(set(_all_diseases))

# Zero-shot classifier for ranking diseases by relevance to the symptom text
zero_shot_classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)


def _normalize(s: str):
    return s.strip().lower().replace(",", " ").replace(".", " ")


def suggest_diseases_from_symptoms(text: str):
    """Map symptom phrases to possible diseases; return unique disease list (does not echo symptoms)."""
    if not text or not text.strip():
        return [], []
    normalized = _normalize(text)
    # Split into tokens (words)
    words = [w for w in normalized.split() if len(w) > 1]
    seen_diseases = set()
    symptoms_matched = []
    # Check each word and multi-word phrases (2-3 words)
    for i, w in enumerate(words):
        if w in SYMPTOM_TO_DISEASES:
            symptoms_matched.append(w)
            for d in SYMPTOM_TO_DISEASES[w]:
                seen_diseases.add(d)
        if i + 1 < len(words):
            two = w + " " + words[i + 1]
            if two in SYMPTOM_TO_DISEASES:
                symptoms_matched.append(two)
                for d in SYMPTOM_TO_DISEASES[two]:
                    seen_diseases.add(d)
        if i + 2 < len(words):
            three = w + " " + words[i + 1] + " " + words[i + 2]
            if three in SYMPTOM_TO_DISEASES:
                symptoms_matched.append(three)
                for d in SYMPTOM_TO_DISEASES[three]:
                    seen_diseases.add(d)
    return list(seen_diseases), list(dict.fromkeys(symptoms_matched))


def rank_diseases_with_nn(text: str, candidate_diseases: list[str] | None = None):
    """
    Use a neural zero-shot classifier to rank diseases with confidence scores.
    """
    if not text or not text.strip():
        return []

    labels = candidate_diseases if candidate_diseases else CANDIDATE_DISEASES
    if not labels:
        return []

    result = zero_shot_classifier(text, labels, multi_label=True)
    labels_out = result.get("labels", [])
    scores_out = result.get("scores", [])

    ranked = []
    for name, score in zip(labels_out, scores_out):
        ranked.append(
            {
                "name": name,
                "confidence": float(score),
            }
        )

    # Sort by confidence descending
    ranked.sort(key=lambda x: x["confidence"], reverse=True)
    return ranked


print("Serving API now...")

def predict_sentence(sentence):
    t_sen = tokenizer.tokenize(sentence)

    sen_code = tokenizer.encode_plus(sentence,
        return_tensors='pt',
        add_special_tokens=True,
        max_length = max_length,
        padding='max_length',
        return_attention_mask=True,
        truncation = True
        )
    inputs = {key: torch.as_tensor(val) for key, val in sen_code.items()}

    attention_mask = inputs['attention_mask'].squeeze(1)
    input_ids = inputs['input_ids'].squeeze(1)

    outputs = model(input_ids, attention_mask)

    predictions = outputs.logits.argmax(dim=-1)
    predictions = [ids_to_label.get(x) for x in predictions.numpy()[0]]

    #beware special tokens
    cutoff = min(len(predictions)-1, len(t_sen))
    predictions = predictions[1:cutoff+1]
    t_sen = t_sen[:cutoff]

    return t_sen, predictions

def clean(tokens, labels):
    cleaned_tokens = []
    cleaned_labels = []
    cnt = 1

    for i in range(len(tokens)):  # same length
        if tokens[i].startswith("##") and len(cleaned_tokens) > 0:
            cleaned_tokens[i - cnt] = cleaned_tokens[i - cnt] + tokens[i][2:]
            cnt = cnt + 1
        else:
            cleaned_tokens.append(tokens[i])
            cleaned_labels.append(labels[i])

    return cleaned_tokens, cleaned_labels


def build_entities(tokens, labels):
    """
    Group B-/I- label sequences into entity spans so the API
    can return a structured list of detected entities.
    """
    entities = []
    i = 0

    while i < len(tokens):
        label = labels[i]

        if label is None:
            i += 1
            continue

        label_str = str(label)

        if label_str == 'O':
            i += 1
            continue

        if not label_str.startswith('B-'):
            i += 1
            continue

        entity_type = label_str[2:]
        start_index = i
        span_tokens = [tokens[i]]
        i += 1

        while i < len(tokens) and str(labels[i]).startswith('I-'):
            span_tokens.append(tokens[i])
            i += 1

        entity_text = " ".join(span_tokens)

        entities.append(
            {
                "text": entity_text,
                "type": entity_type,
                "start": start_index,
                "end": i - 1,
            }
        )

    return entities


def handle_request(data):
    sentences = sentence_detector(data).sents

    tokens = []
    labels = []

    for sentence in sentences:
        new_tokens, new_labels = predict_sentence(sentence.text)
        tokens = tokens + new_tokens
        labels = labels + new_labels

    cleaned_tokens, cleaned_labels = clean(tokens, labels)

    return cleaned_tokens, cleaned_labels


@app.route('/health', methods=['GET'])
def health():
    return jsonify(
        {
            "status": "ok",
            "type": args.type,
            "model_path": model_path,
            "model_loaded": model_loaded,
        }
    )


@app.route('/extract_entities', methods=['POST'])
def main():
    text = request.get_data(as_text=True)
    result = handle_request(text)
    return jsonify({'tokens': result[0], 'entities': result[1]})


@app.route('/extract_entities_structured', methods=['POST'])
def extract_entities_structured():
    """
    Extended endpoint that, in addition to the token/label
    sequence, returns grouped entity spans that are easier
    to consume from frontend clients.
    """
    text = request.get_data(as_text=True)
    tokens, labels = handle_request(text)
    spans = build_entities(tokens, labels)
    return jsonify({'tokens': tokens, 'entities': labels, 'spans': spans})


@app.route('/suggest_diseases', methods=['POST'])
def suggest_diseases():
    """
    Accept symptom text from the patient; return possible diseases (not symptoms).
    Uses a symptom→disease mapping so the result is conditions to consider, not echoed symptoms.
    """
    text = request.get_data(as_text=True)
    possible_diseases, symptoms_matched = suggest_diseases_from_symptoms(text)

    # If heuristic mapping found diseases, rank those; otherwise rank over all known diseases
    candidate_list = possible_diseases if possible_diseases else CANDIDATE_DISEASES
    ranked = rank_diseases_with_nn(text, candidate_list)

    return jsonify({
        'possible_diseases': possible_diseases,
        'symptoms_considered': symptoms_matched,
        'ranked_diseases': ranked,
    })


if __name__ == '__main__':
    LISTEN = ('0.0.0.0',port)
    http_server = WSGIServer( LISTEN, app )
    http_server.serve_forever()
