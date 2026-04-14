# Trained model weights

Place your trained `.pth` model file here (e.g. `medcondbert.pth`) so the API can load it.

## Use a trained model **right now** (no training needed)

You can use the **BioBERT disease NER** model from Hugging Face (already trained for medical/disease NER). From the `src` folder:

```bash
python api.py -t "Medical Condition" -tr
```

No `.pth` file is required; the app uses the pretrained model. First run may download the model from the internet.

---

## Training your own model (optional)

From the `src` folder:

```bash
python main.py -t "Medical Condition" -o ../models/medcondbert.pth -e 5 -b 16
```

Optional: use `-tr True` for BioBERT-based training (Medical Condition only).

After training, start the API with:

```bash
python api.py -t "Medical Condition" -m ../models/medcondbert.pth
```

If no model file is present and you do **not** use `-tr True`, the API starts in **demo mode** (untrained base BERT); predictions will be poor until you add a trained model or use the BioBERT option above.
