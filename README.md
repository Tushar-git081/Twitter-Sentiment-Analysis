# Twitter Sentiment Analysis

Streamlit app for predicting tweet sentiment with the saved TensorFlow model and tokenizer.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Xquik Export Uploads

The app accepts Xquik CSV, JSON, and JSONL exports. Upload an export, choose an
imported tweet, and run it through the existing model.

Supported text fields include `text`, `tweet`, `tweet_text`, `full_text`,
`content`, and nested tweet objects.

## Parser Tests

```bash
python3 -m unittest test_xquik_import.py
```
