import csv
import io
import json


class XquikImportError(ValueError):
    pass


TEXT_KEYS = (
    "text",
    "tweet",
    "tweet_text",
    "full_text",
    "content",
    "body",
)


def load_xquik_texts(uploaded_file):
    name = getattr(uploaded_file, "name", "").lower()
    raw = uploaded_file.read()
    if isinstance(raw, str):
        raw_text = raw
    else:
        raw_text = raw.decode("utf-8-sig")

    if name.endswith(".csv"):
        rows = csv.DictReader(io.StringIO(raw_text))
        records = list(rows)
    elif name.endswith(".jsonl"):
        records = [json.loads(line) for line in raw_text.splitlines() if line.strip()]
    elif name.endswith(".json"):
        records = _records_from_json(json.loads(raw_text))
    else:
        raise XquikImportError("Upload a CSV, JSON, or JSONL export.")

    texts = [_extract_text(record) for record in records]
    texts = [text for text in texts if text]
    if not texts:
        raise XquikImportError("No tweet text field was found in the export.")
    return texts


def _records_from_json(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("results", "tweets", "data", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        return [payload]
    raise XquikImportError("JSON export must contain an object or array.")


def _extract_text(record):
    if isinstance(record, str):
        return record.strip()
    if not isinstance(record, dict):
        return ""

    for key in TEXT_KEYS:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    for key in ("tweet", "post", "result", "data"):
        value = record.get(key)
        if isinstance(value, dict):
            text = _extract_text(value)
            if text:
                return text

    return ""
