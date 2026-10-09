"""Read the same ordered tool corpus for index construction and serving."""

import csv
import hashlib
import io
import json
from pathlib import Path
import sys


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_corpus_snapshot(path):
    """Return documents and the checksum of the exact bytes parsed, in row order."""
    path = Path(path)
    raw = path.read_bytes()
    if path.suffix == ".json":
        documents = json.loads(raw)
    elif path.suffix == ".tsv":
        csv.field_size_limit(sys.maxsize)
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8"), newline=""), delimiter="\t")
        if "document_content" not in (reader.fieldnames or []):
            raise ValueError("TSV corpus requires a document_content column")
        documents = [json.loads(row["document_content"]) for row in reader]
    else:
        raise ValueError("Corpus must be a .tsv or .json file")
    if not isinstance(documents, list) or not documents:
        raise ValueError("Corpus must contain a nonempty list of tool documents")
    for row, document in enumerate(documents):
        if not isinstance(document, dict) or not isinstance(document.get("name"), str) or not document["name"].strip():
            raise ValueError(f"Corpus row {row} requires a nonempty tool name")
    return documents, hashlib.sha256(raw).hexdigest()


def load_documents(path):
    return load_corpus_snapshot(path)[0]
