#!/usr/bin/env python3
"""Encode a released tool corpus with BGE and build its FAISS retrieval index."""

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from paraagent.toolenv.runtime.retrieval_text import document_to_ir_text
from paraagent.toolenv.retrieval_resources import file_sha256, load_corpus_snapshot, load_documents


CATALOGS = {
    "toolenv": ("data/toolenv/toolcorpus_all.tsv", "tool-corpus_all_index_HNSW64.bin", "HNSW64"),
}
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def retrieval_text(document, catalog):
    """Use the ToolEnv embedding text format."""
    if catalog != "toolenv":
        raise ValueError(f"Unknown catalog: {catalog}")
    return document_to_ir_text(document)


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", choices=CATALOGS, required=True)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--corpus", type=Path, help="Custom TSV/JSON corpus; requires --output-index.")
    parser.add_argument("--output-index", type=Path, help="Default: the selected catalog's serving index.")
    parser.add_argument("--embeddings", type=Path, help="Optionally save normalized float32 vectors as .npy.")
    parser.add_argument("--model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--device", default="cpu", help="cpu or cuda:N (relative to CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--batch-size", type=positive_int, help="Default: 64.")
    parser.add_argument("--max-length", type=positive_int, default=512)
    parser.add_argument("--threads", type=positive_int, default=8)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing index/metadata/embedding outputs.")
    args = parser.parse_args()
    args.batch_size = args.batch_size or 64
    if args.corpus and not args.output_index:
        parser.error("--corpus requires --output-index to keep the default serving pair intact")
    if args.device != "cpu" and not (args.device.startswith("cuda:") and args.device[5:].isdigit()):
        parser.error("--device must be cpu or cuda:N")
    relative, index_name, index_type = CATALOGS[args.catalog]
    corpus = args.corpus or args.root / relative
    output = (args.output_index or corpus.with_name(index_name)).resolve()
    corpus = corpus.resolve()
    metadata_path = Path(str(output) + ".meta")
    outputs = [output, metadata_path]
    if args.embeddings:
        args.embeddings = args.embeddings.resolve()
        outputs.append(args.embeddings)
    if len(set(outputs)) != len(outputs) or corpus in outputs:
        parser.error("Output paths must be distinct and must not replace the corpus")
    for path in outputs:
        if path.exists() and (not args.overwrite or not path.is_file()):
            parser.error(f"Output already exists: {path}; use --overwrite to replace files")

    documents, corpus_hash = load_corpus_snapshot(corpus)
    texts = [retrieval_text(document, args.catalog) for document in documents]
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(name, str(args.threads))
    import faiss
    import numpy as np
    import torch
    import transformers.utils.import_utils as transformers_import_utils
    # Provide the Transformers capability check required by FlagEmbedding.
    if not hasattr(transformers_import_utils, "is_torch_fx_available"):
        transformers_import_utils.is_torch_fx_available = lambda: True
    from FlagEmbedding import FlagAutoModel

    torch.set_num_threads(args.threads)
    faiss.omp_set_num_threads(args.threads)
    print(f"Encoding {len(texts)} {args.catalog} tools on {args.device}", flush=True)
    model = FlagAutoModel.from_finetuned(
        args.model,
        query_instruction_for_retrieval=QUERY_INSTRUCTION,
        devices=args.device,
        use_fp16=args.device != "cpu",
    )
    embeddings = np.ascontiguousarray(
        model.encode_corpus(texts, batch_size=args.batch_size, max_length=args.max_length),
        dtype=np.float32,
    )
    if embeddings.ndim != 2 or embeddings.shape[0] != len(documents) or embeddings.shape[1] == 0:
        raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")
    if not np.isfinite(embeddings).all() or np.any(np.linalg.norm(embeddings, axis=1) == 0):
        raise ValueError("Embeddings contain non-finite or zero vectors")
    faiss.normalize_L2(embeddings)
    index = faiss.index_factory(embeddings.shape[1], index_type, faiss.METRIC_INNER_PRODUCT)
    if index_type == "HNSW64":
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 128
    print(f"Building {index_type} on CPU", flush=True)
    index.add(embeddings)
    metadata = {
        "catalog": args.catalog,
        "model": Path(args.model).name if Path(args.model).is_dir() else args.model,
        "corpus_sha256": corpus_hash,
        "text_sha256": hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest(),
        "text_format": f"{args.catalog}-v1",
        "embedding_build": "full_encode",
        "embedding_shape": list(embeddings.shape),
        "rows": len(documents),
        "normalized": True,
        "index_type": index_type,
        "metric": "inner_product",
        "encoding_config": {
            "batch_size": args.batch_size, "max_length": args.max_length,
            "device": args.device, "use_fp16": args.device != "cpu",
            "query_instruction": QUERY_INSTRUCTION,
        },
        "versions": {name: version(name) for name in ("FlagEmbedding", "transformers", "torch", "faiss-cpu")},
    }
    if index_type == "HNSW64":
        metadata.update(hnsw_ef_construction=200, hnsw_ef_search=128)
    if file_sha256(corpus) != corpus_hash:
        raise ValueError("Corpus changed during encoding; rebuild from a stable file")
    # Stage all outputs before replacing existing files; partial builds stay temporary.
    staged = []
    try:
        for path in outputs:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
                temporary = Path(handle.name)
            staged.append((temporary, path))
            if path == output:
                faiss.write_index(index, str(temporary))
                metadata["index_sha256"] = file_sha256(temporary)
            elif path == metadata_path:
                temporary.write_text(json.dumps(metadata, indent=2) + "\n")
            else:
                with temporary.open("wb") as handle:
                    np.save(handle, embeddings, allow_pickle=False)
        for temporary, path in staged:
            os.replace(temporary, path)
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
    print(f"Saved {index.ntotal} vectors ({index.d} dimensions) to {output}")


if __name__ == "__main__":
    main()
