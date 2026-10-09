import os
_n = os.environ.get('TOOL_SEARCH_NUM_THREADS', '32')
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, _n)
import asyncio
from contextlib import suppress
import json
import numpy as np
import faiss
import time
import os
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
import uvicorn
import torch
torch.set_num_threads(int(_n))
faiss.omp_set_num_threads(int(_n))
import transformers.utils.import_utils as _transformers_import_utils
if not hasattr(_transformers_import_utils, 'is_torch_fx_available'):
    _transformers_import_utils.is_torch_fx_available = lambda: True
from FlagEmbedding import FlagAutoModel
from paraagent.toolenv.retrieval_resources import file_sha256, load_corpus_snapshot
import re
app = FastAPI(title='Tool Search API', description='API for Tool searching using FAISS index')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])
model = None
index = None
corpus = None
DEFAULT_TOP_K = int(os.environ.get('TOOL_SEARCH_DEFAULT_TOP_K', '3'))
if DEFAULT_TOP_K < 1:
    raise ValueError('TOOL_SEARCH_DEFAULT_TOP_K must be positive')

class SearchQuery(BaseModel):
    queries: List[str] = Field(min_length=1)
    top_k: int = Field(default_factory=lambda: DEFAULT_TOP_K, ge=1, strict=True)

    @field_validator('queries')
    @classmethod
    def nonempty_queries(cls, queries):
        if any(not query.strip() for query in queries):
            raise ValueError('Queries must not be blank')
        return queries

class SearchResult(BaseModel):
    score: float
    tools: Dict[str, Any]

class QueryResult(BaseModel):
    query: str
    results: List[SearchResult]

class SearchResponse(BaseModel):
    query_results: List[QueryResult]
    total_time: float
    search_time: float

def standardize_category(category):
    save_category = category.replace(' ', '_').replace(',', '_').replace('/', '_')
    while ' ' in save_category or ',' in save_category:
        save_category = save_category.replace(' ', '_').replace(',', '_')
    save_category = save_category.replace('__', '_')
    return save_category

def standardize(string):
    res = re.compile('[^\\u4e00-\\u9fa5^a-z^A-Z^0-9^_]')
    string = res.sub('_', string)
    string = re.sub('(_)\\1+', '_', string).lower()
    while True:
        if len(string) == 0:
            return string
        if string[0] == '_':
            string = string[1:]
        else:
            break
    while True:
        if len(string) == 0:
            return string
        if string[-1] == '_':
            string = string[:-1]
        else:
            break
    if string[0].isdigit():
        string = 'get_' + string
    return string

def change_name(name):
    change_list = ['from', 'class', 'return', 'false', 'true', 'id', 'and']
    if name in change_list:
        name = 'is_' + name
    return name

def load_index(index_path):
    """Load a FAISS index from disk"""
    print(f'Loading index from {index_path}...')
    meta_path = f'{index_path}.meta'
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        if 'index_sha256' in meta and file_sha256(index_path) != meta['index_sha256']:
            raise ValueError('Index checksum does not match its metadata; deploy matching files')
    index = faiss.read_index(index_path)
    if 'nprobe' in meta and hasattr(index, 'nprobe'):
        index.nprobe = meta['nprobe']
        print(f"Setting nprobe to {meta['nprobe']}")
    return index

def load_corpus(dataset_name=''):
    """Load the corpus from dataset"""
    print(f'Loading corpus from {dataset_name}...')
    return load_corpus_snapshot(dataset_name)[0]

def load_resources(index_path, corpus_path):
    loaded_index = load_index(index_path)
    documents, corpus_hash = load_corpus_snapshot(corpus_path)
    if loaded_index.ntotal != len(documents):
        raise ValueError(f'Index/corpus row mismatch: {loaded_index.ntotal} != {len(documents)}')
    meta_path = f'{index_path}.meta'
    if os.path.exists(meta_path):
        with open(meta_path) as handle:
            meta = json.load(handle)
        if 'corpus_sha256' in meta and meta['corpus_sha256'] != corpus_hash:
            raise ValueError('Corpus checksum does not match the index; rebuild after changing rows or order')
        if 'embedding_shape' in meta and meta['embedding_shape'] != [loaded_index.ntotal, loaded_index.d]:
            raise ValueError('Index shape does not match its metadata')
    return loaded_index, documents

def validate_request(queries, top_k):
    if not queries or any(not isinstance(query, str) or not query.strip() for query in queries):
        raise HTTPException(status_code=400, detail='Provide at least one nonblank query')
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise HTTPException(status_code=400, detail='top_k must be a positive integer')

def _encode_and_search(queries: List[str], top_k: int):
    """One CPU-bound encode+faiss pass over a (possibly merged) query batch."""
    start_time = time.time()
    validate_request(queries, top_k)
    query_embeddings = model.encode_queries(queries)
    query_embeddings = np.array(query_embeddings, dtype=np.float32)
    if query_embeddings.shape != (len(queries), index.d):
        raise ValueError('Query embedding shape does not match the index; use the same embedding model')
    if not np.isfinite(query_embeddings).all() or np.any(np.linalg.norm(query_embeddings, axis=1) == 0):
        raise ValueError('Query embeddings contain non-finite or zero vectors')
    faiss.normalize_L2(query_embeddings)
    search_start = time.time()
    scores, indices = index.search(query_embeddings, min(top_k, index.ntotal))
    search_end = time.time()
    return (scores, indices, start_time, search_start, search_end)

def _build_response(queries, scores, indices, top_k, start_time, search_start, search_end):
    query_results = []
    for q_idx, query in enumerate(queries):
        results = []
        for i, idx in enumerate(indices[q_idx][:top_k]):
            if idx != -1:
                results.append(SearchResult(score=float(scores[q_idx][i]), tools=corpus[int(idx)]))
        query_results.append(QueryResult(query=query, results=results))
    return SearchResponse(query_results=query_results, total_time=time.time() - start_time, search_time=search_end - search_start)

def search(queries: List[str], top_k=None):
    """Search one request when TOOL_SEARCH_BATCH_ENABLE=0."""
    global model, index, corpus
    if top_k is None:
        top_k = DEFAULT_TOP_K
    if model is None or index is None or corpus is None:
        raise HTTPException(status_code=500, detail='Search engine not initialized')
    scores, indices, t0, t1, t2 = _encode_and_search(queries, top_k)
    return _build_response(queries, scores, indices, top_k, t0, t1, t2)
_BATCH_ENABLE = os.environ.get('TOOL_SEARCH_BATCH_ENABLE', '1') == '1'
_BATCH_WINDOW_S = float(os.environ.get('TOOL_SEARCH_BATCH_WINDOW_MS', '10')) / 1000.0
_BATCH_MAX = int(os.environ.get('TOOL_SEARCH_MAX_BATCH', '64'))
_batch_queue: Optional['asyncio.Queue'] = None
_batch_task: Optional['asyncio.Task'] = None

async def _batcher_loop():
    """Process one encode/search batch at a time.
    Encoding runs in a worker thread while incoming requests queue for the next batch.
    """
    loop = asyncio.get_running_loop()
    while True:
        items = [await _batch_queue.get()]
        deadline = loop.time() + _BATCH_WINDOW_S
        n_queries = len(items[0][0])
        while n_queries < _BATCH_MAX:
            timeout = deadline - loop.time()
            if timeout <= 0:
                break
            try:
                item = await asyncio.wait_for(_batch_queue.get(), timeout)
            except asyncio.TimeoutError:
                break
            items.append(item)
            n_queries += len(item[0])
        merged: List[str] = []
        for queries, _top_k, _fut in items:
            merged.extend(queries)
        max_k = max((top_k for _q, top_k, _f in items))
        try:
            scores, indices, t0, t1, t2 = await loop.run_in_executor(None, _encode_and_search, merged, max_k)
            offset = 0
            for queries, top_k, fut in items:
                sl = slice(offset, offset + len(queries))
                offset += len(queries)
                if not fut.done():
                    fut.set_result(_build_response(queries, scores[sl], indices[sl], top_k, t0, t1, t2))
        except Exception as exc:
            for _q, _k, fut in items:
                if not fut.done():
                    fut.set_exception(exc)

async def search_batched(queries: List[str], top_k: Optional[int]=None) -> SearchResponse:
    if top_k is None:
        top_k = DEFAULT_TOP_K
    validate_request(queries, top_k)
    if model is None or index is None or corpus is None:
        raise HTTPException(status_code=500, detail='Search engine not initialized')
    if not (_BATCH_ENABLE and _batch_queue is not None):
        return search(queries, top_k)
    fut = asyncio.get_running_loop().create_future()
    await _batch_queue.put((list(queries), int(top_k), fut))
    return await fut

@app.on_event('startup')
async def startup_event():
    """Initialize the search engine on startup"""
    global model, index, corpus
    index_path = os.environ.get('INDEX_PATH', '')
    model_path = os.environ.get('Retriever_Path', '')
    corpus_path = os.environ.get('corpus', '')
    loaded_index, documents = load_resources(index_path, corpus_path)
    print('Loading model...')
    model = FlagAutoModel.from_finetuned(model_path, query_instruction_for_retrieval='Represent this sentence for searching relevant passages: ', devices=os.environ.get('TOOL_SEARCH_DEVICE', 'cpu'))
    index, corpus = loaded_index, documents
    global _batch_queue, _batch_task
    if _BATCH_ENABLE:
        _batch_queue = asyncio.Queue()
        _batch_task = asyncio.get_running_loop().create_task(_batcher_loop())
        print(f'Micro-batching enabled: window={_BATCH_WINDOW_S * 1000:.0f}ms max_batch={_BATCH_MAX}')
    print('Search engine initialized successfully')

@app.on_event('shutdown')
async def shutdown_event():
    global _batch_queue, _batch_task, model, index, corpus
    if _batch_task is not None:
        _batch_task.cancel()
        with suppress(asyncio.CancelledError):
            await _batch_task
    _batch_task = _batch_queue = None
    model = index = corpus = None

@app.post('/search', response_model=SearchResponse)
async def api_search(search_query: SearchQuery):
    """Search the corpus for the most similar documents to the queries"""
    return await search_batched(search_query.queries, search_query.top_k)

@app.get('/search', response_model=SearchResponse)
async def api_search_get(query: str=Query(..., min_length=1, description='The query to search for (for multiple queries, use POST method)'), top_k: Optional[int]=Query(None, ge=1, description='Number of results; defaults to the configured --top-k (3).')):
    """Search the corpus for the most similar documents to the query (GET method)"""
    return await search_batched([query], top_k)

@app.get('/health')
async def health_check():
    """Health check endpoint"""
    if model is None or index is None or corpus is None:
        raise HTTPException(status_code=503, detail='Search engine not fully initialized')
    return {'status': 'healthy', 'index_path': os.environ.get('INDEX_PATH', ''), 'corpus_path': os.environ.get('corpus', ''), 'corpus_rows': len(corpus)}

@app.get('/')
async def root():
    """Root endpoint"""
    return {'message': 'Tool Search API', 'docs': '/docs', 'health': '/health'}

def main():
    global DEFAULT_TOP_K
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description="ToolEnv and benchmark BGE/FAISS retrieval service")
    parser.add_argument('--catalog', choices=['toolenv', 'toolbench', 'apibank'], required=True)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--model', default='BAAI/bge-large-en-v1.5')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int)
    parser.add_argument('--top-k', type=int, default=DEFAULT_TOP_K, help='Default retrieval count (env: TOOL_SEARCH_DEFAULT_TOP_K; default: 3).')
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error('--top-k must be positive')
    DEFAULT_TOP_K = args.top_k
    resources = {
        'toolenv': ('data/toolenv', 'toolcorpus_all.tsv', 'tool-corpus_all_index_HNSW64.bin', 30400),
        'toolbench': ('data/benchmarks/toolbench', 'corpus.tsv', 'index.bin', 30401),
        'apibank': ('data/benchmarks/apibank', 'corpus.json', 'index.bin', 30403),
    }
    directory, corpus_file, index_file, port = resources[args.catalog]
    base = args.root.resolve() / directory
    for name in (corpus_file, index_file):
        if not (base / name).is_file():
            parser.error(f'Missing resource: {base / name}')
    os.environ['corpus'] = str(base / corpus_file)
    os.environ['INDEX_PATH'] = str(base / index_file)
    os.environ['Retriever_Path'] = args.model
    os.environ['TOOL_SEARCH_DEVICE'] = args.device
    uvicorn.run(app, host=args.host, port=args.port or port)

if __name__ == '__main__':
    main()
