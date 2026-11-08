import hashlib, json, math, queue, re, threading
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker

TOKEN_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)

# Same lexical scoring as agent-memory-temp/m5/answer.py.
def word_tokens(text: object) -> list[str]:
    return [value.lower() for value in TOKEN_RE.findall(str(text or ""))]


class BM25:
    def __init__(self, documents, k1=1.5, b=0.75):
        self.documents = [word_tokens(value) for value in documents]
        self.k1, self.b = k1, b
        self.avgdl = sum(map(len, self.documents)) / max(len(self.documents), 1)
        self.postings = defaultdict(list)
        self.norms = np.zeros(len(self.documents), dtype=np.float32)
        df = Counter()
        for index, document in enumerate(self.documents):
            frequencies = Counter(document)
            df.update(frequencies)
            for term, frequency in frequencies.items():
                self.postings[term].append((index, frequency))
            self.norms[index] = k1 * (1 - b + b * len(document) / max(self.avgdl, 1e-6))
        count = len(self.documents)
        self.idf = {
            term: math.log(1 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in df.items()
        }

    def scores(self, query):
        result = np.zeros(len(self.documents), dtype=np.float32)
        for term, query_frequency in Counter(word_tokens(query)).items():
            for index, frequency in self.postings.get(term, []):
                result[index] += (
                    query_frequency * self.idf.get(term, 0.0) * frequency * (self.k1 + 1)
                    / (frequency + self.norms[index])
                )
        return result



class BGERetriever:
    def __init__(self, args):
        self.args = args
        devices = [d.strip() for d in (args.devices or args.device).split(',') if d.strip()]
        if not devices or len(set(devices)) != len(devices):
            raise ValueError('Specify a nonempty list of unique local devices')
        self.encoders = []
        self.rerankers = []
        for device in devices:
            encoder = SentenceTransformer(args.embedding_model_path, device=device,
                                          local_files_only=True)
            encoder.max_seq_length = 8192
            self.encoders.append(encoder)
            self.rerankers.append(FlagReranker(args.reranker_model, devices=[device],
                                             use_fp16=device != 'cpu'))
        self.pool = ThreadPoolExecutor(max_workers=len(devices))
        self.slots = queue.Queue()
        for index in range(len(devices)):
            self.slots.put(index)
        self.lock = threading.Lock()
        self.indexes = {}
        Path(args.index_dir).mkdir(parents=True, exist_ok=True)
        self.traces = []

    def encode_documents(self, texts):
        chunks = [list(chunk) for chunk in np.array_split(np.asarray(texts, dtype=object),
                                                        len(self.encoders))]
        futures = [self.pool.submit(model.encode_document, chunk,
                    normalize_embeddings=True, batch_size=self.args.batch_size,
                    convert_to_numpy=True)
                   for model, chunk in zip(self.encoders, chunks) if chunk]
        return np.concatenate([future.result() for future in futures]).astype(np.float32)

    def score_pairs(self, pairs):
        boundaries = np.linspace(0, len(pairs), len(self.rerankers) + 1, dtype=int)
        futures = [self.pool.submit(model.compute_score, pairs[start:end], normalize=True,
                    batch_size=self.args.batch_size, max_length=self.args.reranker_max_length)
                   for model, start, end in zip(self.rerankers, boundaries[:-1], boundaries[1:])
                   if end > start]
        return np.concatenate([np.asarray(future.result()).reshape(-1) for future in futures])

    def close(self):
        self.pool.shutdown(wait=True)

    def retrieve(self, entries, query_text, limit=20):
        if not entries or limit <= 0:
            return []
        # Never index original_memory/compressed_memory: those are not the
        # baseline's answer evidence. Summary has its own payload schema.
        texts = [e['payload'].get('summary', e['payload'].get('memory', '')) for e in entries]
        signature = hashlib.sha256(json.dumps({
            'model': self.args.embedding_model_path, 'max_length': 8192,
            'normalize': True, 'encoder': 'encode_document',
            'items': [(str(e['id']), t) for e, t in zip(entries, texts)],
        }, sort_keys=True).encode()).hexdigest()
        with self.lock:  # only index creation/loading is serialized
            if signature not in self.indexes:
                path = Path(self.args.index_dir) / (signature + '.npy')
                if path.exists():
                    vectors = np.load(path, allow_pickle=False)
                else:
                    held = [self.slots.get() for _ in self.encoders]
                    try:
                        vectors = self.encode_documents(texts)
                    finally:
                        for index in held:
                            self.slots.put(index)
                    np.save(path, vectors)
                if vectors.shape[0] != len(entries):
                    raise ValueError('Index entry count mismatch')
                self.indexes[signature] = vectors
            vectors = self.indexes[signature]
        # Index construction is shared; each query then leases an independent
        # model replica. The slot is returned before the caller requests its LLM.
        slot = self.slots.get()
        try:
            query = self.encoders[slot].encode_query([query_text], normalize_embeddings=True,
                                              convert_to_numpy=True)[0]
            dense_scores = vectors @ query
            dense_order = np.argsort(-dense_scores, kind='stable')
            order = dense_order
            if self.args.retrieval_backend == 'hybrid':
                bm25 = BM25(texts)
                lexical = np.argsort(-bm25.scores(query_text), kind='stable')
                fused = np.zeros(len(entries))
                for ranking in (dense_order, lexical):
                    fused[ranking] += 1.0 / (self.args.rrf_constant + np.arange(1, len(entries) + 1))
                order = np.argsort(-fused, kind='stable')
            candidates = order[:self.args.candidate_limit]
            scores = np.asarray(self.rerankers[slot].compute_score(
                [[query_text, texts[i]] for i in candidates], normalize=True,
                batch_size=self.args.batch_size,
                max_length=self.args.reranker_max_length)).reshape(-1)
            selected = np.argsort(-scores, kind='stable')[:limit]
            result = []
            for rank in selected:
                i = int(candidates[rank])
                result.append(dict(entries[i], score=float(scores[rank]),
                                   embedding_score=float(dense_scores[i]), source='bge_rerank'))
            self.traces.append(dict(question=query_text, index=signature,
                candidates=[dict(id=str(entries[int(i)]['id']), score=float(score))
                            for i, score in zip(candidates, scores)],
                selected_ids=[str(e['id']) for e in result]))
            return result
        finally:
            self.slots.put(slot)
