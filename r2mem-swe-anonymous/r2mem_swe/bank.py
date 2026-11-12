"""Local-only sentence embeddings; normalized cosine top-k, two separate banks."""
from pathlib import Path
from .core import digest, read, save


def model_signature(path):
    from hashlib import sha256
    root = Path(path)
    configs = {str(p.relative_to(root)): sha256(p.read_bytes()).hexdigest()
               for p in sorted(root.rglob('*.json')) if p.is_file()}
    if not configs:
        raise ValueError('No model configuration/tokenizer JSON found')
    return digest(configs)


def encoder(path):
    if not Path(path).is_dir():
        raise ValueError('Embedding model must be an existing local directory')
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(path, device='cpu', local_files_only=True)


def create_index(bank_path, model_path, output):
    import numpy as np
    bank = read(bank_path)
    if not bank['experiences']:
        raise ValueError('Empty experience bank')
    if Path(output).exists():
        raise FileExistsError(output)
    model = encoder(model_path)
    rows = bank['experiences']
    vectors = model.encode([r['situation'] for r in rows], normalize_embeddings=True)
    vectors = np.asarray(vectors, dtype='float32')
    if vectors.ndim != 2 or not np.isfinite(vectors).all() or (np.linalg.norm(vectors, axis=1) < .99).any():
        raise ValueError('Invalid embedding vectors')
    directory = Path(output)
    directory.mkdir(parents=True)
    np.save(directory/'vectors.npy', vectors, allow_pickle=False)
    save(directory/'bank.json', bank)
    from hashlib import sha256
    save(directory/'index.json', {'bank_sha': digest(bank), 'model_path': str(Path(model_path).resolve()),
                                 'dimension': vectors.shape[1], 'model_signature': model_signature(model_path),
                                 'vectors_sha': sha256((directory/'vectors.npy').read_bytes()).hexdigest()})


class Bank:
    def __init__(self, directory):
        import numpy as np
        from hashlib import sha256
        self.np = np
        self.directory = Path(directory)
        self.meta = read(self.directory/'index.json')
        self.data = read(self.directory/'bank.json')
        if digest(self.data) != self.meta['bank_sha']:
            raise ValueError('Bank changed after indexing')
        if sha256((self.directory/'vectors.npy').read_bytes()).hexdigest() != self.meta['vectors_sha']:
            raise ValueError('Embedding index checksum mismatch')
        self.vectors = np.load(self.directory/'vectors.npy', allow_pickle=False)
        if self.vectors.shape != (len(self.data['experiences']), self.meta['dimension']):
            raise ValueError('Index shape mismatch')
        if model_signature(self.meta['model_path']) != self.meta['model_signature']:
            raise ValueError('Embedding model configuration changed; rebuild index')
        self.model = encoder(self.meta['model_path'])

    def search(self, module, query, k=3):
        if k < 1 or k > 10:
            raise ValueError('Require 1 <= return_k <= recall_k=10')
        indices = [i for i, r in enumerate(self.data['experiences']) if r['module'] == module]
        if not indices:
            return []
        vector = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.vectors[indices] @ vector
        order = self.np.argsort(-scores, kind='stable')[:10][:k]
        return [dict(self.data['experiences'][indices[int(j)]], similarity=float(scores[j])) for j in order]
