import os
import json
import random
import re
import time
from collections import defaultdict
from typing import Dict, List, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import font_manager

from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers import models as st_models
from sentence_transformers.evaluation import InformationRetrievalEvaluator


def configure_matplotlib_fonts() -> None:
    font_paths = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]
    candidate_names = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
    ]

    loaded_names: List[str] = []
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                font_manager.fontManager.addfont(fp)
                name = font_manager.FontProperties(fname=fp).get_name()
                if name and name not in loaded_names:
                    loaded_names.append(name)
            except Exception:
                continue

    installed_names = {f.name for f in font_manager.fontManager.ttflist}
    final_names = [n for n in loaded_names + candidate_names if n in installed_names]
    if not final_names:
        final_names = ["DejaVu Sans"]
        print(
            "Warning: No CJK font found by matplotlib. Chinese text may still render as boxes. "
            "Please install and configure a Chinese font."
        )

    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = final_names
    rcParams["axes.unicode_minus"] = False


configure_matplotlib_fonts()

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

try:
    import faiss

    FAISS_AVAILABLE = True
    print("Faiss available for fast retrieval")
except ImportError:
    FAISS_AVAILABLE = False
    print("Faiss not available, will use batch computation instead")

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["NO_GCE_CHECK"] = "true"
print("Offline mode activated: Hugging Face network access disabled.")


class GeometryBridge(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        rank_size: int = 192,
        alpha: float = 0.0,
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.rank_size = int(rank_size)
        self.alpha = float(alpha)
        self.use_layernorm = bool(use_layernorm)

        self.down_proj = nn.Linear(self.hidden_size, self.rank_size)
        self.up_proj = nn.Linear(self.rank_size, self.hidden_size)
        self.activation = nn.GELU()
        self.layernorm = nn.LayerNorm(self.hidden_size) if self.use_layernorm else nn.Identity()

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        token_embeddings = features.get("token_embeddings")
        if token_embeddings is None:
            return features

        bridge = self.up_proj(self.activation(self.down_proj(token_embeddings)))
        fused = self.alpha * token_embeddings + (1.0 - self.alpha) * bridge
        features["token_embeddings"] = self.layernorm(fused)
        return features

    def get_config_dict(self) -> Dict[str, float | int | bool]:
        return {
            "hidden_size": self.hidden_size,
            "rank_size": self.rank_size,
            "alpha": self.alpha,
            "use_layernorm": self.use_layernorm,
        }

    def save(self, output_path: str, *args, **kwargs) -> None:
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, "config.json"), "w", encoding="utf-8") as f:
            json.dump(self.get_config_dict(), f, ensure_ascii=False, indent=2)
        torch.save(self.state_dict(), os.path.join(output_path, "pytorch_model.bin"))

    @staticmethod
    def load(input_path: str) -> "GeometryBridge":
        with open(os.path.join(input_path, "config.json"), "r", encoding="utf-8") as f:
            cfg = json.load(f)
        module = GeometryBridge(
            hidden_size=int(cfg.get("hidden_size", 768)),
            rank_size=int(cfg.get("rank_size", 192)),
            alpha=float(cfg.get("alpha", 0.0)),
            use_layernorm=bool(cfg.get("use_layernorm", True)),
        )
        state_path = os.path.join(input_path, "pytorch_model.bin")
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location="cpu")
            module.load_state_dict(state, strict=False)
        return module


def enable_three_stage_encoder(
    model: SentenceTransformer,
    rank_size: int = 192,
    alpha: float = 0.0,
) -> SentenceTransformer:
    modules = list(model._modules.values())
    if not modules:
        return model

    transformer_module = modules[0]
    pooling_module = None
    for m in modules:
        if isinstance(m, st_models.Pooling):
            pooling_module = m
            break
    if pooling_module is None:
        raise RuntimeError("Cannot find Pooling module in SentenceTransformer modules.")

    hidden_size = int(transformer_module.get_word_embedding_dimension())
    bridge = GeometryBridge(hidden_size=hidden_size, rank_size=rank_size, alpha=alpha, use_layernorm=True)
    multiview_pooling = st_models.Pooling(
        word_embedding_dimension=hidden_size,
        pooling_mode_cls_token=True,
        pooling_mode_mean_tokens=True,
        pooling_mode_max_tokens=True,
    )
    reduce_dense = st_models.Dense(
        in_features=multiview_pooling.get_sentence_embedding_dimension(),
        out_features=hidden_size,
        activation_function=nn.Tanh(),
    )
    normalize = st_models.Normalize()

    model = SentenceTransformer(
        modules=[transformer_module, bridge, multiview_pooling, reduce_dense, normalize],
        device=device,
    )
    return model


class DomainKnowledgeBase:
    def __init__(
        self,
        records: List[Dict[str, Any]],
        embeddings: np.ndarray,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ) -> None:
        if len(records) == 0:
            raise ValueError("DomainKnowledgeBase records cannot be empty.")
        if embeddings.ndim != 2:
            raise ValueError("DomainKnowledgeBase embeddings must be a 2D array.")
        if len(records) != embeddings.shape[0]:
            raise ValueError("records count must match embeddings rows.")

        self.records = records
        self.embeddings = F.normalize(torch.from_numpy(embeddings).float(), p=2, dim=1)
        self.dim = int(self.embeddings.size(1))
        self.bm25_k1 = float(bm25_k1)
        self.bm25_b = float(bm25_b)

        self.doc_tokens: List[List[str]] = []
        self.doc_len: List[int] = []
        self.df: Dict[str, int] = defaultdict(int)
        self.idf: Dict[str, float] = {}
        self.doc_tf: List[Dict[str, int]] = []
        self.N = len(records)

        for rec in self.records:
            toks = rec.get("bm25_tokens")
            if not isinstance(toks, list) or not toks:
                toks = self.simple_tokenize(str(rec.get("text", "")))
            toks = [str(t).strip().lower() for t in toks if str(t).strip()]
            if not toks:
                toks = ["_empty_"]
            self.doc_tokens.append(toks)
            self.doc_len.append(len(toks))

            tf: Dict[str, int] = defaultdict(int)
            for t in toks:
                tf[t] += 1
            self.doc_tf.append(dict(tf))

            for t in set(toks):
                self.df[t] += 1

        self.avgdl = float(sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 1.0
        for t, df_val in self.df.items():
            self.idf[t] = float(np.log(1.0 + (self.N - df_val + 0.5) / (df_val + 0.5)))

    @staticmethod
    def simple_tokenize(text: str) -> List[str]:
        en_tokens = re.findall(r"[A-Za-z0-9_]+", text.lower())
        zh_tokens = re.findall(r"[\u4e00-\u9fff]", text)
        return en_tokens + zh_tokens

    @classmethod
    def from_dir(
        cls,
        kb_dir: str,
        records_name: str = "kb_records.jsonl",
        embeddings_name: str = "kb_embeddings.npy",
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ) -> "DomainKnowledgeBase":
        records_path = os.path.join(kb_dir, records_name)
        emb_path = os.path.join(kb_dir, embeddings_name)
        if not os.path.exists(records_path):
            raise FileNotFoundError(f"KB records file not found: {records_path}")
        if not os.path.exists(emb_path):
            raise FileNotFoundError(f"KB embedding file not found: {emb_path}")

        records: List[Dict[str, Any]] = []
        with open(records_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))

        embeddings = np.load(emb_path).astype(np.float32)
        return cls(records=records, embeddings=embeddings, bm25_k1=bm25_k1, bm25_b=bm25_b)

    def bm25_scores(self, query_text: str) -> np.ndarray:
        q_tokens = self.simple_tokenize(query_text)
        if not q_tokens:
            return np.zeros(self.N, dtype=np.float32)

        q_tf: Dict[str, int] = defaultdict(int)
        for t in q_tokens:
            q_tf[t] += 1

        scores = np.zeros(self.N, dtype=np.float32)
        k1 = self.bm25_k1
        b = self.bm25_b
        avgdl = self.avgdl if self.avgdl > 0 else 1.0

        for i in range(self.N):
            dl = self.doc_len[i]
            tf_doc = self.doc_tf[i]
            score = 0.0
            for t in q_tf.keys():
                if t not in tf_doc:
                    continue
                tf = tf_doc[t]
                idf = self.idf.get(t, 0.0)
                denom = tf + k1 * (1.0 - b + b * (dl / avgdl))
                score += idf * ((tf * (k1 + 1.0)) / (denom + 1e-8))
            scores[i] = float(score)
        return scores

    def lexical_topk_indices(self, query_text: str, lexical_top_k: int = 30) -> np.ndarray:
        bm25 = self.bm25_scores(query_text)
        lex_k = max(1, min(int(lexical_top_k), self.N))
        return np.argsort(bm25)[::-1][:lex_k].copy()

    def build_lexical_cache(
        self,
        query_texts: List[str],
        lexical_top_k: int = 30,
    ) -> Dict[str, np.ndarray]:
        cache: Dict[str, np.ndarray] = {}
        unique_texts = list({str(q).strip(): None for q in query_texts if str(q).strip()}.keys())
        for qt in tqdm(unique_texts, desc=f"Building BM25 lexical cache@{lexical_top_k}"):
            cache[qt] = self.lexical_topk_indices(qt, lexical_top_k=lexical_top_k)
        return cache

    def retrieve_topk_embeddings(
        self,
        query_text: str,
        query_embedding: torch.Tensor,
        lexical_top_k: int = 30,
        semantic_top_k: int = 4,
        precomputed_candidates: np.ndarray | None = None,
    ) -> torch.Tensor:
        lex_k = max(1, min(int(lexical_top_k), self.N))
        if precomputed_candidates is not None and precomputed_candidates.size > 0:
            cand_idx = np.asarray(precomputed_candidates, dtype=np.int64)[:lex_k]
        else:
            cand_idx = self.lexical_topk_indices(query_text, lexical_top_k=lex_k)
        if cand_idx.size == 0:
            cand_idx = np.arange(min(lex_k, self.N))

        cand_emb = self.embeddings[cand_idx].to(query_embedding.device)
        q = F.normalize(query_embedding.view(1, -1), p=2, dim=1)
        sims = torch.matmul(cand_emb, q.T).view(-1)

        sem_k = max(1, min(int(semantic_top_k), int(sims.numel())))
        top_vals, top_pos = torch.topk(sims, k=sem_k)
        top_emb = cand_emb[top_pos]
        weights = F.softmax(top_vals, dim=0)
        return torch.sum(top_emb * weights.unsqueeze(-1), dim=0)


class DomainVectorFusion(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.proj = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.layernorm = nn.LayerNorm(self.hidden_size)

    def forward(self, text_emb: torch.Tensor, domain_emb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([text_emb, domain_emb], dim=-1)
        x = self.proj(x)
        x = F.gelu(x)
        x = self.layernorm(x)
        return x


class DomainEnhancedEncoder:
    def __init__(
        self,
        base_model: SentenceTransformer,
        knowledge_base: DomainKnowledgeBase,
        fusion_module: DomainVectorFusion,
        lexical_top_k: int = 30,
        semantic_top_k: int = 4,
    ) -> None:
        self.base_model = base_model
        self.knowledge_base = knowledge_base
        self.fusion_module = fusion_module
        self.lexical_top_k = int(lexical_top_k)
        self.semantic_top_k = int(semantic_top_k)

    def encode(
        self,
        sentences,
        batch_size: int = 32,
        normalize: bool = True,
        convert_to_tensor: bool = False,
        device: str | None = None,
        show_progress_bar: bool = False,
        **encode_kwargs,
    ):
        texts = [str(s) for s in list(sentences)]
        with torch.no_grad():
            text_emb = self.base_model.encode(
                texts,
                batch_size=batch_size,
                convert_to_tensor=True,
                device=device,
                show_progress_bar=show_progress_bar,
                normalize_embeddings=False,
                **encode_kwargs,
            )
            text_emb = text_emb.clone()
            self.fusion_module = self.fusion_module.to(text_emb.device)

            fused_list = []
            for i, t in enumerate(texts):
                d = self.knowledge_base.retrieve_topk_embeddings(
                    query_text=t,
                    query_embedding=text_emb[i],
                    lexical_top_k=self.lexical_top_k,
                    semantic_top_k=self.semantic_top_k,
                )
                fused_list.append(self.fusion_module(text_emb[i].unsqueeze(0), d.unsqueeze(0)).squeeze(0))
            fused = torch.stack(fused_list, dim=0)

        if normalize:
            fused = F.normalize(fused, p=2, dim=-1)
        if convert_to_tensor:
            return fused
        return fused.detach().cpu().numpy()


class RunwayDataLoader:
    def __init__(self, data_dir: str = os.path.join("retrieval", "runway-1")):
        self.data_dir = data_dir
        self.corpus: Dict[str, str] = {}
        self.queries: Dict[str, str] = {}
        self.qrels: Dict[str, Dict[str, Dict[str, int]]] = {
            "train": {},
            "dev": {},
            "test": {},
        }

    def load_corpus(self) -> Dict[str, str]:
        path = os.path.join(self.data_dir, "runway-document.json")
        print(f"Loading corpus from {path} ...")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.corpus = {str(k): str(v) for k, v in data.items()}
        print(f"  Loaded {len(self.corpus)} documents")
        return self.corpus

    def load_queries(self) -> Dict[str, str]:
        path = os.path.join(self.data_dir, "runway-queries.json")
        print(f"Loading queries from {path} ...")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.queries = {str(k): str(v) for k, v in data.items()}
        print(f"  Loaded {len(self.queries)} queries")
        return self.queries

    def load_split_qrels(self, split: str) -> Dict[str, Dict[str, int]]:
        assert split in {"train", "dev", "test"}
        path = os.path.join(self.data_dir, f"{split}.csv")
        print(f"Loading {split} qrels from {path} ...")

        qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                qid, doc_id, rel_str = parts[0], parts[1], parts[2]
                try:
                    rel = int(rel_str)
                except ValueError:
                    continue
                if rel <= 0:
                    continue
                qrels[qid][doc_id] = rel

        self.qrels[split] = dict(qrels)
        num_pairs = sum(len(d) for d in self.qrels[split].values())
        print(f"  Loaded {len(self.qrels[split])} queries with {num_pairs} query-doc pairs for {split}")
        return self.qrels[split]

    def create_training_samples(self, split: str = "train", num_negatives_per_pos: int = 1) -> List[InputExample]:
        assert split in self.qrels, f"qrels for split '{split}' not loaded yet"
        qrels = self.qrels[split]

        all_doc_ids = list(self.corpus.keys())

        samples: List[InputExample] = []
        print(f"Creating training samples for split='{split}' with {num_negatives_per_pos} negatives per positive ...")

        for qid, doc_rels in tqdm(qrels.items(), desc="Creating samples"):
            if qid not in self.queries:
                continue
            query_text = self.queries[qid]

            pos_doc_ids = {doc_id for doc_id, rel in doc_rels.items() if rel > 0 and doc_id in self.corpus}
            if not pos_doc_ids:
                continue

            for doc_id in pos_doc_ids:
                doc_text = self.corpus[doc_id]
                samples.append(InputExample(texts=[query_text, doc_text], label=1.0))

                for _ in range(num_negatives_per_pos):
                    neg_doc_id = None
                    for _retry in range(10):
                        candidate = random.choice(all_doc_ids)
                        if candidate not in pos_doc_ids:
                            neg_doc_id = candidate
                            break
                    if neg_doc_id is None:
                        continue
                    neg_doc_text = self.corpus[neg_doc_id]
                    samples.append(InputExample(texts=[query_text, neg_doc_text], label=0.0))

        print(f"  Created {len(samples)} training samples (pos+neg)")
        return samples

    def create_relevance_triplet_samples(
        self,
        split: str = "train",
        max_triplets_per_query: int | None = None,
        topk_docs: Dict[str, List[str]] | None = None,
    ) -> List[InputExample]:
        csv_path = os.path.join(self.data_dir, f"{split}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Split file not found for triplet creation: {csv_path}")

        raw_qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
        with open(csv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                qid, doc_id, rel_str = parts[0], parts[1], parts[2]
                try:
                    rel = int(rel_str)
                except ValueError:
                    continue
                if rel < 0:
                    continue
                raw_qrels[qid][doc_id] = rel

        triplets: List[InputExample] = []
        print(f"Creating relevance-aware triplet samples for split='{split}' from {csv_path} ...")

        for qid, doc_rels in tqdm(raw_qrels.items(), desc="Creating triplets"):
            if qid not in self.queries:
                continue

            items = []
            has_positive = False

            allowed_docs = None
            if topk_docs is not None and qid in topk_docs:
                allowed_docs = set(topk_docs[qid])

            for doc_id, rel in doc_rels.items():
                if doc_id not in self.corpus:
                    continue
                if allowed_docs is not None and doc_id not in allowed_docs:
                    continue
                items.append((doc_id, rel))
                if rel > 0:
                    has_positive = True

            if len(items) < 2 or not has_positive:
                continue

            items.sort(key=lambda x: x[1], reverse=True)

            query_text = self.queries[qid]
            triplets_for_q: List[InputExample] = []

            pos_items = [(doc_id, rel) for doc_id, rel in items if rel > 0]
            if not pos_items:
                continue

            num_zero_neg_per_pos = 2
            num_pos_neg_per_pos = 2

            weighted_pos_items = []
            for doc_id, rel in pos_items:
                if rel >= 5:
                    times = 4
                elif rel == 4:
                    times = 3
                elif rel == 3:
                    times = 2
                else:
                    times = 1
                for _ in range(times):
                    weighted_pos_items.append((doc_id, rel))

            for pos_id, pos_rel in weighted_pos_items:
                pos_text = self.corpus.get(pos_id)
                if pos_text is None:
                    continue

                lower_items = [
                    (doc_id, rel)
                    for doc_id, rel in items
                    if doc_id != pos_id and rel < pos_rel
                ]
                if not lower_items:
                    continue

                lower_zero = [(doc_id, rel) for doc_id, rel in lower_items if rel == 0]
                lower_nonzero = [(doc_id, rel) for doc_id, rel in lower_items if rel > 0]

                if lower_zero:
                    k_zero = min(num_zero_neg_per_pos, len(lower_zero))
                    sampled_zero = random.sample(lower_zero, k_zero)
                    for neg_id, neg_rel in sampled_zero:
                        neg_text = self.corpus.get(neg_id)
                        if neg_text is None:
                            continue
                        rel_diff = float(abs(pos_rel - neg_rel))
                        if rel_diff <= 0.0:
                            continue
                        label = -rel_diff
                        triplets_for_q.append(
                            InputExample(
                                texts=[query_text, pos_text, neg_text],
                                label=label,
                            )
                        )

                if lower_nonzero:
                    k_posneg = min(num_pos_neg_per_pos, len(lower_nonzero))
                    sampled_posneg = random.sample(lower_nonzero, k_posneg)
                    for neg_id, neg_rel in sampled_posneg:
                        neg_text = self.corpus.get(neg_id)
                        if neg_text is None:
                            continue
                        rel_diff = float(abs(pos_rel - neg_rel))
                        if rel_diff <= 0.0:
                            continue
                        label = rel_diff
                        triplets_for_q.append(
                            InputExample(
                                texts=[query_text, pos_text, neg_text],
                                label=label,
                            )
                        )

            if not triplets_for_q:
                continue

            if max_triplets_per_query is not None and len(triplets_for_q) > max_triplets_per_query:
                triplets_for_q = random.sample(triplets_for_q, max_triplets_per_query)

            triplets.extend(triplets_for_q)

        if triplets:
            nonzero_triplets: List[InputExample] = []
            zero_triplets: List[InputExample] = []
            for ex in triplets:
                if ex.label is None:
                    continue
                try:
                    label_val = float(ex.label)
                except (TypeError, ValueError):
                    continue
                if label_val > 0.0:
                    nonzero_triplets.append(ex)
                elif label_val < 0.0:
                    zero_triplets.append(ex)

            n_nonzero = len(nonzero_triplets)
            n_zero = len(zero_triplets)

            if n_nonzero > 0 and n_zero > 0:
                max_zero = int((2.0 / 3.0) * n_nonzero)
                if n_zero > max_zero:
                    if max_zero > 0:
                        zero_triplets.sort(
                            key=lambda ex: abs(float(ex.label)) if ex.label is not None else float("inf")
                        )
                        zero_triplets = zero_triplets[:max_zero]
                    else:
                        zero_triplets = []

                triplets = nonzero_triplets + zero_triplets
                random.shuffle(triplets)

        print(f"  Created {len(triplets)} relevance-aware triplet training samples")
        return triplets


def mine_topk_docs_for_queries(
    model: SentenceTransformer,
    corpus: Dict[str, str],
    queries: Dict[str, str],
    top_k: int = 100,
    batch_size: int = 128,
    use_faiss: bool = True,
) -> Dict[str, List[str]]:
    if not queries:
        return {}

    print("\nMining Top-K docs for training queries ...")

    corpus_ids = list(corpus.keys())
    corpus_texts = [corpus[cid] for cid in corpus_ids]
    if not corpus_ids:
        return {}

    actual_top_k = min(top_k, len(corpus_ids))

    corpus_emb = model.encode(
        corpus_texts,
        convert_to_tensor=True,
        batch_size=batch_size,
        device=device,
        show_progress_bar=True,
    )
    corpus_emb = F.normalize(corpus_emb, p=2, dim=1)

    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]

    query_emb = model.encode(
        query_texts,
        convert_to_tensor=True,
        batch_size=batch_size,
        device=device,
        show_progress_bar=True,
    )
    query_emb = F.normalize(query_emb, p=2, dim=1)

    topk_result: Dict[str, List[str]] = {}

    if use_faiss and FAISS_AVAILABLE:
        print("  Using Faiss for Top-K mining ...")
        corpus_np = corpus_emb.cpu().numpy().astype("float32")
        query_np = query_emb.cpu().numpy().astype("float32")
        dim = corpus_np.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(corpus_np)
        scores, indices = index.search(query_np, actual_top_k)

        for i, qid in enumerate(query_ids):
            doc_indices = indices[i][:actual_top_k]
            topk_docs = [corpus_ids[idx] for idx in doc_indices]
            topk_result[qid] = topk_docs
    else:
        print("  Computing Top-K by batched similarities ...")
        for i in range(0, len(query_ids), batch_size):
            batch_q_ids = query_ids[i : i + batch_size]
            batch_q = query_emb[i : i + batch_size]
            sims = torch.mm(batch_q, corpus_emb.T)
            sims_np = sims.cpu().numpy()
            for j, qid in enumerate(batch_q_ids):
                scores = sims_np[j]
                idxs = np.argsort(scores)[::-1][:actual_top_k].copy()
                topk_docs = [corpus_ids[idx] for idx in idxs]
                topk_result[qid] = topk_docs

    print(f"  Mined Top-{actual_top_k} docs for {len(topk_result)} queries")
    return topk_result


class OptimizedRetrievalEvaluator:
    def __init__(
        self,
        corpus: Dict[str, str],
        queries: Dict[str, str],
        qrels: Dict[str, Dict[str, int]],
        batch_size: int = 128,
        use_faiss: bool = True,
    ) -> None:
        self.corpus = corpus
        self.queries = queries
        self.qrels = qrels
        self.batch_size = batch_size
        self.use_faiss = use_faiss and FAISS_AVAILABLE

    @staticmethod
    def _normalize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
        return F.normalize(embeddings, p=2, dim=1)

    def _mrr(self, ranked: List[str], relevant: set) -> float:
        for i, doc in enumerate(ranked, 1):
            if doc in relevant:
                return 1.0 / i
        return 0.0

    def _ndcg(self, ranked: List[str], rel_dict: Dict[str, int], k: int) -> float:
        if not rel_dict:
            return 0.0

        dcg = 0.0
        for i, doc in enumerate(ranked[:k], 1):
            rel = rel_dict.get(doc, 0)
            if rel <= 0:
                continue
            gain = float(2**rel - 1)
            dcg += gain / float(np.log2(i + 1.0))

        rel_scores = [r for r in rel_dict.values() if r > 0]
        if not rel_scores:
            return 0.0
        rel_scores.sort(reverse=True)

        idcg = 0.0
        for i, rel in enumerate(rel_scores[:k], 1):
            gain = float(2**rel - 1)
            idcg += gain / float(np.log2(i + 1.0))

        if idcg == 0.0:
            return 0.0
        return float(dcg / idcg)

    def _recall(self, ranked: List[str], relevant: set) -> float:
        if not ranked:
            return 0.0
        return float(len(set(ranked) & relevant) / len(ranked))

    def _precision(self, ranked: List[str], relevant: set) -> float:
        if not ranked:
            return 0.0
        return float(len(set(ranked) & relevant) / len(ranked))

    def evaluate(self, model: SentenceTransformer, k_values: List[int] | None = None) -> Dict[str, Dict[int, float]]:
        if k_values is None:
            k_values = [1, 3, 5, 10, 20]

        print("\nEvaluating model on retrieval task ...")

        corpus_ids = list(self.corpus.keys())
        corpus_texts = [self.corpus[cid] for cid in corpus_ids]
        print("  Encoding corpus ...")
        corpus_emb = model.encode(
            corpus_texts,
            convert_to_tensor=True,
            batch_size=self.batch_size,
            device=device,
            show_progress_bar=True,
        )
        corpus_emb = self._normalize_embeddings(corpus_emb)

        query_ids = list(self.queries.keys())
        query_texts = [self.queries[qid] for qid in query_ids]
        print("  Encoding queries ...")
        query_emb = model.encode(
            query_texts,
            convert_to_tensor=True,
            batch_size=self.batch_size,
            device=device,
            show_progress_bar=True,
        )
        query_emb = self._normalize_embeddings(query_emb)

        metrics: Dict[str, Dict[int, List[float]]] = {
            "MRR": {k: [] for k in k_values},
            "NDCG": {k: [] for k in k_values},
            "Recall": {k: [] for k in k_values},
            "Precision": {k: [] for k in k_values},
        }

        max_k = max(k_values)

        if self.use_faiss:
            print("  Using Faiss for fast retrieval ...")
            corpus_np = corpus_emb.cpu().numpy().astype("float32")
            query_np = query_emb.cpu().numpy().astype("float32")
            dim = corpus_np.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(corpus_np)
            scores, indices = index.search(query_np, max_k)

            for i, qid in enumerate(tqdm(query_ids, desc="Evaluating")):
                if qid not in self.qrels:
                    continue
                rel_dict = self.qrels[qid]
                relevant = {d for d, r in rel_dict.items() if r > 0}
                if not relevant:
                    continue

                for k in k_values:
                    top_docs = [corpus_ids[idx] for idx in indices[i][:k]]
                    metrics["MRR"][k].append(self._mrr(top_docs, relevant))
                    metrics["NDCG"][k].append(self._ndcg(top_docs, rel_dict, k))
                    metrics["Recall"][k].append(self._recall(top_docs, relevant))
                    metrics["Precision"][k].append(self._precision(top_docs, relevant))
        else:
            print("  Computing similarities in batches ...")
            for i in tqdm(range(0, len(query_ids), self.batch_size), desc="Evaluating"):
                batch_q = query_emb[i : i + self.batch_size]
                sims = torch.mm(batch_q, corpus_emb.T)
                for j, qid in enumerate(query_ids[i : i + self.batch_size]):
                    if qid not in self.qrels:
                        continue
                    rel_dict = self.qrels[qid]
                    relevant = {d for d, r in rel_dict.items() if r > 0}
                    if not relevant:
                        continue

                    scores = sims[j].cpu().numpy()
                    idxs = np.argsort(scores)[::-1].copy()
                    for k in k_values:
                        top_docs = [corpus_ids[idx] for idx in idxs[:k]]
                        metrics["MRR"][k].append(self._mrr(top_docs, relevant))
                        metrics["NDCG"][k].append(self._ndcg(top_docs, rel_dict, k))
                        metrics["Recall"][k].append(self._recall(top_docs, relevant))
                        metrics["Precision"][k].append(self._precision(top_docs, relevant))

        final_metrics: Dict[str, Dict[int, float]] = {}
        for metric_name, k_dict in metrics.items():
            final_metrics[metric_name] = {}
            for k, values in k_dict.items():
                if values:
                    final_metrics[metric_name][k] = float(np.mean(values))
                else:
                    final_metrics[metric_name][k] = 0.0

        return final_metrics
