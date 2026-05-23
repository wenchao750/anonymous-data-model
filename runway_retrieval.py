
"""\
Runway ??????

- ??: retrieval/sbert-base-nli-mean-tokens
- ??: retrieval/runway-1 ????? runway-document.json, runway-queries.json, train.csv, dev.csv, test.csv
- ??: ????-??-???????? SBERT ????
- ??: ??????? MRR / NDCG / Precision / Recall@K???????
"""

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


# ==================== ??????? ====================

def configure_matplotlib_fonts() -> None:
    """Configure matplotlib font fallback to avoid findfont warnings and Chinese glyph boxes."""
    font_paths = [
        r"C:\Windows\Fonts\msyh.ttc",  # Microsoft YaHei
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\simhei.ttf",  # SimHei
        r"C:\Windows\Fonts\simsun.ttc",  # SimSun
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


# Configure fonts once at startup.
configure_matplotlib_fonts()

# ????
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ??? GPU 0?2?3????? CUDA_VISIBLE_DEVICES=0,2,3?
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

# ????????? main ?????? DDP ????
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ???? Faiss ????
try:
    import faiss  # type: ignore

    FAISS_AVAILABLE = True
    print("Faiss available for fast retrieval")
except ImportError:  # pragma: no cover - ??? faiss ?????
    FAISS_AVAILABLE = False
    print("Faiss not available, will use batch computation instead")

# ?? Hugging Face ????????????
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["NO_GCE_CHECK"] = "true"
print("Offline mode activated: Hugging Face network access disabled.")


class GeometryBridge(nn.Module):
    """Geometry bridge between transformer and pooling: low-rank bottleneck + residual gate."""

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

    def save(self, output_path: str, *args, **kwargs) -> None:  # type: ignore[override]
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, "config.json"), "w", encoding="utf-8") as f:
            json.dump(self.get_config_dict(), f, ensure_ascii=False, indent=2)
        torch.save(self.state_dict(), os.path.join(output_path, "pytorch_model.bin"))

    @staticmethod
    def load(input_path: str) -> "GeometryBridge":  # type: ignore[override]
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
    """Rebuild SentenceTransformer as: Transformer -> GeometryBridge -> MultiViewPooling -> Dense -> Normalize."""
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
    """Domain KB with BM25-inspired lexical prefilter + dense rerank."""

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
            # Standard BM25 IDF with +1 to keep positive values.
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
    """Fuse original embedding and domain embedding: linear + non-linear activation."""

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
    """Inference/training-time encoder wrapper with BM25 prefilter + domain fusion."""

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


# ==================== Runway ????? ====================


class RunwayDataLoader:
    """Runway ?????

    ?????????:
        data_dir/
            runway-document.json   # ????: doc_id -> text
            runway-queries.json    # ????: query_id -> text
            train.csv              # query_id, doc_id, relevance
            dev.csv
            test.csv

    CSV ????:
        queries_68     document_68_1-3     5
    ???????????? 1-5 ???????
    """

    def __init__(self, data_dir: str = os.path.join("retrieval", "runway-1")):
        self.data_dir = data_dir
        self.corpus: Dict[str, str] = {}
        self.queries: Dict[str, str] = {}
        self.qrels: Dict[str, Dict[str, Dict[str, int]]] = {
            "train": {},
            "dev": {},
            "test": {},
        }

    # -------- ?????? --------

    def load_corpus(self) -> Dict[str, str]:
        path = os.path.join(self.data_dir, "runway-document.json")
        print(f"Loading corpus from {path} ...")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # ???? json ????? doc_id
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
        """? train/dev/test CSV ??? qrels ???

        ????: {query_id: {doc_id: relevance_int}}
        ?????? > 0 ????
        """

        assert split in {"train", "dev", "test"}
        path = os.path.join(self.data_dir, f"{split}.csv")
        print(f"Loading {split} qrels from {path} ...")

        qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # ??????????? / ?????
                parts = line.split()
                if len(parts) < 3:
                    continue
                qid, doc_id, rel_str = parts[0], parts[1], parts[2]
                try:
                    rel = int(rel_str)
                except ValueError:
                    # ?????????
                    continue
                if rel <= 0:
                    continue
                qrels[qid][doc_id] = rel

        self.qrels[split] = dict(qrels)
        num_pairs = sum(len(d) for d in self.qrels[split].values())
        print(f"  Loaded {len(self.qrels[split])} queries with {num_pairs} query-doc pairs for {split}")
        return self.qrels[split]

    # -------- ?????? --------

    def create_training_samples(self, split: str = "train", num_negatives_per_pos: int = 1) -> List[InputExample]:
        """?? qrels ?? Sentence-Transformers ? InputExample ???"""

        assert split in self.qrels, f"qrels for split '{split}' not loaded yet"
        qrels = self.qrels[split]

        # ??? query ?????????????? corpus ?????????????
        all_doc_ids = list(self.corpus.keys())

        samples: List[InputExample] = []
        print(f"Creating training samples for split='{split}' with {num_negatives_per_pos} negatives per positive ...")

        for qid, doc_rels in tqdm(qrels.items(), desc="Creating samples"):
            if qid not in self.queries:
                continue
            query_text = self.queries[qid]

            # ?? query ???? doc_id ??
            pos_doc_ids = {doc_id for doc_id, rel in doc_rels.items() if rel > 0 and doc_id in self.corpus}
            if not pos_doc_ids:
                continue

            for doc_id in pos_doc_ids:
                doc_text = self.corpus[doc_id]
                # ????
                samples.append(InputExample(texts=[query_text, doc_text], label=1.0))

                # ???????????????????????? doc ?????
                for _ in range(num_negatives_per_pos):
                    neg_doc_id = None
                    # ????????????
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
        """?? 1-5 ???????? (query, pos_doc, neg_doc) ????

        - ???? query ??????? (d_i, d_j)??? rel_i != rel_j?
          ??????????:
              anchor = query ??
              positive = ??????????
              negative = ??????????
          label = |rel_pos - rel_neg|???? TripletLoss ??? margin?
          ??????????margin ???????????

        - ????????????? max_triplets_per_query ???? query
          ????????????????
        """

        # ?????? rel=0???? 0~5 ?????????
        # - ??? split CSV (train/dev/test) ???? qid, doc_id, rel?
        # - ??? query ????? rel ???????????????? 0?
        # - ??????? rel=0??? label ??????????? Loss ??????

        csv_path = os.path.join(self.data_dir, f"{split}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Split file not found for triplet creation: {csv_path}")

        # ??? split ???? 0~5 ???????? / ???
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

            # ??? query ?? (doc_id, rel)???? corpus ???????
            # ????????? rel>0 ??????????????
            items = []
            has_positive = False

            # ????????????? Top-K ?????? Top-K ??????????
            # ????????? vs ???????Top-K ??????
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

            # ??????????????? (? rel, ? rel) ?
            items.sort(key=lambda x: x[1], reverse=True)

            query_text = self.queries[qid]
            triplets_for_q: List[InputExample] = []

            # ?????????????? rel>0 ??????? lower-rel / rel=0 ???????
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

                # ?? hard zero ??
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

                # ?????????
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

        # ???? rel>0 ??????????? 60%???????? rel=0 ??????
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
                        # ? |label| ?????????????hard?? zero ???
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
    """?????????????? query ?? Top-K ??????? hard negatives?

    ??:
        {query_id: [doc_id1, doc_id2, ..., doc_idK]}
    """

    if not queries:
        return {}

    print("\nMining Top-K docs for training queries ...")

    corpus_ids = list(corpus.keys())
    corpus_texts = [corpus[cid] for cid in corpus_ids]
    if not corpus_ids:
        return {}

    actual_top_k = min(top_k, len(corpus_ids))

    # ??? L2 ????????
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


# ==================== ????? ====================


class OptimizedRetrievalEvaluator:
    """???????? - L2 ??? + ?? Faiss / ????????

    ?????: MRR / NDCG / Recall / Precision
    """

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
        """Graded NDCG: ?? qrels ???????????"""
        if not rel_dict:
            return 0.0

        # ?? DCG????????? k ??????????
        dcg = 0.0
        for i, doc in enumerate(ranked[:k], 1):
            rel = rel_dict.get(doc, 0)
            if rel <= 0:
                continue
            gain = float(2**rel - 1)
            dcg += gain / float(np.log2(i + 1.0))

        # ?? IDCG??? query ?????????????????
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
        # As requested: use the exact same calculation as Precision.
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

        # ?????
        aggregated: Dict[str, Dict[int, float]] = {}
        for metric_name, vals in metrics.items():
            aggregated[metric_name] = {}
            for k, arr in vals.items():
                aggregated[metric_name][k] = float(np.mean(arr)) if arr else 0.0

        return aggregated


# ==================== ????? Triplet Loss ====================


class RelevanceAwareTripletLoss(torch.nn.Module):
    """???? Triplet Loss?? margin ?????????

    ?? (query, pos_doc, neg_doc) ??????? label = |rel_pos - rel_neg|?
    ????????????

        loss_i = max(0, margin_i + sim(a, n) - sim(a, p))

    ???
        margin_i = base_margin * (rel_diff_i / max_rel_diff)

    ?????? 1~5 ??????
    - rel_pos ? rel_neg ?????margin ?????????????"easy / hard negatives"??
    - ???????semi-hard?margin ???????????
    """

    def __init__(
        self,
        model: SentenceTransformer,
        knowledge_base: DomainKnowledgeBase | None = None,
        fusion_module: DomainVectorFusion | None = None,
        lexical_cache: Dict[str, np.ndarray] | None = None,
        lexical_top_k: int = 30,
        semantic_top_k: int = 4,
        base_margin: float = 0.25,
        max_rel_diff: float = 4.0,
        max_zero_margin_ratio: float = 0.6,
        weight_zero: float = 0.3,
        weight_nonzero: float = 1.0,
        isotropy_weight: float = 0.02,
    ) -> None:
        super().__init__()
        self.model = model
        self.knowledge_base = knowledge_base
        self.fusion_module = fusion_module
        self.lexical_cache = lexical_cache or {}
        self.lexical_top_k = int(lexical_top_k)
        self.semantic_top_k = int(semantic_top_k)
        self.base_margin = float(base_margin)
        self.max_rel_diff = float(max_rel_diff) if max_rel_diff > 0 else 1.0
        self.max_zero_margin_ratio = float(max_zero_margin_ratio)
        # ??: ?? rel=0 ???????? & ?? 0 ?????
        self.weight_zero = float(weight_zero)
        self.weight_nonzero = float(weight_nonzero)
        self.isotropy_weight = float(isotropy_weight)

    def _decode_texts_from_features(self, features: Dict[str, torch.Tensor]) -> List[str]:
        transformer = self.model[0] if hasattr(self.model, "__getitem__") else None
        tokenizer = getattr(transformer, "tokenizer", None) if transformer is not None else None
        input_ids = features.get("input_ids")
        if tokenizer is None or input_ids is None:
            bsz = 0 if input_ids is None else int(input_ids.size(0))
            return [""] * bsz
        ids = input_ids.detach().cpu().tolist()
        texts = tokenizer.batch_decode(ids, skip_special_tokens=True)
        return [str(t).strip() for t in texts]

    def _apply_domain_fusion(self, reps: torch.Tensor, texts: List[str]) -> torch.Tensor:
        if self.knowledge_base is None or self.fusion_module is None:
            return reps
        self.fusion_module = self.fusion_module.to(reps.device)
        out = []
        for i, text in enumerate(texts):
            key = str(text).strip()
            precomputed = self.lexical_cache.get(key)
            domain_vec = self.knowledge_base.retrieve_topk_embeddings(
                query_text=text,
                query_embedding=reps[i],
                lexical_top_k=self.lexical_top_k,
                semantic_top_k=self.semantic_top_k,
                precomputed_candidates=precomputed,
            )
            fused = self.fusion_module(reps[i].unsqueeze(0), domain_vec.unsqueeze(0)).squeeze(0)
            out.append(fused)
        return torch.stack(out, dim=0)

    def _isotropy_regularization(self, reps: List[torch.Tensor]) -> torch.Tensor:
        z = torch.cat(reps, dim=0)
        z = F.normalize(z, p=2, dim=-1)
        if z.size(0) < 2:
            return z.new_tensor(0.0)
        cov = torch.matmul(z.T, z) / float(z.size(0))
        eye = torch.eye(cov.size(0), device=cov.device, dtype=cov.dtype)
        return torch.mean((cov - eye) ** 2)

    def forward(self, sentence_features, labels: torch.Tensor):  # type: ignore[override]
        # sentence_features: [features_anchor, features_pos, features_neg]
        reps = [self.model(feat)["sentence_embedding"] for feat in sentence_features]
        text_triplet = [self._decode_texts_from_features(feat) for feat in sentence_features]
        # Speed-up mode: only fuse anchor branch during training.
        reps[0] = self._apply_domain_fusion(reps[0], text_triplet[0])
        anchor, positive, negative = reps

        # ????????????
        sim_pos = F.cosine_similarity(anchor, positive)
        sim_neg = F.cosine_similarity(anchor, negative)

        if labels is None:
            # ???????????? margin
            margin = self.base_margin
            weight_vec = None
        else:
            raw_labels = labels.view(-1).to(anchor.device).float()
            # ??: label<0 ??????????? rel=0???????
            #       |label| = |rel_pos - rel_neg| ???????????
            has_zero = (raw_labels < 0.0).float()
            rel_diff = raw_labels.abs()
            scale = rel_diff / self.max_rel_diff
            margin = self.base_margin * scale

            # ??? rel=0 ?????? margin ??? max_zero_margin_ratio * base_margin
            if hasattr(self, "max_zero_margin_ratio"):
                zero_ratio = float(self.max_zero_margin_ratio)
                if zero_ratio < 1.0:
                    zero_scale = torch.clamp(scale, max=zero_ratio)
                    margin_zero = self.base_margin * zero_scale
                    margin = torch.where(has_zero > 0, margin_zero, margin)

            # ??? rel=0 ??????? 0 ?????????
            if hasattr(self, "weight_zero") and hasattr(self, "weight_nonzero"):
                if self.weight_zero != 1.0 or self.weight_nonzero != 1.0:
                    weight_vec = torch.where(
                        has_zero > 0,
                        torch.full_like(raw_labels, float(self.weight_zero)),
                        torch.full_like(raw_labels, float(self.weight_nonzero)),
                    )
                else:
                    weight_vec = None
            else:
                weight_vec = None

        # Broadcast margin ? batch ??
        if not torch.is_tensor(margin):
            margin = torch.full_like(sim_pos, float(margin))

        losses_triplet = F.relu(margin + sim_neg - sim_pos)

        # ??????????? loss ?????? 0 / ? 0 ????
        if labels is not None and "weight_vec" in locals() and weight_vec is not None:
            # ??????? losses_triplet ????
            if weight_vec.shape != losses_triplet.shape:
                weight_vec = weight_vec.view_as(losses_triplet)
            losses_triplet = losses_triplet * weight_vec

        loss_triplet = losses_triplet.mean()
        if self.isotropy_weight > 0:
            loss_iso = self._isotropy_regularization([anchor, positive, negative])
            return loss_triplet + self.isotropy_weight * loss_iso
        return loss_triplet


# ==================== ????? ====================


def plot_metrics_comparison(results: Dict[str, Dict[int, float]], save_path: str) -> None:
    """???? K ???????? (MRR / NDCG / Recall / Precision)?"""

    metrics = ["MRR", "NDCG", "Recall", "Precision"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("?? K ????????", fontsize=16, fontweight="bold")

    for idx, metric in enumerate(metrics):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]

        if metric not in results:
            ax.axis("off")
            continue

        k_values = sorted(results[metric].keys())
        values = [results[metric][k] for k in k_values]

        ax.plot(k_values, values, marker="o", linewidth=2, markersize=8)
        ax.set_xlabel("K", fontsize=12)
        ax.set_ylabel(f"{metric}@K", fontsize=12)
        ax.set_title(f"{metric}@K", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(k_values)
        ax.set_xticklabels([str(k) for k in k_values])

        if values:
            max_idx = int(np.argmax(values))
            ax.annotate(
                f"{values[max_idx]:.4f}",
                xy=(k_values[max_idx], values[max_idx]),
                xytext=(10, 10),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.7),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Metrics comparison plot saved to {save_path}")
    plt.close()


def plot_metrics_bar_chart(results: Dict[str, Dict[int, float]], k_value: int, save_path: str) -> None:
    """???? K ????? (MRR / NDCG / Recall / Precision)?"""

    metrics = ["MRR", "NDCG", "Recall", "Precision"]
    values = [results.get(metric, {}).get(k_value, 0.0) for metric in metrics]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#98D8C8"]
    bars = ax.bar(metrics, values, color=colors, alpha=0.85, edgecolor="black", linewidth=1.2)

    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title(f"????? K={k_value} ????", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis="y")

    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Bar chart (K={k_value}) saved to {save_path}")
    plt.close()


def plot_metrics_heatmap(results: Dict[str, Dict[int, float]], save_path: str) -> None:
    """????-K ???? (MRR / NDCG / Recall / Precision)?"""

    metrics = ["MRR", "NDCG", "Recall", "Precision"]
    if not results or "MRR" not in results or not results["MRR"]:
        print("No results to plot heatmap.")
        return

    k_values = sorted(results["MRR"].keys())

    data = np.zeros((len(metrics), len(k_values)), dtype=float)
    for i, metric in enumerate(metrics):
        for j, k in enumerate(k_values):
            data[i, j] = results.get(metric, {}).get(k, 0.0)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(np.arange(len(k_values)))
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_xticklabels([f"K={k}" for k in k_values])
    ax.set_yticklabels(metrics)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(metrics)):
        for j in range(len(k_values)):
            ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", color="black", fontsize=9)

    ax.set_title("?????????", fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Score")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Heatmap saved to {save_path}")
    plt.close()


# ==================== ????? ====================


def main() -> None:
    print("=" * 80)
    print("Runway ?????? (SBERT)")
    print("  ??: sbert-base-nli-mean-tokens")
    print("  ??: runway-final")
    print("=" * 80)

    # ????
    MODEL_PATH = os.path.abspath(os.path.join("models","sbert-base-nli-mean-tokens"))
    DATA_DIR = os.path.join("runway-final")
    OUTPUT_DIR = os.path.join("output-xiaorong","zong")
    DOMAIN_KB_DIR = os.path.join( "domain_kb")

    # ?????
    BATCH_SIZE = 48
    EPOCHS = 100  # ????????????
    EVAL_BATCH_SIZE = 128
    USE_FAISS = FAISS_AVAILABLE

    # Scheme-A three-stage settings
    ENABLE_THREE_STAGE = True
    GEOMETRY_RANK = 192
    GEOMETRY_ALPHA = 0.0
    ISOTROPY_WEIGHT = 0.02
    BM25_TOPK = 30
    DOMAIN_SEM_TOPK = 4

    # Efficiency analysis settings
    ENABLE_EFFICIENCY_LOG = True
    EFFICIENCY_WARMUP_RATIO = 0.05
    EFFICIENCY_ROUND_DIGITS = 6

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. ????
    print("\n" + "=" * 80)
    print("Loading Runway data ...")
    print("=" * 80)

    loader = RunwayDataLoader(DATA_DIR)

    corpus = loader.load_corpus()
    queries = loader.load_queries()

    train_qrels = loader.load_split_qrels("train")
    dev_qrels = loader.load_split_qrels("dev")
    test_qrels = loader.load_split_qrels("test")

    # 2. ??????
    print("\n" + "=" * 80)
    print("Loading local SentenceTransformer model ...")
    print("=" * 80)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"???????: {MODEL_PATH}")

    model = SentenceTransformer(MODEL_PATH, device=device, local_files_only=True)
    print(f"  Model loaded from {MODEL_PATH}")

    if ENABLE_THREE_STAGE:
        model = enable_three_stage_encoder(
            model,
            rank_size=GEOMETRY_RANK,
            alpha=GEOMETRY_ALPHA,
        )
        print(
            "  Three-stage encoder enabled: Transformer -> GeometryBridge -> "
            "MultiViewPooling"
        )

    domain_kb = DomainKnowledgeBase.from_dir(DOMAIN_KB_DIR)
    fusion_module = DomainVectorFusion(hidden_size=model.get_sentence_embedding_dimension())
    enhanced_model = DomainEnhancedEncoder(
        base_model=model,
        knowledge_base=domain_kb,
        fusion_module=fusion_module,
        lexical_top_k=BM25_TOPK,
        semantic_top_k=DOMAIN_SEM_TOPK,
    )
    print(
        f"  Domain KB enabled: BM25 top-{BM25_TOPK} prefilter -> "
        f"semantic top-{DOMAIN_SEM_TOPK} -> dynamic weighted fusion"
    )

    # ???????????? Top-K ??????? hard negative???? rel=0?
    # ?????????? vs ???????Top-K ??????
    train_queries_text = {qid: queries[qid] for qid in train_qrels.keys() if qid in queries}
    train_query_text_values = list(train_queries_text.values())
    train_lexical_cache = domain_kb.build_lexical_cache(
        query_texts=train_query_text_values,
        lexical_top_k=BM25_TOPK,
    )
    print(f"  Offline BM25 cache built for {len(train_lexical_cache)} unique train queries")

    # 3. ??????? (?? dev ?????????? InformationRetrievalEvaluator)
    dev_queries_for_eval = {qid: queries[qid] for qid in dev_qrels.keys() if qid in queries}

    val_evaluator = InformationRetrievalEvaluator(
        queries=dev_queries_for_eval,
        corpus=corpus,
        relevant_docs=dev_qrels,
        name="validation",
        batch_size=EVAL_BATCH_SIZE,
        ndcg_at_k=[10],
    )

    # 4. ?????????? Triplet Loss ??????
    print("\n" + "=" * 80)
    print("Start training ...")
    print("=" * 80)

    train_loss = RelevanceAwareTripletLoss(
        model,
        knowledge_base=domain_kb,
        fusion_module=fusion_module,
        lexical_cache=train_lexical_cache,
        lexical_top_k=BM25_TOPK,
        semantic_top_k=DOMAIN_SEM_TOPK,
        base_margin=0.25,
        max_rel_diff=4.0,
        isotropy_weight=ISOTROPY_WEIGHT,
    )

    mining_stages = 3
    epochs_per_stage = max(1, EPOCHS // mining_stages)

    efficiency_metrics: Dict[str, Any] = {
        "enabled": ENABLE_EFFICIENCY_LOG,
        "warmup_ratio": EFFICIENCY_WARMUP_RATIO,
        "stages": [],
    }

    for stage in range(mining_stages):
        print(f"\n----- Hard negative mining stage {stage + 1}/{mining_stages} -----")

        stage_t0 = time.perf_counter()
        mine_t0 = time.perf_counter()
        topk_docs_train = mine_topk_docs_for_queries(
            enhanced_model,
            corpus,
            train_queries_text,
            top_k=100,
            batch_size=EVAL_BATCH_SIZE,
            use_faiss=USE_FAISS,
        )
        mine_seconds = time.perf_counter() - mine_t0

        # ??????? Top-K ???????????????????? hard zero ????????
        sample_t0 = time.perf_counter()
        train_samples = loader.create_relevance_triplet_samples(
            "train",
            max_triplets_per_query=200,
            topk_docs=topk_docs_train,
        )
        train_loader = DataLoader(train_samples, shuffle=True, batch_size=BATCH_SIZE)
        sample_build_seconds = time.perf_counter() - sample_t0

        total_steps = len(train_loader) * epochs_per_stage if len(train_loader) > 0 else 0
        warmup_steps = int(EFFICIENCY_WARMUP_RATIO * total_steps) if total_steps > 0 else 0
        print(f"Stage {stage + 1}: Total steps: {total_steps} | Warmup steps: {warmup_steps}")

        if total_steps <= 0:
            print("Warning: no training samples found for this stage, skip training phase.")
            continue

        fit_t0 = time.perf_counter()
        model.fit(
            train_objectives=[(train_loader, train_loss)],
            epochs=epochs_per_stage,
            warmup_steps=warmup_steps,
            evaluator=val_evaluator,
            evaluation_steps=max(1, len(train_loader)),
            output_path=OUTPUT_DIR,
            save_best_model=True,
            show_progress_bar=True,
            optimizer_params={"lr": 4e-5},
        )
        train_fit_seconds = time.perf_counter() - fit_t0
        stage_seconds = time.perf_counter() - stage_t0

        effective_steps = max(1, total_steps - warmup_steps)
        samples_per_step = BATCH_SIZE
        stage_metrics = {
            "stage": stage + 1,
            "mine_seconds": round(mine_seconds, EFFICIENCY_ROUND_DIGITS),
            "sample_build_seconds": round(sample_build_seconds, EFFICIENCY_ROUND_DIGITS),
            "train_fit_seconds": round(train_fit_seconds, EFFICIENCY_ROUND_DIGITS),
            "stage_total_seconds": round(stage_seconds, EFFICIENCY_ROUND_DIGITS),
            "total_steps": int(total_steps),
            "warmup_steps": int(warmup_steps),
            "effective_steps": int(effective_steps),
            "train_step_latency_ms": round((train_fit_seconds / effective_steps) * 1000.0, EFFICIENCY_ROUND_DIGITS),
            "steps_per_sec": round(effective_steps / max(train_fit_seconds, 1e-12), EFFICIENCY_ROUND_DIGITS),
            "samples_per_sec": round((effective_steps * samples_per_step) / max(train_fit_seconds, 1e-12), EFFICIENCY_ROUND_DIGITS),
            "train_samples": int(len(train_samples)),
            "bm25_topk": int(BM25_TOPK),
            "fusion_topk": int(DOMAIN_SEM_TOPK),
        }
        efficiency_metrics["stages"].append(stage_metrics)
        print("Stage efficiency:", json.dumps(stage_metrics, ensure_ascii=False))

    if efficiency_metrics["stages"]:
        total_mine = sum(s["mine_seconds"] for s in efficiency_metrics["stages"])
        total_sample = sum(s["sample_build_seconds"] for s in efficiency_metrics["stages"])
        total_fit = sum(s["train_fit_seconds"] for s in efficiency_metrics["stages"])
        total_stage = sum(s["stage_total_seconds"] for s in efficiency_metrics["stages"])
        total_effective_steps = sum(int(s["effective_steps"]) for s in efficiency_metrics["stages"])
        efficiency_metrics["summary"] = {
            "mine_seconds_total": round(total_mine, EFFICIENCY_ROUND_DIGITS),
            "sample_build_seconds_total": round(total_sample, EFFICIENCY_ROUND_DIGITS),
            "train_fit_seconds_total": round(total_fit, EFFICIENCY_ROUND_DIGITS),
            "stage_total_seconds": round(total_stage, EFFICIENCY_ROUND_DIGITS),
            "avg_train_step_latency_ms": round((total_fit / max(total_effective_steps, 1)) * 1000.0, EFFICIENCY_ROUND_DIGITS),
            "avg_steps_per_sec": round(total_effective_steps / max(total_fit, 1e-12), EFFICIENCY_ROUND_DIGITS),
            "avg_samples_per_sec": round((total_effective_steps * BATCH_SIZE) / max(total_fit, 1e-12), EFFICIENCY_ROUND_DIGITS),
        }

    print(f"Training complete. Best model saved to {OUTPUT_DIR}")
    fusion_path = os.path.join(OUTPUT_DIR, "domain_fusion.pt")
    torch.save(fusion_module.state_dict(), fusion_path)
    print(f"Domain fusion weights saved to {fusion_path}")

    # 5. ?????????????????
    print("\n" + "=" * 80)
    print("Evaluating best model on test set ...")
    print("=" * 80)

    best_model = SentenceTransformer(OUTPUT_DIR, device=device, local_files_only=True)
    best_fusion_module = DomainVectorFusion(hidden_size=best_model.get_sentence_embedding_dimension())
    if not os.path.exists(fusion_path):
        raise FileNotFoundError(f"domain_fusion.pt not found for evaluation: {fusion_path}")
    best_fusion_state = torch.load(fusion_path, map_location=device)
    best_fusion_module.load_state_dict(best_fusion_state)
    print(f"Loaded domain fusion weights from {fusion_path}")
    best_enhanced_model = DomainEnhancedEncoder(
        base_model=best_model,
        knowledge_base=domain_kb,
        fusion_module=best_fusion_module,
        lexical_top_k=BM25_TOPK,
        semantic_top_k=DOMAIN_SEM_TOPK,
    )

    test_evaluator = OptimizedRetrievalEvaluator(
        corpus=corpus,
        queries=queries,
        qrels=test_qrels,
        batch_size=EVAL_BATCH_SIZE,
        use_faiss=USE_FAISS,
    )

    k_values = [1, 3, 5, 10, 20]
    results = test_evaluator.evaluate(best_enhanced_model, k_values=k_values)

    # ???????
    print("\nTest set retrieval results:")
    print(json.dumps(results, indent=2, ensure_ascii=False))

    results_path = os.path.join(OUTPUT_DIR, "evaluation_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {results_path}")

    if ENABLE_EFFICIENCY_LOG:
        efficiency_path = os.path.join(OUTPUT_DIR, "efficiency_metrics.json")
        with open(efficiency_path, "w", encoding="utf-8") as f:
            json.dump(efficiency_metrics, f, indent=2, ensure_ascii=False)
        print(f"Efficiency metrics saved to {efficiency_path}")

    # 6. ???: ???? MRR / NDCG / Recall / Precision ????
    print("\n" + "=" * 80)
    print("Visualizing metrics ...")
    print("=" * 80)

    comparison_path = os.path.join(OUTPUT_DIR, "metrics_comparison.png")
    plot_metrics_comparison(results, comparison_path)

    bar_k10_path = os.path.join(OUTPUT_DIR, "metrics_bar_k10.png")
    plot_metrics_bar_chart(results, k_value=10, save_path=bar_k10_path)

    bar_k3_path = os.path.join(OUTPUT_DIR, "metrics_bar_k3.png")
    plot_metrics_bar_chart(results, k_value=3, save_path=bar_k3_path)

    heatmap_path = os.path.join(OUTPUT_DIR, "metrics_heatmap.png")
    plot_metrics_heatmap(results, heatmap_path)

    print("\nAll visualizations saved in:")
    print(f"  {OUTPUT_DIR}")



if __name__ == "__main__":
    main()
