import os
import json
import hashlib
from pathlib import Path
import logging
from typing import List, Optional, Tuple

from dotenv import load_dotenv
import pandas as pd
import numpy as np
import faiss
from openai import OpenAI

from Custom_Functions.query_engine import QueryEngine
from Custom_Functions.log_utils import get_logger

logger = get_logger(__name__)

# Load environment variables from .env file
load_dotenv()

# Verify API key is loaded
if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("OPENAI_API_KEY not found. Please set it in your .env file or environment variables.")

META_FILENAME = "meta.json"
INDEX_FILENAME = "index.faiss"
IDS_FILENAME = "ids.npy"

class FaissQueryEngine(QueryEngine):
    """
    Extends QueryEngine with semantic search capability using FAISS and OpenAI embeddings.
    Persists the FAISS index to disk to avoid rebuilding on every run.
    """

    def __init__(
        self,
        csv_path: str,
        embedded_columns: List[str],
        embedding_model: str = "text-embedding-3-small",
        index_dir: str = "index/faiss",
        batch_size: int = 64,
        rebuild: bool = False,
    ):
        """
        Args:
            csv_path: Path to the CSV file loaded by the parent QueryEngine.
            embedded_columns: List of dataframe column names to concatenate for embeddings.
            embedding_model: OpenAI embeddings model name.
            index_dir: Directory where FAISS index + metadata are stored.
            batch_size: Number of texts to embed per API request (speeds up embedding).
            rebuild: If True, force a rebuild of the FAISS index even if one exists.
        """
        super().__init__(csv_path)

        if self.df.empty:
            logger.error("DataFrame is empty. Cannot initialize FAISS index.")
            return

        # Validate columns early
        missing = [c for c in embedded_columns if c not in self.df.columns]
        if missing:
            raise ValueError(f"embedded_columns not found in dataframe: {missing}")

        self.embedded_columns = embedded_columns
        self.embedding_model = embedding_model
        self.index_dir = Path(index_dir)
        self.batch_size = max(1, int(batch_size))
        self.index: Optional[faiss.Index] = None
        self.id_mapping: List[int] = []

        # Precompute the text data to embed
        self.text_data: List[str] = (
            self.df[self.embedded_columns].astype(str).agg(" ".join, axis=1).tolist()
        )

        # Decide whether to load or build
        if (not rebuild) and self._index_exists() and self._meta_matches():
            logger.info("Loading existing FAISS index from disk...")
            self._load_index()
        else:
            logger.info("Building FAISS index (this is done once; subsequent runs will load it).")
            embeddings = self._create_embeddings(self.text_data)
            self._build_faiss_index(embeddings)
            self._save_index()

    # ----------------------
    # Public helpers
    # ----------------------
    def preview(self) -> pd.DataFrame:
        """Returns the first few rows of the DataFrame for inspection."""
        if self.df.empty:
            logger.warning("DataFrame is empty. Please check the CSV file path.")
            return pd.DataFrame()
        return self.df.head()

    def semantic_search(self, query: str, k: int = 5) -> pd.DataFrame:
        """
        Performs semantic search for the given query and returns the top-k matching rows
        **with a 'score' column** (lower = more similar for L2 index).
        """
        if self.index is None:
            logger.error("FAISS index is not initialized.")
            return pd.DataFrame()

        # Generate embedding for the query (shape = (1, dim))
        query_vec = self._create_embeddings([query])
        distances, indices = self.index.search(query_vec, k)

        # Convert to DataFrame; attach scores
        idxs = indices[0].tolist()
        dists = distances[0].tolist()

        results = self.df.iloc[idxs].copy().reset_index(drop=True)
        results["score"] = dists
        return results

    # ----------------------
    # Index build/load/save
    # ----------------------
    def _index_exists(self) -> bool:
        return (
            self.index_dir.exists()
            and (self.index_dir / INDEX_FILENAME).exists()
            and (self.index_dir / IDS_FILENAME).exists()
            and (self.index_dir / META_FILENAME).exists()
        )

    def _dataset_fingerprint(self) -> str:
        """
        Create a stable fingerprint for (csv_path + embedded_columns + embedding_model).
        If any of these change, we should rebuild.
        """
        parts = {
            "csv_path": str(getattr(self, "csv_path", "")),
            "embedded_columns": self.embedded_columns,
            "embedding_model": self.embedding_model,
            "row_count": int(len(self.text_data)),
        }
        raw = json.dumps(parts, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _meta_matches(self) -> bool:
        try:
            meta = json.loads((self.index_dir / META_FILENAME).read_text())
            return meta.get("fingerprint") == self._dataset_fingerprint()
        except Exception:
            return False

    def _save_index(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_dir / INDEX_FILENAME))
        np.save(self.index_dir / IDS_FILENAME, np.array(self.id_mapping, dtype=np.int64))
        meta = {
            "fingerprint": self._dataset_fingerprint(),
            "embedding_model": self.embedding_model,
            "embedded_columns": self.embedded_columns,
        }
        (self.index_dir / META_FILENAME).write_text(json.dumps(meta, indent=2))
        logger.info(f"FAISS index persisted to: {self.index_dir}")

    def _load_index(self) -> None:
        self.index = faiss.read_index(str(self.index_dir / INDEX_FILENAME))
        ids = np.load(self.index_dir / IDS_FILENAME)
        self.id_mapping = ids.tolist()
        logger.info(f"Loaded FAISS index with {len(self.id_mapping)} vectors.")

    # ----------------------
    # Embeddings + FAISS
    # ----------------------
    def _create_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generates embeddings for a list of texts using OpenAI's embedding API.
        Uses batching to reduce round-trips. Returns shape (N, D).
        """
        api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI(api_key=api_key)

        vectors: List[List[float]] = []
        # Batch the requests for speed
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            try:
                resp = client.embeddings.create(model=self.embedding_model, input=batch)
                # OpenAI returns one embedding per input in order
                for item in resp.data:
                    vectors.append(item.embedding)
            except Exception as e:
                logger.error(f"Embedding batch failed at [{start}:{start+len(batch)}]: {e}")
                # Fall back with zero vectors of detected dimension (infer from first non-empty vector later)
                # We'll temporarily append None and fix shape after loop if needed.
                for _ in batch:
                    vectors.append(None)  # type: ignore

        # Infer dimension from first good vector
        dim: Optional[int] = None
        for v in vectors:
            if isinstance(v, list):
                dim = len(v)
                break
        if dim is None:
            # Nothing worked; return zeros
            return np.zeros((len(texts), 1536), dtype=np.float32)

        # Replace failed items (None) with zeros of correct dimension
        fixed = []
        for v in vectors:
            if isinstance(v, list):
                fixed.append(v)
            else:
                fixed.append([0.0] * dim)
        arr = np.asarray(fixed, dtype=np.float32)
        return arr

    def _build_faiss_index(self, embeddings: np.ndarray) -> None:
        """Builds a FAISS L2 index from the provided embeddings."""
        if embeddings.ndim != 2:
            raise ValueError(f"Embeddings must be 2D, got shape {embeddings.shape}")

        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        self.index = index
        self.id_mapping = list(range(len(embeddings)))
        logger.info(f"FAISS index built with {len(embeddings)} vectors (dim={dim}).")