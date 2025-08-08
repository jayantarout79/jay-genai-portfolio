import os
import logging
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

class FaissQueryEngine(QueryEngine):
    """
    Extends QueryEngine with semantic search capability using FAISS and OpenAI embeddings.
    Performs semantic search using OpenAI embeddings and a FAISS index.
    """
    def __init__(self, csv_path: str, embedded_columns: list, embedding_model: str = "text-embedding-3-large"):
        super().__init__(csv_path)

        if self.df.empty:
            logger.error("DataFrame is empty. Cannot initialize FAISS index.")
            return
        self.embedded_columns = embedded_columns
        self.embedding_model = embedding_model
        self.index = None
        self.id_mapping = []
        
        # Step 1: Prepare text data for embedding
        logger.info("Preparing text data for embedding...")
        self.text_data = self.df[self.embedded_columns].astype(str).agg(' '.join, axis=1).tolist()

        # Step 2: Generate embeddings
        logger.info("Generating embeddings...")
        self.embeddings = self._create_embeddings(self.text_data)

        # Step 3: Build FAISS index
        logger.info("Building FAISS index...")
        self._build_faiss_index(self.embeddings)

    def preview(self) -> pd.DataFrame:
        """Returns the first few rows of the DataFrame for inspection."""
        if self.df.empty:
            logger.warning("DataFrame is empty. Please check the CSV file path.")
            return pd.DataFrame()
        return self.df.head()

    def _create_embeddings(self, texts: list) -> np.ndarray:
        """Generates embeddings for a list of texts using OpenAI's embedding API."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY not found in environment variables.")
            return np.zeros((len(texts), 1536), dtype=np.float32)

        client = OpenAI(api_key=api_key)
        vectors = []
        for text in texts:
            try:
                response = client.embeddings.create(
                    model=self.embedding_model,
                    input=text
                )
                vector = response.data[0].embedding
                vectors.append(vector)
            except Exception as e:
                logger.error(f"Error generating embedding for text '{text}': {e}")
                vectors.append(np.zeros(1536))
        return np.array(vectors, dtype=np.float32)
    
    def _build_faiss_index(self, embeddings: np.ndarray):
        """Builds a FAISS index from the provided embeddings."""
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        self.id_mapping = list(range(len(embeddings)))
        logger.info(f"FAISS index built with {len(embeddings)} vectors.")

    def semantic_search(self, query: str, k: int = 5) -> pd.DataFrame:
        """Performs semantic search for the given query and returns the top-k matching rows."""
        if self.index is None:
            logger.error("FAISS index is not initialized.")
            return pd.DataFrame()

        # Generate embedding for the query
        query_embedding = self._create_embeddings([query])

        # Search the index
        distances, indices = self.index.search(query_embedding, k)
        
        # Retrieve corresponding rows from the DataFrame
        results = self.df.iloc[indices[0]]
        return results.reset_index(drop=True)