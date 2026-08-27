"""RAG utilities for document loading, vectorization, and hybrid search."""

import re
import pickle
import uuid
from pathlib import Path
from typing import Any, List, Dict, Optional, Union
from functools import lru_cache

_TOKEN_PATTERN = re.compile(r"\b\w+\b")
_BM25_CACHE: Dict[str, Any] = {}

# ─── Lazy-loaded LangChain dependencies ───────────────────────────────

def _lazy_loaders():
    from langchain_community.document_loaders import TextLoader, JSONLoader
    return TextLoader, JSONLoader

def _lazy_splitter():
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    return RecursiveCharacterTextSplitter

def _lazy_chroma():
    from langchain_chroma import Chroma
    return Chroma

def _lazy_embeddings():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings

def _lazy_import_bm25():
    from rank_bm25 import BM25Okapi
    return BM25Okapi

def _lazy_rerank():
    from langchain_huggingface import HuggingFaceCrossEncoder
    from langchain_core.documents import Document as LCDocument
    return HuggingFaceCrossEncoder, LCDocument


# ─── Caching BM25 Index ───────────────────────────────────────────────

def _get_bm25_index(persist_directory: Path, collection_name: str):
    """Retrieve the BM25 index for a given collection, using a module-level cache.

    Checks the in-memory cache for the requested index. If not found, loads it
    from the corresponding pickle file on disk and stores it in the cache for
    future lookups. This avoids redundant disk I/O for subsequent queries.

    Args:
        persist_directory: The directory path where the BM25 index is stored.
        collection_name: The name of the collection to identify the specific
                         index file (e.g., 'rag_collection').

    Returns:
        A dictionary containing the loaded BM25 model and associated document IDs.

    Raises:
        FileNotFoundError: If the pickle file for the specified collection does
                           not exist in the persist directory.
        pickle.UnpicklingError: If the file is corrupted or cannot be deserialized.
    """
    cache_key = str(persist_directory) + collection_name
    if cache_key not in _BM25_CACHE:
        bm25_path = persist_directory / f"{collection_name}_bm25.pkl"
        with open(bm25_path, "rb") as f:
            _BM25_CACHE[cache_key] = pickle.load(f)
    return _BM25_CACHE[cache_key]


# ─── Document Loading Functions ───────────────────────────────────────

def load_txt_documents(
    file_path: Union[str, Path],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List:
    """Load a .txt file (or directory of .txt files) into LangChain Document objects.

    Args:
        file_path: Path to a single .txt file or a directory containing .txt files.
        chunk_size: Maximum size of each text chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        A list of langchain Document objects split into chunks.
    """
    TextLoader, JSONLoader = (
        _lazy_loaders()
    )
    RecursiveCharacterTextSplitter = (
        _lazy_splitter()
    )

    if chunk_overlap >= chunk_size:
        raise ValueError(f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})")

    file_path = Path(file_path)

    if file_path.is_file():
        loader = TextLoader(str(file_path), encoding="utf-8")
        documents = loader.load()
    elif file_path.is_dir():
        documents = []
        for txt_file in file_path.glob("*.txt"):
            loader = TextLoader(str(txt_file), encoding="utf-8")
            documents.extend(loader.load())
    else:
        raise FileNotFoundError(f"Path does not exist: {file_path}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


def load_json_documents(
    file_path: Union[str, Path],
    jq_schema: str = ".",
    text_content_key: Optional[str] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List:
    """Load a .json file into LangChain Document objects.

    Args:
        file_path: Path to a single .json file or a directory containing .json files.
        jq_schema: JSONPath-like schema to extract content (e.g., '.items[*]').
        text_content_key: If set, only the value under this key in each record is used as page_content.
        chunk_size: Maximum size of each text chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        A list of langchain Document objects split into chunks.
    """
    TextLoader, JSONLoader = (
        _lazy_loaders()
    )
    RecursiveCharacterTextSplitter = (
        _lazy_splitter()
    )

    if chunk_overlap >= chunk_size:
        raise ValueError(f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})")

    file_path = Path(file_path)

    if file_path.is_file():
        loader = JSONLoader(
            file=str(file_path),
            jq_schema=jq_schema,
            text_content=text_content_key,
        )
        documents = loader.load()
    elif file_path.is_dir():
        documents = []
        for json_file in file_path.glob("*.json"):
            loader = JSONLoader(
                file=str(json_file),
                jq_schema=jq_schema,
                text_content=text_content_key,
            )
            documents.extend(loader.load())
    else:
        raise FileNotFoundError(f"Path does not exist: {file_path}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


# ─── Chroma RAG Vector Store Functions ────────────────────────────────

@lru_cache(maxsize=1)
def _get_cached_embedder(model_name: str) -> "HuggingFaceEmbeddings":
    HuggingFaceEmbeddings = (_lazy_embeddings())
    return HuggingFaceEmbeddings(model_name=model_name,)


def create_chroma_rag(
    documents: List,
    persist_directory: Union[str, Path] = "./chroma_db",
    collection_name: str = "rag_collection",
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
) -> "Chroma":
    """Create a Chroma vector store from LangChain Document objects for RAG.

    Embeds the provided documents using embeddings and stores them
    in a persistent Chroma collection on disk.

    Args:
        documents: A list of langchain Document objects (e.g., from load_txt_documents).
        persist_directory: Directory where the Chroma database will be persisted.
        collection_name: Name of the Chroma collection to create or update.
        embedding_model: The embedding model to use.

    Returns:
        A Chroma vector store instance ready for similarity search queries.

    Example:
        >>> docs = load_txt_documents("data/docs/")
        >>> vector_store = create_chroma_rag(docs)
        >>> results = vector_store.similarity_search("my query", k=3)
    """
    Chroma = (
        _lazy_chroma()
    )

    BM25Okapi = _lazy_import_bm25()

    persist_directory = Path(persist_directory)
    persist_directory.mkdir(parents=True, exist_ok=True)

    doc_ids = []
    for doc in documents:
        doc_id = str(uuid.uuid4())
        doc_ids.append(doc_id)
        doc.metadata["__id__"] = doc_id

    embeddings = _get_cached_embedder(model_name=embedding_model)

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(persist_directory),
        collection_name=collection_name,
        ids=doc_ids,
    )

    tokenized_corpus = [_TOKEN_PATTERN.findall(doc.page_content.lower()) for doc in documents]
    bm25_index = BM25Okapi(tokenized_corpus)

    bm25_data = {
        "bm25_model": bm25_index,
        "doc_ids": doc_ids,
    }

    bm25_path = persist_directory / f"{collection_name}_bm25.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    return vector_store


def load_chroma_rag(
    persist_directory: Union[str, Path] = "./chroma_db",
    collection_name: str = "rag_collection",
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
) -> "Chroma":
    """Load an existing Chroma vector store from disk for querying.

    Args:
        persist_directory: Directory where the Chroma database was persisted.
        collection_name: Name of the Chroma collection to load.
        embedding_model: The embedding model used when creating the store.

    Returns:
        A Chroma vector store instance ready for similarity search queries.

    Raises:
        FileNotFoundError: If the persist directory does not exist.

    Example:
        >>> vector_store = load_chroma_rag()
        >>> results = vector_store.similarity_search("my query", k=3)
    """
    Chroma = (
        _lazy_chroma()
    )

    persist_directory = Path(persist_directory)

    if not persist_directory.exists():
        raise FileNotFoundError(
            f"Chroma database directory does not exist: {persist_directory}. "
            "Please create a RAG store first using create_chroma_rag()."
        )

    embeddings = _get_cached_embedder(model_name=embedding_model)

    vector_store = Chroma(
        persist_directory=str(persist_directory),
        collection_name=collection_name,
        embedding_function=embeddings,
    )

    return vector_store


# ─── Hybrid Search: BM25 + Semantic (Reciprocal Rank Fusion) ──────────

def hybrid_search(
    vector_store: "Chroma",
    query: str,
    k: int = 20,
    bm25_weight: float = 0.5,
    semantic_weight: float = 0.5,
    bm25_top_k: int = 50,
    semantic_top_k: int = 50,
    rrf_k: int = 60,
) -> List["Document"]:
    """Perform hybrid search combining BM25 keyword matching and semantic similarity.

    Uses Reciprocal Rank Fusion (RRF) to merge results from sparse (BM25) and dense
    (embedding-based) retrieval methods. This often yields better recall and precision
    than either method alone.

    Args:
        vector_store: A Chroma vector store instance (from create_chroma_rag or load_chroma_rag).
        query: The natural language query string to search for.
        k: Number of top documents to return after fusion.
        bm25_weight: Weight for BM25-ranked results in RRF blending (0-1).
        semantic_weight: Weight for semantic-ranked results in RRF blending (0-1).
        bm25_top_k: Number of documents to retrieve from BM25-ranked search before fusion.
        semantic_top_k: Number of documents to retrieve from semantic search before fusion.
        rrf_k: Smoothing constant for Reciprocal Rank Fusion (default 60).

    Returns:
        A list of langchain Document objects ranked by hybrid relevance.

    Example:
        >>> vector_store = load_chroma_rag()
        >>> results = hybrid_search(vector_store, "machine learning basics", k=20)
        >>> for doc in results:
        ...     print(doc.page_content[:200])
    """
    from langchain_core.documents import Document as LCDocument

    persist_dir_str = getattr(vector_store, "_persist_directory", None) or getattr(vector_store, "persist_directory", "./chroma_db")
    persist_directory = Path(persist_dir_str)
    
    collection_name = getattr(vector_store, "_collection_name", "rag_collection")
    bm25_path = persist_directory / f"{collection_name}_bm25.pkl"

    if not bm25_path.exists():
        raise FileNotFoundError(
            f"BM25-Indexdatei nicht unter {bm25_path} gefunden. "
            "Bitte führe zuerst 'create_chroma_rag' aus."
        )

    bm25_data = _get_bm25_index(persist_directory, collection_name)
    
    bm25_model = bm25_data["bm25_model"]
    global_doc_ids = bm25_data["doc_ids"]

    tokenized_query = _TOKEN_PATTERN.findall(query.lower())
    bm25_scores = bm25_model.get_scores(tokenized_query)
    
    bm25_ranked_pairs = sorted(
        zip(bm25_scores, global_doc_ids),
        key=lambda x: x[0],
        reverse=True
    )[:bm25_top_k]
    
    bm25_ranked_ids = [doc_id for score, doc_id in bm25_ranked_pairs if score > 0]

    semantic_results = vector_store.similarity_search_with_score(
        query=query, 
        k=semantic_top_k
    )

    semantic_ranked_ids = []
    for doc, _score in semantic_results:
        doc_id = doc.metadata.get("__id__")
        if doc_id:
            semantic_ranked_ids.append(doc_id)

    if not bm25_ranked_ids and not semantic_ranked_ids:
        return []

    weighted_rrf_scores: Dict[str, float] = {}

    for rank, doc_id in enumerate(semantic_ranked_ids):
        weighted_rrf_scores[doc_id] = weighted_rrf_scores.get(doc_id, 0.0) + (semantic_weight / (rrf_k + (rank + 1)))

    for rank, doc_id in enumerate(bm25_ranked_ids):
        weighted_rrf_scores[doc_id] = weighted_rrf_scores.get(doc_id, 0.0) + (bm25_weight / (rrf_k + (rank + 1)))

    fused_ranked_ids = [
        doc_id for doc_id, _score in sorted(
            weighted_rrf_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
    ][:k]

    if not fused_ranked_ids:
        return []

    chroma_fetch = vector_store.get(ids=fused_ranked_ids)
    
    fetched_docs = chroma_fetch.get("documents", [])
    fetched_metadatas = chroma_fetch.get("metadatas", [{}] * len(fetched_docs))
    fetched_ids = chroma_fetch.get("ids", [])

    id_to_lc_doc = {}
    for doc_id, text, metadata in zip(fetched_ids, fetched_docs, fetched_metadatas):
        id_to_lc_doc[doc_id] = LCDocument(
            page_content=text,
            metadata=metadata
        )

    final_documents = [
        id_to_lc_doc[doc_id] 
        for doc_id in fused_ranked_ids 
        if doc_id in id_to_lc_doc
    ]

    return final_documents


# ─── Reranking ────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_cached_encoder(model_name: str) -> "HuggingFaceCrossEncoder":
    HuggingFaceCrossEncoder, LCDocument = (_lazy_rerank())
    return HuggingFaceCrossEncoder(model_name=model_name)


def ai_rerank_documents(
    query: str,
    documents: List["LCDocument"],
    model: str = "BAAI/bge-reranker-base",
    top_k: int = 3,
    score_threshold: float = 0.3
) -> List["LCDocument"]:
    """Rerank documents locally using a Hugging Face Cross-Encoder model.

    Scores each document based on its relevance to the query, filters out
    documents below a relevance threshold, and returns the top-k most
    relevant documents.

    Args:
        query: The user query string to evaluate relevance against.
        documents: A list of LangChain Document objects to rerank.
        model: The Hugging Face Cross-Encoder model name to use for scoring
               (e.g., "BAAI/bge-reranker-base").
        top_k: The maximum number of documents to return after filtering.
        score_threshold: The minimum relevance score required for a document
                         to be included in the results.

    Returns:
        A list of LangChain Document objects sorted by relevance score in
        descending order. Each document's metadata will include a
        "relevance_score" key.

    Note:
        If an error occurs during reranking, the function falls back to
        returning the first `top_k` documents from the original list without
        reranking.
    """
    if not documents:
        return []

    try:
        encoder = _get_cached_encoder(model)

        pairs = [[query, doc.page_content] for doc in documents]
        scores = encoder.score(pairs)

        for doc, score in zip(documents, scores):
            doc.metadata["relevance_score"] = float(score)

        reranked_docs = sorted(
            documents, 
            key=lambda x: x.metadata["relevance_score"], 
            reverse=True
        )
        
        filtered_docs = [
            doc for doc in reranked_docs 
            if doc.metadata["relevance_score"] >= score_threshold
        ]

        return filtered_docs[:top_k]

    except Exception as e:
        print(f"Error encountered during reranking: {e}. Using fallback.")
        return documents[:top_k]