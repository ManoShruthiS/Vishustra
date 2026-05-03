import logging
from typing import Dict, Any, List, Optional, Protocol, Tuple
import uuid
import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class RouterPath(BaseModel):
    """
    Represents a definable route within the Vishustra framework, encapsulating
    metadata for semantic routing decisions.

    Attributes:
        name (str): A unique identifier for this path.
        description (str): A detailed natural language description of what this
                           path handles or represents. This description is used
                           to generate embeddings for semantic matching.
        target_id (str): An identifier for the actual backend component (e.g.,
                         a specific chain ID, tool name, or sub-agent function)
                         that this path routes to.
        metadata (Dict[str, Any]): Arbitrary additional metadata associated
                                   with this path.
        embedding (Optional[List[float]]): The pre-computed embedding of the
                                           description. Populated internally.
    """
    name: str = Field(..., description="Unique name for the routing path.")
    description: str = Field(..., description="Natural language description for semantic matching.")
    target_id: str = Field(..., description="Identifier for the actual component this path routes to.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata for the path.")
    embedding: Optional[List[float]] = Field(None, exclude=True, description="Pre-computed embedding of the description.")

    class Config:
        """Pydantic configuration for RouterPath."""
        arbitrary_types_allowed = True


class BaseEmbeddingModel(Protocol):
    """
    Abstract base class for embedding models used by the SemanticRouter.
    Any concrete embedding model must implement the `embed` method.
    """
    def embed(self, text: str) -> List[float]:
        """
        Generates a numerical vector embedding for the given text.

        Args:
            text (str): The input string to embed.

        Returns:
            List[float]: A list of floats representing the embedding vector.
        """
        ...

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generates numerical vector embeddings for a batch of texts.
        Default implementation iterates over `embed`. Implementations can
        override for efficiency.

        Args:
            texts (List[str]): A list of input strings to embed.

        Returns:
            List[List[float]]: A list of embedding vectors.
        """
        return [self.embed(text) for text in texts]


class BaseVectorStore(Protocol):
    """
    Abstract base class for vector store implementations used by the SemanticRouter.
    This defines the interface for storing and querying vector embeddings.
    """
    def add_vector(self, vector: List[float], document_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Adds a single vector with an associated ID and optional metadata to the store.

        Args:
            vector (List[float]): The embedding vector to add.
            document_id (str): A unique identifier for the document associated with the vector.
            metadata (Optional[Dict[str, Any]]): Arbitrary metadata to store alongside the vector.
        """
        ...

    def search(self, query_vector: List[float], k: int = 1) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Performs a similarity search against the stored vectors.

        Args:
            query_vector (List[float]): The vector to search for similar items.
            k (int): The number of top similar results to return.

        Returns:
            List[Tuple[str, float, Dict[str, Any]]]: A list of tuples, where each tuple contains
            (document_id, similarity_score, metadata). Results are ordered by similarity score (highest first).
        """
        ...

    def get_by_id(self, document_id: str) -> Optional[Tuple[List[float], Dict[str, Any]]]:
        """
        Retrieves a vector and its metadata by its document_id.

        Args:
            document_id (str): The ID of the document to retrieve.

        Returns:
            Optional[Tuple[List[float], Dict[str, Any]]]: A tuple of (vector, metadata) if found, else None.
        """
        ...


class InMemoryVectorStore:
    """
    A simple in-memory implementation of a vector store for demonstration
    and testing purposes. Uses cosine similarity for search.
    """
    def __init__(self):
        self._vectors: Dict[str, Tuple[np.ndarray, Dict[str, Any]]] = {}

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Computes the cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def add_vector(self, vector: List[float], document_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Adds a vector to the in-memory store."""
        if document_id in self._vectors:
            logger.warning(f"Vector with ID '{document_id}' already exists. Overwriting.")
        self._vectors[document_id] = (np.array(vector, dtype=np.float32), metadata if metadata is not None else {})
        logger.debug(f"Added vector with ID: {document_id}")

    def search(self, query_vector: List[float], k: int = 1) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Performs a cosine similarity search."""
        query_vec_np = np.array(query_vector, dtype=np.float32)
        if not self._vectors:
            return []

        similarities = []
        for doc_id, (vec, meta) in self._vectors.items():
            score = self._cosine_similarity(query_vec_np, vec)
            similarities.append((doc_id, score, meta))

        # Sort by similarity score in descending order
        similarities.sort(key=lambda x: x[1], reverse=True)
        logger.debug(f"Search performed. Top {k} results found.")
        return similarities[:k]

    def get_by_id(self, document_id: str) -> Optional[Tuple[List[float], Dict[str, Any]]]:
        """Retrieves a vector and its metadata by its document_id."""
        item = self._vectors.get(document_id)
        if item:
            return item[0].tolist(), item[1]
        return None

    def __len__(self) -> int:
        """Returns the number of vectors currently stored."""
        return len(self._vectors)


class MockEmbeddingModel:
    """
    A mock embedding model for testing purposes. Generates deterministic
    but non-meaningful embeddings.
    """
    def __init__(self, embed_dim: int = 1536):
        self.embed_dim = embed_dim
        logger.warning("Using MockEmbeddingModel. Embeddings are not semantically meaningful.")

    def embed(self, text: str) -> List[float]:
        """Generates a pseudo-random embedding based on the hash of the text."""
        # A simple way to get a deterministic but different vector for each unique text
        seed = sum(ord(c) for c in text)
        rng = np.random.default_rng(seed)
        return rng.rand(self.embed_dim).tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a batch of texts."""
        return [self.embed(text) for text in texts]


class SemanticRouter:
    """
    The SemanticRouter intelligently dispatches incoming queries to appropriate
    backend components (chains, tools, agents) based on semantic similarity.

    It uses an embedding model to vectorize both the incoming query and
    pre-registered route descriptions, then performs a vector similarity search
    to find the best matching route(s).

    Args:
        embedding_model (BaseEmbeddingModel): An instance of a class implementing
                                              BaseEmbeddingModel, responsible for
                                              generating text embeddings.
        vector_store (BaseVectorStore, optional): An instance of a class implementing
                                                  BaseVectorStore. Defaults to
                                                  `InMemoryVectorStore`.
        threshold (float, optional): A minimum similarity score required for a path
                                     to be considered a valid match. Defaults to 0.7.
    """
    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        vector_store: Optional[BaseVectorStore] = None,
        threshold: float = 0.7,
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store if vector_store is not None else InMemoryVectorStore()
        self.threshold = threshold
        self._registered_paths: Dict[str, RouterPath] = {}
        logger.info(f"SemanticRouter initialized with threshold={self.threshold}")

    def _embed_text(self, text: str) -> List[float]:
        """Internal helper to embed text using the configured embedding model."""
        try:
            return self.embedding_model.embed(text)
        except Exception as e:
            logger.error(f"Failed to embed text '{text[:50]}...': {e}")
            raise

    def add_path(self, path: RouterPath) -> None:
        """
        Registers a new routing path with the router.

        The path's description will be embedded and stored in the vector store
        for future semantic matching.

        Args:
            path (RouterPath): An instance of RouterPath containing the path's
                               name, description, and target_id.

        Raises:
            ValueError: If a path with the same name already exists.
        """
        if path.name in self._registered_paths:
            raise ValueError(f"Path with name '{path.name}' already exists.")

        logger.debug(f"Adding path '{path.name}' with description: '{path.description}'")
        embedding = self._embed_text(path.description)
        path.embedding = embedding  # Store embedding on the path object too
        self.vector_store.add_vector(
            vector=embedding,
            document_id=path.name,
            metadata={
                "target_id": path.target_id,
                "description": path.description,
                **path.metadata
            }
        )
        self._registered_paths[path.name] = path
        logger.info(f"Path '{path.name}' successfully added and embedded.")

    def route(self, query: str, k: int = 1) -> List[Tuple[RouterPath, float]]:
        """
        Routes an incoming query to one or more best-matching paths.

        The query is embedded, and a similarity search is performed against
        the descriptions of registered paths. Paths with a similarity score
        above the configured threshold are returned.

        Args:
            query (str): The natural language query to route.
            k (int): The maximum number of top matching paths to return.

        Returns:
            List[Tuple[RouterPath, float]]: A list of tuples, where each tuple
            contains the matched RouterPath object and its similarity score
            with the query. The list is sorted by score in descending order.
            Returns an empty list if no suitable paths are found above the threshold.
        """
        if not self._registered_paths:
            logger.warning("No paths registered in the SemanticRouter. Returning empty list.")
            return []

        query_embedding = self._embed_text(query)
        results = self.vector_store.search(query_embedding, k=k)

        matched_paths: List[Tuple[RouterPath, float]] = []
        for doc_id, score, metadata in results:
            if score >= self.threshold:
                # Retrieve the full RouterPath object from our internal registry
                # This ensures we return the Pydantic object with all its original fields.
                original_path = self._registered_paths.get(doc_id)
                if original_path:
                    # Create a copy or update the score if needed, but for returning, original is fine
                    matched_paths.append((original_path, score))
                else:
                    logger.warning(f"Vector store returned doc_id '{doc_id}' but it's not in registered paths.")
            else:
                logger.debug(f"Path '{doc_id}' scored {score:.4f} but is below threshold {self.threshold:.4f}.")

        # Ensure results are sorted by score (vector store should do this, but defensive check)
        matched_paths.sort(key=lambda x: x[1], reverse=True)
        
        if matched_paths:
            logger.info(f"Query '{query[:50]}...' routed to: {[(p.name, f'{s:.2f}') for p, s in matched_paths]}")
        else:
            logger.info(f"Query '{query[:50]}...' did not match any path above threshold {self.threshold}.")

        return matched_paths

    def get_path_by_name(self, name: str) -> Optional[RouterPath]:
        """
        Retrieves a registered path by its unique name.

        Args:
            name (str): The name of the path to retrieve.

        Returns:
            Optional[RouterPath]: The RouterPath object if found, otherwise None.
        """
        return self._registered_paths.get(name)

    def list_paths(self) -> List[RouterPath]:
        """
        Returns a list of all currently registered RouterPath objects.

        Returns:
            List[RouterPath]: A list of all registered paths.
        """
        return list(self._registered_paths.values())

    def __len__(self) -> int:
        """Returns the number of registered paths."""
        return len(self._registered_paths)

# Example Usage (typically kept in a separate examples/tests folder, but included here for demonstration)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    # Set the SemanticRouter's logger to DEBUG for more detailed output during execution
    logging.getLogger(__name__).setLevel(logging.DEBUG)

    print("--- Initializing Semantic Router ---")
    # 1. Initialize Components
    mock_embedder = MockEmbeddingModel(embed_dim=768) # Using a common embedding dimension
    in_memory_vec_store = InMemoryVectorStore()
    
    # Initialize the router with a default threshold
    router = SemanticRouter(
        embedding_model=mock_embedder,
        vector_store=in_memory_vec_store,
        threshold=0.6 # A moderately permissive threshold for mock embeddings
    )

    # 2. Define and Add Router Paths
    print("\n--- Registering Router Paths ---")
    path_qa = RouterPath(
        name="documentation_qa",
        description="Handles questions about the Vishustra framework's documentation, features, and usage.",
        target_id="vishustra_docs_chain",
        metadata={"category": "support", "model_context": "docs_qa_model"}
    )
    path_code_gen = RouterPath(
        name="code_generation",
        description="Generates Python code snippets or functions based on user requests, especially for Vishustra components.",
        target_id="vishustra_code_generator_tool",
        metadata={"category": "development", "language": "python"}
    )
    path_sales = RouterPath(
        name="sales_inquiry",
        description="Routes queries related to product pricing, subscriptions, licensing, or commercial inquiries.",
        target_id="sales_agent_handoff_chain",
        metadata={"category": "business", "contact_info": "sales@vishustra.com"}
    )
    path_bug_report = RouterPath(
        name="bug_report",
        description="Collects and processes information about bugs, errors, or unexpected behavior in the framework.",
        target_id="issue_tracker_tool",
        metadata={"category": "support", "priority_level": "medium"}
    )
    path_greetings = RouterPath(
        name="general_greeting",
        description="Handles basic greetings, farewells, and pleasantries.",
        target_id="greeting_response_chain",
        metadata={"category": "conversational"}
    )

    router.add_path(path_qa)
    router.add_path(path_code_gen)
    router.add_path(path_sales)
    router.add_path(path_bug_report)
    router.add_path(path_greetings)

    print(f"\nRouter has {len(router)} paths registered.")
    print("Currently registered paths:")
    for p in router.list_paths():
        print(f"  - '{p.name}': '{p.description[:70]}...'")

    # 3. Route Incoming Queries
    print("\n--- Routing Incoming Queries ---")

    queries = [
        "How do I use the memory buffer component in Vishustra?",
        "Write a Python function to integrate with a new LLM provider.",
        "What are the pricing plans for the enterprise version?",
        "I found a critical bug with the agent's tool execution.",
        "Hello, how are you today?",
        "Tell me about the history of artificial intelligence.", # This should not match well with our paths
        "I need help debugging my agent configuration.",
    ]

    for i, query in enumerate(queries):
        print(f"\nQuery {i+1}: '{query}'")
        matched_paths = router.route(query, k=2) # Get top 2 matches

        if matched_paths:
            for j, (path, score) in enumerate(matched_paths):
                print(f"  Match {j+1}: Path='{path.name}', Score={score:.4f}, Target='{path.target_id}', Metadata={path.metadata}")
        else:
            print("  No suitable path found above the threshold.")

    # Demonstrate retrieving a path by name
    print("\n--- Demonstrating path retrieval by name ---")
    qa_path_retrieved = router.get_path_by_name("documentation_qa")
    if qa_path_retrieved:
        print(f"Successfully retrieved path by name 'documentation_qa': {qa_path_retrieved.description}")
    else:
        print("Failed to retrieve path 'documentation_qa' by name.")

    # Demonstrate error handling for adding duplicate path
    print("\n--- Demonstrating error for duplicate path addition ---")
    try:
        router.add_path(RouterPath(name="documentation_qa", description="Another doc path description.", target_id="dummy_chain"))
    except ValueError as e:
        print(f"Caught expected error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")