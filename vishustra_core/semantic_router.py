import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

try:
    import numpy as np
    from scipy.spatial.distance import cosine
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    # This block allows the module to be imported even if heavy optional dependencies
    # are not installed, but functions relying on them will raise a specific error.
    # In a full framework, dependency management would be more centralized.
    _missing_deps_error = str(e)
    np = None
    cosine = None
    SentenceTransformer = None
    logging.warning(
        f"Missing optional dependencies for semantic_router: {e}. "
        "Some functionalities may not be available. "
        "Please install numpy, scipy, and sentence-transformers for full functionality."
    )

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Route:
    """
    Represents a specific route or scenario within the Vishustra framework.

    Each route defines a particular intent or context that the system can handle,
    and maps it to a target component responsible for processing that intent.

    Attributes:
        name (str): A unique identifier for the route (e.g., "customer_support", "sales_inquiry").
        description (str): A natural language description of what this route handles.
                           This description is crucial for the SemanticRouter to
                           semantically match incoming queries.
        target_component_id (str): The ID of the Vishustra component (e.g., agent, chain, LLM, tool)
                                   that should handle this route if matched.
        metadata (dict): Optional additional key-value pairs associated with the route,
                         allowing for flexible extensibility (e.g., required permissions,
                         specific LLM parameters).
    """
    name: str
    description: str
    target_component_id: str
    metadata: dict = field(default_factory=dict)


class EmbeddingModel(ABC):
    """
    Abstract Base Class for embedding models.

    Defines the standard interface for any model capable of converting text into
    numerical vector representations (embeddings). This abstraction allows
    plugging in different embedding providers (e.g., local, OpenAI, Cohere)
    without modifying the core router logic.
    """

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a list of text inputs.

        Args:
            texts (List[str]): A list of strings to embed.

        Returns:
            List[List[float]]: A list of embedding vectors, where each vector is
                               a list of floats. The order of embeddings corresponds
                               to the order of input texts.
        """
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """
        Returns the expected dimension of the embedding vectors produced by this model.

        Returns:
            int: The dimensionality (number of features) of the embeddings.
        """
        pass


class SentenceTransformerEmbeddingModel(EmbeddingModel):
    """
    Concrete implementation of EmbeddingModel using Sentence Transformers.

    Leverages pre-trained models from the 'sentence-transformers' library to
    generate high-quality, dense text embeddings locally. This is a common
    choice for efficient semantic similarity tasks.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initializes the SentenceTransformerEmbeddingModel.

        Args:
            model_name (str): The name of the Sentence Transformer model to load
                              from Hugging Face. Defaults to "all-MiniLM-L6-v2"
                              due to its balance of performance and efficiency.
        Raises:
            ImportError: If 'sentence_transformers' or 'numpy' is not installed.
            Exception: For any issues during model loading.
        """
        if SentenceTransformer is None or np is None:
            raise ImportError(
                f"SentenceTransformerEmbeddingModel requires 'sentence-transformers' and 'numpy'. "
                f"Please install them via 'pip install sentence-transformers numpy'. "
                f"Original error: {_missing_deps_error}"
            )
        try:
            logger.info(f"Loading SentenceTransformer model: {model_name}...")
            self._model = SentenceTransformer(model_name)
            self._embedding_dimension = self._model.get_sentence_embedding_dimension()
            logger.info(f"SentenceTransformer model '{model_name}' loaded successfully. "
                        f"Embedding dimension: {self._embedding_dimension}")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model '{model_name}': {e}")
            raise

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a list of text inputs using the loaded model.

        Args:
            texts (List[str]): A list of strings to embed.

        Returns:
            List[List[float]]: A list of embedding vectors. Returns an empty list
                               if the input texts list is empty.
        """
        if not texts:
            return []
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def get_embedding_dimension(self) -> int:
        """
        Returns the dimension of the embedding vectors produced by this model.
        """
        return self._embedding_dimension


class VectorStore(ABC):
    """
    Abstract Base Class for a vector store.

    Defines the standard interface for any component capable of efficiently
    storing and searching vector embeddings. This allows for interchangeable
    backend vector databases (e.g., FAISS, Qdrant, Milvus, ChromaDB).
    """

    @abstractmethod
    def add_vectors(self, vectors: List[List[float]], metadata: List[dict]):
        """
        Adds vectors and their associated metadata to the store.

        Args:
            vectors (List[List[float]]): A list of embedding vectors to add.
            metadata (List[dict]): A list of dictionaries, where each dictionary
                                   corresponds to the metadata of the respective vector.
                                   The order must match 'vectors'.
        """
        pass

    @abstractmethod
    def similarity_search(self, query_vector: List[float], k: int = 1) -> List[Tuple[dict, float]]:
        """
        Performs a similarity search to find the top 'k' most similar vectors.

        Args:
            query_vector (List[float]): The embedding vector of the query to search for.
            k (int): The number of top similar results to retrieve.

        Returns:
            List[Tuple[dict, float]]: A list of tuples, where each tuple contains
                                      the metadata of a matched vector and its
                                      similarity score (e.g., cosine similarity).
                                      Higher score indicates greater similarity.
                                      The list is sorted by similarity in descending order.
        """
        pass


class InMemoryVectorStore(VectorStore):
    """
    A lightweight, in-memory vector store implementation.

    This store is suitable for development, testing, or scenarios with a small
    number of routes where persistence and high-scale performance are not critical.
    It uses NumPy for vector operations and SciPy for cosine similarity calculation.

    For production environments with many routes or high query loads, a dedicated
    vector database (e.g., FAISS, Qdrant, Milvus) is strongly recommended
    due to its optimized indexing and search capabilities.
    """

    def __init__(self):
        """Initializes the InMemoryVectorStore."""
        if np is None or cosine is None:
            raise ImportError(
                f"InMemoryVectorStore requires 'numpy' and 'scipy'. "
                f"Please install them via 'pip install numpy scipy'. "
                f"Original error: {_missing_deps_error}"
            )
        self._vectors: List[np.ndarray] = []
        self._metadata: List[dict] = []
        logger.info("InMemoryVectorStore initialized.")

    def add_vectors(self, vectors: List[List[float]], metadata: List[dict]):
        """
        Adds vectors and their associated metadata to the store.

        Args:
            vectors (List[List[float]]): A list of embedding vectors to add.
            metadata (List[dict]): A list of dictionaries, where each dictionary
                                   corresponds to the metadata of the respective vector.
        Raises:
            ValueError: If the number of vectors and metadata entries do not match.
        """
        if len(vectors) != len(metadata):
            raise ValueError("Number of vectors must match the number of metadata entries.")

        for vec, meta in zip(vectors, metadata):
            self._vectors.append(np.array(vec, dtype=np.float32))
            self._metadata.append(meta)
        logger.debug(f"Added {len(vectors)} vectors to store. Total vectors: {len(self._vectors)}")

    def similarity_search(self, query_vector: List[float], k: int = 1) -> List[Tuple[dict, float]]:
        """
        Performs a similarity search to find the top 'k' most similar vectors.

        Calculates cosine similarity between the query vector and all stored vectors.
        Cosine similarity ranges from -1 (opposite) to 1 (identical). Higher scores
        indicate greater similarity.

        Args:
            query_vector (List[float]): The embedding vector of the query.
            k (int): The number of top similar results to retrieve.

        Returns:
            List[Tuple[dict, float]]: A list of tuples (metadata, similarity_score).
                                      Returns an empty list if the store is empty.
        """
        if not self._vectors:
            return []

        query_vec_np = np.array(query_vector, dtype=np.float32)
        similarities = []

        for i, stored_vec in enumerate(self._vectors):
            try:
                # Cosine distance returns 0 for identical vectors, 1 for orthogonal.
                # Cosine similarity is 1 - cosine distance.
                similarity = 1 - cosine(query_vec_np, stored_vec)
                similarities.append((self._metadata[i], float(similarity))) # Ensure float for JSON serialization if needed
            except ValueError as e:
                logger.error(f"Error calculating similarity for vector {i}: {e}. Skipping this vector.")
                continue

        # Sort by similarity in descending order
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:k]


class SemanticRouter:
    """
    The SemanticRouter intelligently dispatches incoming queries to appropriate
    Vishustra components based on the semantic meaning of the query.

    It operates by embedding user queries and a predefined set of route descriptions.
    It then uses a vector store to find the semantically closest route, enabling
    highly modular and dynamic orchestration of LLM applications without
    relying on rigid keyword matching or complex rule engines.
    """

    def __init__(
        self,
        routes: List[Route],
        embedding_model: Optional[EmbeddingModel] = None,
        vector_store: Optional[VectorStore] = None,
        similarity_threshold: float = 0.75
    ):
        """
        Initializes the SemanticRouter.

        Args:
            routes (List[Route]): A list of predefined Route objects that the
                                  router can dispatch to. Each route defines an
                                  intent and its corresponding handler.
            embedding_model (Optional[EmbeddingModel]): The embedding model to use
                                                        for converting text to vectors.
                                                        If None, defaults to
                                                        SentenceTransformerEmbeddingModel
                                                        ("all-MiniLM-L6-v2").
            vector_store (Optional[VectorStore]): The vector store to use for
                                                  storing and searching route embeddings.
                                                  If None, defaults to InMemoryVectorStore.
            similarity_threshold (float): The minimum cosine similarity score required
                                          for a route to be considered a match.
                                          Scores range from -1 (opposite) to 1 (identical).
                                          A higher threshold means stricter matching.
                                          Defaults to 0.75.
        Raises:
            ValueError: If an empty list of routes is provided.
            ImportError: If required dependencies for default models are missing.
        """
        if not routes:
            raise ValueError("SemanticRouter must be initialized with at least one route.")

        self._embedding_model = embedding_model or SentenceTransformerEmbeddingModel()
        self._vector_store = vector_store or InMemoryVectorStore()

        if not (0 <= similarity_threshold <= 1):
            logger.warning(f"Similarity threshold {similarity_threshold} is outside "
                           "the typical range [0, 1] for cosine similarity. "
                           "Values outside this range might lead to unexpected behavior.")
        self._similarity_threshold = similarity_threshold

        # Store routes in a map for quick lookup by name after a match
        self._routes_map: dict[str, Route] = {route.name: route for route in routes}

        logger.info(f"SemanticRouter initializing with {len(routes)} routes.")
        self._load_routes(routes)
        logger.info(f"SemanticRouter ready. Default similarity threshold: {similarity_threshold}")

    def _load_routes(self, routes: List[Route]):
        """
        Internal method to embed route descriptions and add them to the vector store.
        """
        descriptions = [route.description for route in routes]
        route_names = [route.name for route in routes]

        if not descriptions:
            logger.warning("No route descriptions to embed.")
            return

        logger.debug(f"Embedding {len(descriptions)} route descriptions...")
        embeddings = self._embedding_model.embed(descriptions)
        logger.debug("Route descriptions embedded.")

        # Prepare metadata for the vector store search results
        metadata = [{"route_name": name} for name in route_names]
        self._vector_store.add_vectors(embeddings, metadata)
        logger.info(f"Added {len(embeddings)} route embeddings to vector store.")

    def route(self, query: str) -> Optional[Route]:
        """
        Routes an incoming query to the most semantically similar predefined route.

        This method first embeds the query, then performs a similarity search
        against all stored route descriptions. If a route is found with a
        similarity score above the configured threshold, that route is returned.

        Args:
            query (str): The user's input query or an internal prompt that needs
                         to be routed to a specific Vishustra component.

        Returns:
            Optional[Route]: The matched Route object if a sufficiently similar
                             route is found (i.e., above `similarity_threshold`),
                             otherwise None.
        """
        if not query or not query.strip():
            logger.warning("Received an empty or whitespace-only query, cannot route.")
            return None

        logger.debug(f"Attempting to route query: '{query[:75]}{'...' if len(query) > 75 else ''}'")
        try:
            query_embedding = self._embedding_model.embed([query])[0]
        except Exception as e:
            logger.error(f"Failed to embed query '{query[:50]}...': {e}")
            return None

        search_results = self._vector_store.similarity_search(query_embedding, k=1)

        if not search_results:
            logger.info("No routes found in vector store to compare against.")
            return None

        matched_metadata, similarity = search_results[0]
        matched_route_name = matched_metadata.get("route_name")

        if matched_route_name is None:
            logger.error("Vector store returned a match without 'route_name' in metadata. Skipping.")
            return None

        if similarity >= self._similarity_threshold:
            matched_route = self._routes_map.get(matched_route_name)
            if matched_route:
                logger.info(
                    f"Query matched route '{matched_route.name}' "
                    f"with similarity {similarity:.4f} (threshold: {self._similarity_threshold:.2f}). "
                    f"Target component: '{matched_route.target_component_id}'"
                )
                return matched_route
            else:
                logger.error(f"Matched route name '{matched_route_name}' not found in internal routes map. "
                             "This indicates an inconsistency.")
                return None
        else:
            logger.info(
                f"No route found above similarity threshold {self._similarity_threshold:.2f}. "
                f"Best match was '{matched_route_name}' with similarity {similarity:.4f}."
            )
            return None


if __name__ == "__main__":
    # Example Usage for SemanticRouter
    try:
        # Define a set of routes that Vishustra can handle
        example_routes = [
            Route(
                name="customer_support",
                description="Answer questions about product features, technical issues, troubleshooting, bugs.",
                target_component_id="support_agent"
            ),
            Route(
                name="sales_inquiry",
                description="Handle questions about pricing, subscriptions, different plans, purchasing, upgrades.",
                target_component_id="sales_pipeline_tool"
            ),
            Route(
                name="documentation_search",
                description="Find information in our internal knowledge base, user manuals, or official documentation.",
                target_component_id="docs_retriever"
            ),
            Route(
                name="general_chat",
                description="Engage in general conversation, small talk, chit-chat, or off-topic discussions.",
                target_component_id="chat_llm"
            ),
            Route(
                name="refund_request",
                description="Process requests for refunds, cancellations, billing adjustments, or account closure.",
                target_component_id="billing_system_tool"
            ),
            Route(
                name="api_documentation",
                description="Provide details, examples, or usage guides for our API endpoints and SDKs.",
                target_component_id="api_doc_agent"
            ),
        ]

        # Initialize the router.
        # For a real application, you might inject specific `EmbeddingModel`
        # and `VectorStore` instances (e.g., OpenAI embeddings with Qdrant).
        # We'll use the defaults here for simplicity (SentenceTransformer + InMemory).
        router = SemanticRouter(routes=example_routes, similarity_threshold=0.7)

        test_queries = [
            "What's the cost of the enterprise plan?",                      # -> sales_inquiry
            "My application is crashing when I open it, help!",             # -> customer_support
            "Can I get a refund for my last month's subscription?",        # -> refund_request
            "Tell me about the weather today.",                             # -> general_chat (or None)
            "Where can I find the API reference?",                          # -> api_documentation
            "How do I integrate with your platform using Python?",          # -> api_documentation
            "I need to cancel my annual membership.",                       # -> refund_request
            "What features does the standard package include?",             # -> sales_inquiry
            "How to troubleshoot network issues?",                          # -> customer_support
            "Where is the guide for getting started?",                      # -> documentation_search
            "Hello, how are you?",                                          # -> general_chat
            "I want to know the specifications of product X.",              # -> customer_support
            "What is the average rainfall in Seattle in October?"           # -> No suitable route
        ]

        print("\n--- Semantic Router Test Cases ---")
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            matched_route = router.route(query)
            if matched_route:
                print(f"  -> Matched Route: '{matched_route.name}'")
                print(f"     Target Component: '{matched_route.target_component_id}'")
                # print(f"     Route Description: '{matched_route.description}'")
            else:
                print("  -> No suitable route found above the similarity threshold.")

        # Test with an empty query
        print("\n--- Testing with empty query ---")
        matched_route = router.route("   ")
        if matched_route:
            print(f"Empty query -> Matched: '{matched_route.name}' (unexpected, should be None)")
        else:
            print("Empty query -> No match (as expected).")

        # Test with no routes (should raise ValueError during initialization)
        print("\n--- Testing router initialization with no routes (expected ValueError) ---")
        try:
            _ = SemanticRouter(routes=[])
        except ValueError as e:
            print(f"Caught expected error: {e}")
        except Exception as e:
            print(f"Caught unexpected error type: {type(e).__name__}: {e}")

    except ImportError as e:
        print(f"\nERROR: Could not run example due to missing dependencies: {e}")
        print("Please install required libraries: pip install numpy scipy sentence-transformers")
    except Exception as e:
        print(f"\nAn unexpected error occurred during example execution: {type(e).__name__}: {e}")