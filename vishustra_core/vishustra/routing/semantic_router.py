import abc
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

# Attempt to import OpenAI if available, otherwise provide a fallback
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    # Mock class to avoid NameError if openai isn't installed.
    # A warning is logged when this mock is used.
    class OpenAI:
        def __init__(self, *args, **kwargs):
            logging.warning("OpenAI client not available. Please install 'openai' package for OpenAI embeddings.")
        def embeddings(self):
            class MockEmbeddings:
                def create(self, input, model):
                    raise NotImplementedError("OpenAI client not available. Cannot create embeddings.")
            return MockEmbeddings()

# Pydantic is a core dependency for data models in Vishustra
try:
    from pydantic import BaseModel, Field, PrivateAttr
except ImportError:
    raise ImportError("Pydantic is required for Vishustra. Please install it with 'pip install pydantic'.")

# Numpy for vector operations (dot product, norm)
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logging.warning("NumPy is not available. Some vector operations might be less optimized or unavailable.")


_logger = logging.getLogger(__name__)

# --- Vishustra Core Data Models ---

class Route(BaseModel):
    """
    Represents a potential route or destination within the Vishustra framework.

    A route defines a specific path or action the framework can take, such as
    invoking a particular LLM chain, calling an agent tool, or directing to a
    specific RAG pipeline.

    Attributes:
        name: A unique identifier for the route (e.g., "qa_chain", "customer_service_tool").
        description: A natural language description of what this route does or
                     when it should be used. This description is used for
                     semantic matching.
        metadata: Optional dictionary for additional route-specific parameters
                  or configuration that might be needed by the downstream handler.
    """
    name: str = Field(..., description="Unique identifier for the route.")
    description: str = Field(..., description="Natural language description of the route's purpose.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata for the route.")

    def __hash__(self):
        """Allows Route objects to be used in sets or as dictionary keys based on their name."""
        return hash(self.name)

    def __eq__(self, other):
        """Defines equality for Route objects based on their name."""
        if not isinstance(other, Route):
            return NotImplemented
        return self.name == other.name

class RouterInput(BaseModel):
    """
    Represents the input to the SemanticRouter, typically a user query or system prompt.

    Attributes:
        text: The primary text content to be routed. This will be embedded.
        context: Optional dictionary for additional context that might influence
                 routing decisions or be passed downstream.
    """
    text: str = Field(..., description="The primary text content to be routed.")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context for routing.")

class RouteMatch(BaseModel):
    """
    Represents a successful match found by the SemanticRouter.

    Attributes:
        route: The matched Route object.
        similarity_score: The similarity score (e.g., cosine similarity)
                          between the input and the matched route's description.
    """
    route: Route = Field(..., description="The matched Route object.")
    similarity_score: float = Field(..., description="Similarity score of the match.")

# --- Vishustra Abstract Interfaces (Plugins) ---

class IEmbeddingModel(abc.ABC):
    """
    Abstract Base Class for embedding models.

    Defines the interface for any model capable of converting text into numerical
    vector representations (embeddings).
    """

    @abc.abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a list of text inputs.

        Args:
            texts: A list of strings to be embedded.

        Returns:
            A list of lists of floats, where each inner list is the embedding
            vector for the corresponding input text.
        """
        raise NotImplementedError

class IVectorStore(abc.ABC):
    """
    Abstract Base Class for vector stores.

    Defines the interface for storing and querying vector embeddings, typically
    with associated metadata/payloads.
    """

    @abc.abstractmethod
    def add(self,
            vectors: List[List[float]],
            payloads: List[Dict[str, Any]],
            ids: Optional[List[str]] = None) -> List[str]:
        """
        Adds vectors and their associated payloads to the vector store.

        Args:
            vectors: A list of embedding vectors.
            payloads: A list of dictionaries, where each dictionary corresponds
                      to the metadata for the respective vector.
            ids: Optional list of unique identifiers for the vectors. If None,
                 the store should generate them.

        Returns:
            A list of the IDs assigned to the added vectors.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def query(self,
              vector: List[float],
              top_k: int = 1) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Queries the vector store for the top_k most similar vectors to the
        given query vector.

        Args:
            vector: The query embedding vector.
            top_k: The number of top similar results to retrieve.

        Returns:
            A list of tuples, where each tuple contains (id, similarity_score, payload)
            for the matched vectors, sorted by similarity_score in descending order.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def delete(self, ids: List[str]) -> None:
        """
        Deletes vectors from the store by their IDs.

        Args:
            ids: A list of IDs of vectors to delete.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def clear(self) -> None:
        """
        Clears all vectors and payloads from the vector store.
        """
        raise NotImplementedError

# --- Vishustra Concrete Implementations (Defaults/Examples) ---

class OpenAIEmbeddingModel(IEmbeddingModel):
    """
    An embedding model implementation using OpenAI's API.

    Requires the 'openai' package to be installed and an OpenAI API key
    configured (e.g., via OPENAI_API_KEY environment variable).
    """
    def __init__(self, model_name: str = "text-embedding-ada-002", api_key: Optional[str] = None):
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "The 'openai' package is not installed. "
                "Please install it with 'pip install openai' to use OpenAIEmbeddingModel."
            )
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        _logger.info(f"Initialized OpenAIEmbeddingModel with model: {self.model_name}")

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings using OpenAI's API.
        """
        if not texts:
            return []
        try:
            response = self.client.embeddings.create(input=texts, model=self.model_name)
            embeddings = [data.embedding for data in response.data]
            return embeddings
        except Exception as e:
            _logger.error(f"Error generating OpenAI embeddings: {e}")
            raise

class DummyEmbeddingModel(IEmbeddingModel):
    """
    A placeholder embedding model for testing or when no real model is configured.
    Returns fixed-length random vectors.
    """
    def __init__(self, embedding_dimension: int = 1536):
        self.embedding_dimension = embedding_dimension
        _logger.warning(f"Initialized DummyEmbeddingModel. This model returns random embeddings and is for testing only.")

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generates random embeddings.
        """
        if not NUMPY_AVAILABLE:
            _logger.error("NumPy is not installed. DummyEmbeddingModel cannot generate random vectors.")
            raise RuntimeError("NumPy is required for DummyEmbeddingModel when generating random vectors.")
        return [np.random.rand(self.embedding_dimension).tolist() for _ in texts]

class InMemoryVectorStore(BaseModel, IVectorStore):
    """
    A simple, in-memory vector store implementation for demonstration and testing.
    Not suitable for production environments with large datasets or persistent storage needs.
    """
    _vectors: Dict[str, List[float]] = PrivateAttr(default_factory=dict)
    _payloads: Dict[str, Dict[str, Any]] = PrivateAttr(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True # Allow instances of IVectorStore as fields

    def __init__(self, **data):
        super().__init__(**data)
        self._vectors = {}
        self._payloads = {}
        _logger.info("Initialized InMemoryVectorStore.")

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calculates the cosine similarity between two vectors."""
        if not NUMPY_AVAILABLE:
            raise RuntimeError("NumPy is required for vector operations in InMemoryVectorStore.")
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0 # Handle zero vectors gracefully
        return float(dot_product / (norm_v1 * norm_v2)) # Ensure float return type

    def add(self,
            vectors: List[List[float]],
            payloads: List[Dict[str, Any]],
            ids: Optional[List[str]] = None) -> List[str]:
        """
        Adds vectors to the in-memory store.
        """
        if not NUMPY_AVAILABLE:
            raise RuntimeError("NumPy is required for vector operations in InMemoryVectorStore.")

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in vectors]
        elif len(ids) != len(vectors):
            raise ValueError("Length of 'ids' must match length of 'vectors'.")
        if len(payloads) != len(vectors):
            raise ValueError("Length of 'payloads' must match length of 'vectors'.")

        added_ids = []
        for i, vec in enumerate(vectors):
            _id = ids[i]
            if _id in self._vectors:
                _logger.warning(f"Vector with ID '{_id}' already exists. Overwriting.")
            self._vectors[_id] = vec
            self._payloads[_id] = payloads[i]
            added_ids.append(_id)
        _logger.debug(f"Added {len(added_ids)} vectors to InMemoryVectorStore.")
        return added_ids

    def query(self,
              vector: List[float],
              top_k: int = 1) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Queries the in-memory store for similar vectors using cosine similarity.
        """
        if not NUMPY_AVAILABLE:
            raise RuntimeError("NumPy is required for vector operations in InMemoryVectorStore.")

        if not self._vectors:
            return []

        similarities = []
        for _id, stored_vec in self._vectors.items():
            score = self._cosine_similarity(vector, stored_vec)
            similarities.append((_id, score, self._payloads[_id]))

        similarities.sort(key=lambda x: x[1], reverse=True)
        _logger.debug(f"Queried InMemoryVectorStore, found {len(similarities)} results.")
        return similarities[:top_k]

    def delete(self, ids: List[str]) -> None:
        """
        Deletes vectors from the in-memory store by their IDs.
        """
        deleted_count = 0
        for _id in ids:
            if _id in self._vectors:
                del self._vectors[_id]
                del self._payloads[_id]
                deleted_count += 1
        _logger.debug(f"Deleted {deleted_count} vectors from InMemoryVectorStore.")

    def clear(self) -> None:
        """
        Clears all data from the in-memory store.
        """
        self._vectors.clear()
        self._payloads.clear()
        _logger.info("Cleared InMemoryVectorStore.")

# --- Vishustra Core Component: SemanticRouter ---

class SemanticRouter(BaseModel):
    """
    The SemanticRouter is a core Vishustra component for dynamically directing
    user queries or system prompts to the most appropriate backend processing
    route (e.g., an LLM chain, a tool, a RAG pipeline).

    It uses an embedding model to convert input text and route descriptions
    into vector embeddings, and a vector store to find the semantically
    closest matching route.

    Attributes:
        embedding_model: An instance of an IEmbeddingModel for generating embeddings.
        vector_store: An instance of an IVectorStore for storing and querying route embeddings.
        routes: A list of Route objects that this router can potentially direct to.
        similarity_threshold: The minimum cosine similarity score required for a
                              route to be considered a match. Routes below this
                              threshold will not be returned.
    """
    embedding_model: IEmbeddingModel = Field(..., description="The embedding model to use.")
    vector_store: IVectorStore = Field(..., description="The vector store to index and query routes.")
    routes: List[Route] = Field(default_factory=list, description="The list of available routes.")
    similarity_threshold: float = Field(0.75, ge=0.0, le=1.0, description="Minimum similarity score for a match.")

    _route_name_to_id: Dict[str, str] = PrivateAttr(default_factory=dict)
    _id_to_route: Dict[str, Route] = PrivateAttr(default_factory=dict)

    class Config:
        """Pydantic configuration for SemanticRouter."""
        arbitrary_types_allowed = True # Allow IEmbeddingModel and IVectorStore instances

    def __init__(self, **data):
        super().__init__(**data)
        self._route_name_to_id = {}
        self._id_to_route = {}
        if self.routes:
            _logger.info(f"Initializing SemanticRouter with {len(self.routes)} predefined routes.")
            self._index_routes(self.routes)

    def _index_routes(self, new_routes: List[Route]) -> None:
        """Helper to embed and add routes to the vector store."""
        if not new_routes:
            return

        descriptions = [route.description for route in new_routes]
        try:
            route_embeddings = self.embedding_model.embed(descriptions)
        except Exception as e:
            _logger.error(f"Failed to generate embeddings for routes: {e}")
            raise

        payloads = []
        route_ids = []
        routes_to_index = []

        for i, route in enumerate(new_routes):
            if route.name in self._route_name_to_id:
                _logger.warning(f"Route '{route.name}' already indexed. Skipping re-indexing.")
                continue

            # Store route name in payload to retrieve the full Route object later
            payloads.append({"route_name": route.name})
            # Generate a consistent ID using UUID5 (name-based)
            route_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, route.name))
            route_ids.append(route_id)
            routes_to_index.append(route)

            self._route_name_to_id[route.name] = route_id
            self._id_to_route[route_id] = route

        if routes_to_index:
            self.vector_store.add(vectors=route_embeddings, payloads=payloads, ids=route_ids)
            _logger.info(f"Indexed {len(routes_to_index)} new routes into the vector store.")
        else:
            _logger.info("No new unique routes to index.")

    def add_route(self, route: Route) -> None:
        """
        Adds a new route to the router. The route's description will be embedded
        and added to the vector store.
        """
        if route.name in self._route_name_to_id:
            _logger.warning(f"Route with name '{route.name}' already exists. Skipping add operation.")
            return

        self.routes.append(route) # Add to the Pydantic field list
        self._index_routes([route])
        _logger.info(f"Added new route: '{route.name}'.")

    def remove_route(self, route_name: str) -> None:
        """
        Removes a route from the router by its name.
        """
        if route_name not in self._route_name_to_id:
            _logger.warning(f"Route with name '{route_name}' not found. Cannot remove.")
            return

        route_id = self._route_name_to_id[route_name]
        try:
            self.vector_store.delete([route_id])
        except Exception as e:
            _logger.error(f"Failed to delete route '{route_name}' from vector store: {e}")
            raise

        # Remove from internal tracking maps and Pydantic field list
        self.routes = [r for r in self.routes if r.name != route_name]
        del self._route_name_to_id[route_name]
        del self._id_to_route[route_id]

        _logger.info(f"Removed route: '{route_name}'.")

    def determine_route(self, input_data: RouterInput) -> Optional[RouteMatch]:
        """
        Determines the most semantically relevant route for a given input.

        The input text is embedded, and then the vector store is queried
        to find the most similar route description. If a match is found
        above the `similarity_threshold`, a `RouteMatch` object is returned.

        Args:
            input_data: The `RouterInput` containing the text to be routed.

        Returns:
            An optional `RouteMatch` object if a suitable route is found,
            otherwise None.
        """
        if not self._id_to_route:
            _logger.warning("No routes are indexed in the SemanticRouter. Returning None.")
            return None

        try:
            input_embedding = self.embedding_model.embed([input_data.text])[0]
        except Exception as e:
            _logger.error(f"Failed to embed input text: {e}")
            return None

        # Query the vector store for the top match
        query_results = self.vector_store.query(input_embedding, top_k=1)

        if not query_results:
            _logger.debug(f"No matching routes found for input: '{input_data.text[:50]}...'")
            return None

        # Extract the best result
        best_id, best_score, best_payload = query_results[0]
        matched_route_name = best_payload.get("route_name")

        if matched_route_name is None:
            _logger.error(f"Payload for route ID {best_id} missing 'route_name'. Cannot retrieve route.")
            return None

        if best_score >= self.similarity_threshold:
            matched_route = self._id_to_route.get(best_id)
            if matched_route:
                _logger.info(f"Successfully routed input to '{matched_route.name}' with score: {best_score:.4f}")
                return RouteMatch(route=matched_route, similarity_score=best_score)
            else:
                _logger.error(f"Route object for ID {best_id} / name '{matched_route_name}' not found in internal cache. This indicates a potential data inconsistency.")
                return None
        else:
            _logger.debug(f"Best matching route '{matched_route_name}' had score {best_score:.4f}, "
                          f"below threshold {self.similarity_threshold:.4f}. No route determined.")
            return None

    def get_available_routes(self) -> List[Route]:
        """
        Returns a copy of the currently available routes in the router.
        """
        return list(self.routes)

    def clear_routes(self) -> None:
        """
        Removes all routes from the router and clears the vector store.
        """
        try:
            self.vector_store.clear()
        except Exception as e:
            _logger.error(f"Failed to clear vector store: {e}")
            raise

        self.routes.clear()
        self._route_name_to_id.clear()
        self._id_to_route.clear()
        _logger.info("All routes cleared from SemanticRouter.")

# --- Example Usage (could be in a separate examples or tests file) ---

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO) # Set log level for demo output

    # 1. Define some routes
    customer_service_route = Route(
        name="customer_service_chain",
        description="Handles queries related to customer support, order status, returns, or technical issues.",
        metadata={"chain_type": "customer_support", "priority": "high"}
    )
    product_info_route = Route(
        name="product_information_tool",
        description="Provides details about Vishustra features, capabilities, pricing, and documentation.",
        metadata={"tool_name": "vishustra_docs_lookup", "access_level": "public"}
    )
    developer_guide_route = Route(
        name="developer_guide_rag",
        description="Assists with code examples, API usage, integration patterns, and best practices for developers.",
        metadata={"rag_index": "dev_docs", "audience": "developer"}
    )
    off_topic_route = Route(
        name="off_topic_response",
        description="Responds to queries that are irrelevant, conversational, or outside the scope of Vishustra functionality.",
        metadata={"response_type": "fallback", "mood": "polite"}
    )

    all_routes = [customer_service_route, product_info_route, developer_guide_route, off_topic_route]

    # 2. Instantiate Embedding Model and Vector Store
    # For this demo, we'll use a DummyEmbeddingModel unless OpenAI is truly set up.
    embedding_model_instance: IEmbeddingModel
    if OPENAI_AVAILABLE:
        try:
            # Requires OPENAI_API_KEY env var or explicit api_key parameter
            embedding_model_instance = OpenAIEmbeddingModel()
            _logger.info("Using OpenAIEmbeddingModel for the demo.")
        except Exception as e:
            _logger.warning(f"Could not initialize OpenAIEmbeddingModel ({e}). Falling back to DummyEmbeddingModel.")
            embedding_model_instance = DummyEmbeddingModel()
            logging.getLogger().setLevel(logging.DEBUG) # Dummy might need more debug for clarity
    else:
        embedding_model_instance = DummyEmbeddingModel()
        _logger.warning("Using DummyEmbeddingModel for the demo as OpenAI is not available.")
        logging.getLogger().setLevel(logging.DEBUG) # Dummy might need more debug for clarity


    vector_store_instance = InMemoryVectorStore()

    # 3. Create the SemanticRouter
    router = SemanticRouter(
        embedding_model=embedding_model_instance,
        vector_store=vector_store_instance,
        routes=all_routes,
        similarity_threshold=0.7 # Adjust based on embedding model quality
    )

    print("\n--- Vishustra Semantic Router Demo ---")

    # Test queries
    queries = [
        "How can I check the status of my recent order?",  # Customer Service
        "Tell me about the new features in Vishustra v2.0.",  # Product Info
        "I need an example of how to integrate a custom tool with Vishustra agents.",  # Developer Guide
        "What's the weather like today?",  # Off-topic
        "Help me reset my account password.", # Customer Service
        "Show me the documentation for the routing module.", # Developer Guide (or Product Info, depends on descriptions)
        "What are the pricing plans for your services?", # Product Info
        "Can you tell me a joke?", # Off-topic
        "I have a technical problem with my deployment." # Customer Service
    ]

    for i, query in enumerate(queries):
        print(f"\nQuery {i+1}: '{query}'")
        matched_route = router.determine_route(RouterInput(text=query))

        if matched_route:
            print(f"  -> Matched Route: {matched_route.route.name}")
            print(f"     Description: '{matched_route.route.description}'")
            print(f"     Similarity Score: {matched_route.similarity_score:.4f}")
            print(f"     Metadata: {matched_route.route.metadata}")
        else:
            print("  -> No suitable route found.")

    # Demonstrate adding a new route dynamically
    print("\n--- Adding a new route dynamically ---")
    new_route = Route(
        name="billing_inquiries",
        description="Handles questions about invoices, payments, subscriptions, or billing cycles.",
        metadata={"chain_type": "billing_handler"}
    )
    router.add_route(new_route)

    query_new = "I have a question about my latest invoice."
    print(f"\nQuery for new route: '{query_new}'")
    matched_route_new = router.determine_route(RouterInput(text=query_new))
    if matched_route_new:
        print(f"  -> Matched Route: {matched_route_new.route.name}")
        print(f"     Similarity Score: {matched_route_new.similarity_score:.4f}")
    else:
        print("  -> No suitable route found for new query.")

    # Demonstrate removing a route
    print("\n--- Removing a route dynamically ---")
    router.remove_route("off_topic_response")
    query_removed = "What's up?"
    print(f"\nQuery after removing 'off_topic_response': '{query_removed}'")
    matched_route_removed = router.determine_route(RouterInput(text=query_removed))
    if matched_route_removed:
        print(f"  -> Matched Route: {matched_route_removed.route.name}")
        print(f"     Similarity Score: {matched_route_removed.similarity_score:.4f}")
    else:
        print("  -> No suitable route found (as expected).")

    # Demonstrate clearing all routes
    print("\n--- Clearing all routes ---")
    router.clear_routes()
    query_cleared = "Any query"
    print(f"\nQuery after clearing all routes: '{query_cleared}'")
    matched_route_cleared = router.determine_route(RouterInput(text=query_cleared))
    if matched_route_cleared:
        print(f"  -> Matched Route: {matched_route_cleared.route.name}")
    else:
        print("  -> No suitable route found (as expected).")

    print("\nVishustra Semantic Router Demo Complete.")