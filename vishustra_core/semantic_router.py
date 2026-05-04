import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, Tuple

from pydantic import BaseModel, Field, ValidationError

# Configure logging for the module
logger = logging.getLogger(__name__)
# Set default log level, can be overridden by framework configuration
logger.setLevel(logging.INFO) 
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not logger.handlers: # Prevent adding multiple handlers if file is reloaded
    logger.addHandler(handler)


# --- Protocols for Abstraction ---

class EmbeddingModelProtocol(Protocol):
    """
    Protocol for an embedding model within Vishustra.
    Assumes a synchronous text embedding method.
    """
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of texts into a list of embedding vectors.

        Args:
            texts: A list of strings to embed.

        Returns:
            A list of lists of floats, where each inner list is an embedding vector.
            The outer list corresponds to the input texts, and each inner list
            is the embedding vector for a single text.
        """
        ...

class VectorStoreProtocol(Protocol):
    """
    Protocol for a vector store within Vishustra.
    Assumes basic add and similarity search functionalities.
    """
    @abstractmethod
    def add(self, vectors: List[List[float]], metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Adds vectors to the store with optional associated metadata.
        Each entry in `vectors` corresponds to an entry in `metadatas`.

        Args:
            vectors: A list of embedding vectors.
            metadatas: Optional list of dictionaries, where each dictionary
                       contains metadata for a corresponding vector.
                       Metadata should include at least a unique identifier
                       like 'route_name' to retrieve the original item.
        """
        ...

    @abstractmethod
    def search(self, query_vector: List[float], k: int = 1) -> List[Tuple[Dict[str, Any], float]]:
        """
        Performs a similarity search for the query vector.

        Args:
            query_vector: The vector to search for.
            k: The number of nearest neighbors to return.

        Returns:
            A list of tuples, where each tuple contains (metadata, similarity_score).
            The metadata dictionary should ideally include the original 'route_name'
            or a similar identifier stored during the `add` operation.
            Results are ordered by similarity score in descending order.
        """
        ...

    @abstractmethod
    def clear(self) -> None:
        """Clears all vectors and metadata from the store."""
        ...

    # Future enhancement: `delete` method by ID or metadata for proper route updates.
    # @abstractmethod
    # def delete(self, metadata_key: str, metadata_value: Any) -> None:
    #     """Deletes vectors based on a metadata key-value pair."""
    #     ...


# --- Data Models ---

class Route(BaseModel):
    """
    Represents a specific route or pathway in the Vishustra framework.
    This model defines the structure for semantic routing rules.

    Attributes:
        name (str): A unique identifier for the route (e.g., "customer_service", "code_generation").
        description (str): A natural language description of what this route handles.
                           This description is crucial for semantic matching and will be
                           embedded by the EmbeddingModel.
        target (Any): The actual target associated with this route. This could be
                      a function, an object instance, a string identifier for another
                      component (e.g., a chain ID, tool name), or any other relevant data
                      that the router should return when this route is selected.
                      Vishustra's core logic will then interpret this target.
        threshold (float): The minimum similarity score required for this route to be
                           considered a match. If the similarity between the query
                           and this route's description is below this value, the route
                           will not be selected, even if it's the most similar.
                           Defaults to 0.75, must be between 0.0 and 1.0.
    """
    name: str = Field(..., description="Unique identifier for the route.")
    description: str = Field(..., description="Natural language description for semantic matching.")
    target: Any = Field(..., description="The object or identifier associated with this route.")
    threshold: float = Field(0.75, ge=0.0, le=1.0, description="Minimum similarity score to match this route.")

    class Config:
        arbitrary_types_allowed = True # Allow 'target' to be any type


# --- Semantic Router Implementation ---

class SemanticRouter:
    """
    The SemanticRouter intelligently directs incoming queries to the most
    semantically relevant "route" based on a predefined set of route descriptions.

    It leverages an `EmbeddingModelProtocol` to convert user queries and route
    descriptions into vector representations, and a `VectorStoreProtocol` to perform
    efficient similarity searches. This enables dynamic, context-aware routing
    within Vishustra, improving modularity and responsiveness of LLM applications.

    Attributes:
        _embedding_model (EmbeddingModelProtocol): The model used to generate embeddings.
        _vector_store (VectorStoreProtocol): The vector database used for route indexing and search.
        _routes (Dict[str, Route]): A dictionary mapping route names to Route objects,
                                    maintaining the active set of routing rules.
        _default_threshold (float): A global minimum similarity threshold. If no route-specific
                                    threshold is met (or if a route does not explicitly define one),
                                    this global threshold is used as a fallback.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModelProtocol,
        vector_store: VectorStoreProtocol,
        routes: Optional[List[Route]] = None,
        default_threshold: float = 0.70,
    ):
        """
        Initializes the SemanticRouter with essential components and an initial set of routes.

        Args:
            embedding_model: An instance adhering to `EmbeddingModelProtocol`. This model
                             is responsible for converting text into vector embeddings.
            vector_store: An instance adhering to `VectorStoreProtocol`. This store holds
                          the embeddings of route descriptions and facilitates similarity searches.
            routes: An optional list of `Route` objects to register upon initialization.
                    If provided, these routes will be embedded and indexed immediately.
            default_threshold: A global similarity threshold (0.0 to 1.0). If the best
                               matching route's score is below its own `threshold` (or
                               this `default_threshold` if the route doesn't specify one),
                               no route is considered a match. Defaults to 0.70.

        Raises:
            TypeError: If `embedding_model` or `vector_store` do not adhere to their
                       respective protocols.
            ValueError: If `default_threshold` is outside the valid range [0.0, 1.0].
        """
        if not isinstance(embedding_model, EmbeddingModelProtocol):
            raise TypeError("`embedding_model` must adhere to `EmbeddingModelProtocol`.")
        if not isinstance(vector_store, VectorStoreProtocol):
            raise TypeError("`vector_store` must adhere to `VectorStoreProtocol`.")

        try:
            # Validate default_threshold using Pydantic's Field for consistency
            self._default_threshold: float = Field(default_threshold, ge=0.0, le=1.0).default
        except ValidationError as e:
            raise ValueError(f"Invalid `default_threshold` value: {e}")

        self._embedding_model: EmbeddingModelProtocol = embedding_model
        self._vector_store: VectorStoreProtocol = vector_store
        self._routes: Dict[str, Route] = {}
        
        if routes:
            for route in routes:
                self.add_route(route) # This also handles embedding and indexing
        else:
            logger.info("SemanticRouter initialized with no initial routes.")

    def add_route(self, route: Route) -> None:
        """
        Adds a new `Route` to the router. The route's description will be embedded
        and added to the vector store. If a route with the same name already
        exists, it will be updated (its internal definition changed, and its
        embedding re-indexed).

        Args:
            route: The `Route` object to add.

        Raises:
            Exception: If embedding or indexing fails for any reason.
        """
        if route.name in self._routes:
            logger.warning(f"Route '{route.name}' already exists. Overwriting its definition and re-indexing.")
            # In a production vector store, we'd typically delete the old embedding
            # associated with route.name before adding the new one to avoid duplicates
            # or stale data. Given VectorStoreProtocol lacks `delete`, we rely on
            # the underlying vector store to handle potential ID collisions (e.g., if
            # it stores unique IDs and overwrites, or if it's acceptable to have
            # multiple embeddings for the same conceptual route, which will then
            # be handled by `search` returning the most similar).
        
        self._routes[route.name] = route
        self._embed_and_index_route(route)
        logger.info(f"Route '{route.name}' added/updated successfully.")

    def remove_route(self, route_name: str) -> None:
        """
        Removes a route from the router's internal state.

        Note: This implementation primarily removes the route from the router's
        `_routes` dictionary. It does *not* explicitly remove the corresponding
        embedding from the underlying `VectorStoreProtocol` because the protocol
        lacks a `delete` method. For a full cleanup, a `VectorStoreProtocol`
        implementation would need to expose a `delete_by_metadata` or `delete_by_id`
        method, or a full re-indexing of all remaining routes would be required.

        Args:
            route_name: The name of the route to remove.
        """
        if route_name not in self._routes:
            logger.warning(f"Attempted to remove non-existent route '{route_name}'. No action taken.")
            return

        del self._routes[route_name]
        logger.info(f"Route '{route_name}' removed from router's internal state.")
        # Future enhancement: VectorStoreProtocol should have a `delete_by_id` method
        # self._vector_store.delete(metadata_key="route_name", metadata_value=route_name)

    def _embed_and_index_route(self, route: Route) -> None:
        """
        Internal method to embed a route's description and add it to the vector store.
        """
        try:
            embedding = self._embedding_model.embed([route.description])[0]
            self._vector_store.add(
                vectors=[embedding],
                metadatas=[{"route_name": route.name, "description": route.description}]
            )
        except Exception as e:
            logger.error(f"Failed to embed and index route '{route.name}': {e}", exc_info=True)
            raise # Re-raise to ensure calling code knows of the failure

    def _embed_query(self, query: str) -> List[float]:
        """
        Internal method to embed a user query string.
        """
        try:
            return self._embedding_model.embed([query])[0]
        except Exception as e:
            logger.error(f"Failed to embed query: '{query[:100]}...': {e}", exc_info=True)
            raise

    def route(self, query: str, default_target: Optional[Any] = None) -> Tuple[Optional[str], Any, float]:
        """
        Routes an incoming query to the most semantically relevant target based on
        the registered routes.

        The process involves:
        1. Embedding the incoming query.
        2. Searching the vector store for the most similar route description.
        3. Retrieving the full `Route` object for the best match.
        4. Applying the route-specific `threshold` (or the global `default_threshold`)
           to determine if the match is sufficiently strong.
        5. Returning the matched route's name, target, and similarity score,
           or a default if no satisfactory match is found.

        Args:
            query: The user's input query string that needs to be routed.
            default_target: An optional target to return if no registered route meets
                            the similarity threshold. This can be a fallback mechanism,
                            e.g., routing to a generic LLM or an error handler.

        Returns:
            A tuple containing three elements:
            1. `Optional[str]`: The `name` of the matched route, or `None` if no route
                                 met the threshold.
            2. `Any`: The `target` associated with the matched route, or `default_target`
                      if no route was matched.
            3. `float`: The similarity score of the best match (between 0.0 and 1.0),
                        or `0.0` if no match occurred.
        """
        if not self._routes:
            logger.warning("SemanticRouter has no routes registered. Returning default_target.")
            return None, default_target, 0.0

        try:
            query_embedding = self._embed_query(query)
        except Exception:
            logger.error(f"Failed to embed query for routing. Returning default target.")
            return None, default_target, 0.0

        # Search for the top k=1 most similar route in the vector store
        search_results = self._vector_store.search(query_embedding, k=1)

        if not search_results:
            logger.debug(f"No semantic matches found in vector store for query: '{query[:100]}...'.")
            return None, default_target, 0.0

        best_match_metadata, best_score = search_results[0]
        matched_route_name = best_match_metadata.get("route_name")

        if not matched_route_name or matched_route_name not in self._routes:
            # This can happen if vector store has stale data or a corrupted entry
            logger.warning(f"Vector store returned an unknown or stale route_name: '{matched_route_name}'. "
                           f"Best score: {best_score:.4f}. Returning default target.")
            return None, default_target, 0.0

        matched_route = self._routes[matched_route_name]

        # Apply the route-specific threshold, falling back to global default
        effective_threshold = matched_route.threshold
        
        if best_score >= effective_threshold:
            logger.info(f"Query '{query[:50]}...' routed to '{matched_route_name}' "
                        f"with score {best_score:.4f} (threshold {effective_threshold:.2f}).")
            return matched_route.name, matched_route.target, best_score
        else:
            logger.debug(
                f"Best match for query '{query[:50]}...' was '{matched_route_name}' "
                f"with score {best_score:.4f}, which is below its effective threshold {effective_threshold:.2f}. "
                "Returning default target."
            )
            return None, default_target, 0.0

    def get_registered_routes(self) -> Dict[str, Route]:
        """
        Returns a copy of the currently registered routes.
        """
        return self._routes.copy()


# --- Example/Dummy Implementations for Vishustra (for testing/demonstration) ---
# In a full Vishustra framework, these would typically be imported from other
# core modules (e.g., `vishustra.embeddings.openai`, `vishustra.vector_stores.qdrant`).

class DummyEmbeddingModel(EmbeddingModelProtocol):
    """
    A simple dummy embedding model for testing purposes within Vishustra.
    Generates a deterministic (but not semantically meaningful) embedding based
    on the input text. Real models would use a neural network.
    """
    def embed(self, texts: List[str]) -> List[List[float]]:
        logger.debug(f"DummyEmbeddingModel: Embedding {len(texts)} texts.")
        embeddings = []
        for text in texts:
            # Create a simple, deterministic "embedding" for demonstration.
            # In reality, this would be a deep learning model output.
            vector = [float(ord(c)) / 128.0 for c in text[:32]] # Use first 32 chars for dimension
            # Pad with zeros if the text is shorter than 32 characters
            while len(vector) < 32:
                vector.append(0.0)
            embeddings.append(vector)
        return embeddings

class InMemoryVectorStore(VectorStoreProtocol):
    """
    A basic in-memory vector store implementation for demonstration and testing
    within Vishustra. It stores vectors and their associated metadata in lists
    and performs similarity search using cosine similarity.
    """
    def __init__(self):
        self._vectors: List[List[float]] = []
        self._metadatas: List[Dict[str, Any]] = []
        logger.debug("InMemoryVectorStore initialized.")

    def add(self, vectors: List[List[float]], metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        if not vectors:
            return
        if metadatas is not None and len(vectors) != len(metadatas):
            raise ValueError("Number of vectors and metadatas must match.")

        self._vectors.extend(vectors)
        if metadatas is not None:
            self._metadatas.extend(metadatas)
        else:
            # If no metadata provided, append empty dicts
            self._metadatas.extend([{} for _ in vectors])
        logger.debug(f"Added {len(vectors)} vectors to InMemoryVectorStore. Total: {len(self._vectors)}.")

    def search(self, query_vector: List[float], k: int = 1) -> List[Tuple[Dict[str, Any], float]]:
        if not self._vectors:
            return []
        if not query_vector:
            raise ValueError("Query vector cannot be empty.")

        similarities: List[Tuple[int, float]] = []
        for i, stored_vec in enumerate(self._vectors):
            if len(query_vector) != len(stored_vec):
                logger.warning(
                    f"Query vector dimension ({len(query_vector)}) mismatch "
                    f"with stored vector dimension ({len(stored_vec)}) at index {i}. Skipping this vector."
                )
                continue
            score = self._cosine_similarity(query_vector, stored_vec)
            similarities.append((i, score))

        # Sort by similarity score in descending order
        similarities.sort(key=lambda item: item[1], reverse=True)
        
        results: List[Tuple[Dict[str, Any], float]] = []
        for i, score in similarities[:k]:
            results.append((self._metadatas[i], score))
        
        logger.debug(f"Performed search, found {len(results)} top-k results.")
        return results

    def clear(self) -> None:
        self._vectors = []
        self._metadatas = []
        logger.debug("InMemoryVectorStore cleared.")

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """
        Calculates the cosine similarity between two vectors.
        Returns 0.0 for zero vectors to avoid division by zero.
        """
        dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
        magnitude1 = sum(v**2 for v in vec1)**0.5
        magnitude2 = sum(v**2 for v in vec2)**0.5

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0 # Handle cases with zero vectors
        
        # Ensure the score is clamped to [0, 1] as some float precision
        # issues or specific similarity metrics might slightly exceed these bounds.
        # Cosine similarity is [-1, 1], but for semantic similarity, we usually
        # expect non-negative similarity when vectors are roughly in the same cone.
        # If embeddings are normalized, this can be skipped.
        return max(0.0, min(1.0, dot_product / (magnitude1 * magnitude2)))

# Example usage (for internal testing/demonstration, would typically be in test files):
if __name__ == "__main__":
    logger.setLevel(logging.DEBUG) # Show debug messages for example

    # 1. Initialize dummy components
    embedding_model = DummyEmbeddingModel()
    vector_store = InMemoryVectorStore()

    # 2. Define routes
    route_customer_service = Route(
        name="customer_service",
        description="Handles queries about account management, billing, refunds, and support.",
        target="customer_service_chain",
        threshold=0.8
    )
    route_product_info = Route(
        name="product_information",
        description="Provides details about Vishustra features, pricing, and capabilities.",
        target="product_info_tool",
        threshold=0.7
    )
    route_code_generation = Route(
        name="code_generation",
        description="Assists with writing code, debugging, and providing programming examples in Python, Java, etc.",
        target={"type": "agent", "name": "coder_agent"},
        threshold=0.85
    )
    route_general_chat = Route(
        name="general_chat",
        description="Engages in casual conversation, small talk, and general knowledge questions.",
        target="conversational_llm_chain",
        threshold=0.6
    )

    # 3. Initialize Semantic Router
    router = SemanticRouter(
        embedding_model=embedding_model,
        vector_store=vector_store,
        routes=[route_customer_service, route_product_info, route_code_generation],
        default_threshold=0.65
    )

    # Add another route dynamically
    router.add_route(route_general_chat)

    print("\n--- Testing Routing ---")

    test_queries = [
        "I need help with my monthly bill.",
        "What are the pricing plans for Vishustra's enterprise features?",
        "Write a Python function to sort a list of dictionaries.",
        "Tell me a joke.",
        "How do I reset my password?",
        "What is the maximum token limit for embedding models?", # Should go to product_information
        "Can you help me fix this syntax error in my Java code?",
        "Hi there!"
    ]

    for i, query in enumerate(test_queries):
        print(f"\nQuery {i+1}: '{query}'")
        matched_name, target, score = router.route(query, default_target="fallback_to_unmatched_handler")
        print(f"  Routed to: {matched_name}")
        print(f"  Target: {target}")
        print(f"  Similarity Score: {score:.4f}")
        if matched_name is None:
            print("  (No specific route matched above threshold)")

    print("\n--- Testing Route Update ---")
    updated_customer_service = Route(
        name="customer_service",
        description="Dedicated support for account issues, payments, and refunds. Also handles technical support.",
        target="premium_customer_service_chain",
        threshold=0.82
    )
    router.add_route(updated_customer_service) # Update existing route

    query_after_update = "I have a technical problem with my account."
    print(f"\nQuery (after update): '{query_after_update}'")
    matched_name, target, score = router.route(query_after_update)
    print(f"  Routed to: {matched_name}")
    print(f"  Target: {target}")
    print(f"  Similarity Score: {score:.4f}")

    print("\n--- Testing Route Removal ---")
    router.remove_route("general_chat")
    query_after_removal = "Tell me a joke."
    print(f"\nQuery (after removing 'general_chat'): '{query_after_removal}'")
    matched_name, target, score = router.route(query_after_removal, default_target="no_chat_available")
    print(f"  Routed to: {matched_name}")
    print(f"  Target: {target}")
    print(f"  Similarity Score: {score:.4f}")

    print("\n--- Current Registered Routes ---")
    for name, route_obj in router.get_registered_routes().items():
        print(f"  - {name}: {route_obj.description[:50]}...")
