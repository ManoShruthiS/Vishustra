"""
Vishustra's SemanticRouter module provides intelligent, context-aware routing of user queries
to appropriate backend components (chains, agents, tools, etc.).

It works by embedding incoming queries and comparing them semantically against predefined
"routes," each representing a specific intent or functionality. The router then selects
the best-matching route based on semantic similarity, allowing the framework to dynamically
dispatch queries without explicit keyword matching or rigid rule sets.

This module is designed for high modularity, allowing different embedding models and
vector stores to be plugged in.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Protocol
import asyncio
import numpy as np
from pydantic import BaseModel, Field, PrivateAttr

# Optional dependencies for embedding, to be handled gracefully
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
    print("Warning: sentence-transformers not found. Install 'sentence-transformers' to use SentenceTransformerEmbeddingModel.")

# --- Interfaces and Abstract Base Classes ---

class BaseEmbeddingModel(Protocol):
    """
    Protocol for any embedding model used by the SemanticRouter.
    Defines the asynchronous methods required to generate embeddings.
    """
    @abstractmethod
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Asynchronously generates embeddings for a list of texts.

        Args:
            texts: A list of strings to embed.

        Returns:
            A list of embeddings, where each embedding is a list of floats.
        """
        ...

    @abstractmethod
    async def embed_query(self, text: str) -> List[float]:
        """
        Asynchronously generates an embedding for a single query text.

        Args:
            text: The single string query to embed.

        Returns:
            A single embedding as a list of floats.
        """
        ...

class BaseVectorStoreClient(ABC):
    """
    Abstract Base Class for vector store clients.
    Defines the interface for interacting with any vector database.
    """
    @abstractmethod
    async def add_vectors(self, vectors: List[List[float]], metadatas: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """
        Asynchronously adds vectors to the vector store.

        Args:
            vectors: A list of vectors (list of floats).
            metadatas: Optional list of dictionaries, where each dict corresponds
                       to the metadata for a vector.

        Returns:
            A list of unique IDs assigned to the added vectors.
        """
        pass

    @abstractmethod
    async def search(self, query_vector: List[float], k: int = 1) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Asynchronously performs a similarity search in the vector store.

        Args:
            query_vector: The vector to search with.
            k: The number of top similar results to return.

        Returns:
            A list of tuples, where each tuple contains (id, similarity_score, metadata).
        """
        pass

    @abstractmethod
    async def delete_vectors(self, ids: List[str]) -> None:
        """
        Asynchronously deletes vectors from the vector store by their IDs.

        Args:
            ids: A list of vector IDs to delete.
        """
        pass

    @abstractmethod
    async def clear(self) -> None:
        """
        Asynchronously clears all vectors from the store.
        """
        pass

# --- Concrete Implementations (Examples) ---

class SentenceTransformerEmbeddingModel(BaseEmbeddingModel):
    """
    Concrete implementation of BaseEmbeddingModel using the sentence-transformers library.
    This adapter ensures the synchronous `SentenceTransformer.encode` method is run
    asynchronously using a thread pool executor to avoid blocking the event loop.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if SentenceTransformer is None:
            raise ImportError(
                "SentenceTransformerEmbeddingModel requires 'sentence-transformers' to be installed. "
                "Please install it with: pip install sentence-transformers"
            )
        self._model = SentenceTransformer(model_name)
        # Default to CPU to avoid strict GPU dependency, can be configured
        self._model.to("cpu")
        self._embedding_dimension: int = self._model.get_sentence_embedding_dimension()

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """See BaseEmbeddingModel."""
        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: self._model.encode(texts, convert_to_numpy=True).tolist()
        )
        return embeddings

    async def embed_query(self, text: str) -> List[float]:
        """See BaseEmbeddingModel."""
        loop = asyncio.get_running_loop()
        embedding = await loop.run_in_executor(
            None, lambda: self._model.encode([text], convert_to_numpy=True)[0].tolist()
        )
        return embedding

class InMemoryVectorStoreClient(BaseVectorStoreClient):
    """
    A simple, in-memory implementation of BaseVectorStoreClient for testing and
    demonstration purposes. Not suitable for production with large datasets
    due to lack of persistence and scalability.
    """
    def __init__(self):
        self._vectors: Dict[str, np.ndarray] = {}
        self._metadatas: Dict[str, Dict[str, Any]] = {}
        self._next_id: int = 0
        # Use an asyncio Lock to protect shared state during concurrent async operations
        self._lock = asyncio.Lock()

    async def add_vectors(self, vectors: List[List[float]], metadatas: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """See BaseVectorStoreClient."""
        added_ids = []
        async with self._lock:
            for i, vec in enumerate(vectors):
                vec_id = str(self._next_id)
                self._next_id += 1
                self._vectors[vec_id] = np.asarray(vec, dtype=np.float32)
                self._metadatas[vec_id] = metadatas[i] if metadatas and i < len(metadatas) else {}
                added_ids.append(vec_id)
        return added_ids

    async def search(self, query_vector: List[float], k: int = 1) -> List[Tuple[str, float, Dict[str, Any]]]:
        """See BaseVectorStoreClient."""
        query_vec_np = np.asarray(query_vector, dtype=np.float32)
        results: List[Tuple[str, float, Dict[str, Any]]] = []

        async with self._lock:
            if not self._vectors:
                return []

            # Extract all stored vectors and their IDs for batch processing
            all_vectors = np.array(list(self._vectors.values()))
            all_ids = list(self._vectors.keys())

            # Calculate cosine similarity using NumPy for efficiency
            # Handle zero-norm vectors gracefully to prevent division by zero
            norm_query_vec = query_vec_np / (np.linalg.norm(query_vec_np) or 1e-9)
            norms_all_vectors = np.linalg.norm(all_vectors, axis=1)
            # Replace zero norms with a small value to prevent division by zero
            norms_all_vectors[norms_all_vectors == 0] = 1e-9
            norm_all_vectors = all_vectors / norms_all_vectors[:, np.newaxis]

            similarities = np.dot(norm_all_vectors, norm_query_vec)

            # Get indices of top k results
            # `argsort` returns indices that would sort an array, `[::-1]` reverses for descending order
            top_k_indices = np.argsort(similarities)[::-1][:k]

            for idx in top_k_indices:
                vec_id = all_ids[idx]
                score = float(similarities[idx])
                results.append((vec_id, score, self._metadatas.get(vec_id, {})))
        return results

    async def delete_vectors(self, ids: List[str]) -> None:
        """See BaseVectorStoreClient."""
        async with self._lock:
            for vec_id in ids:
                self._vectors.pop(vec_id, None)
                self._metadatas.pop(vec_id, None)

    async def clear(self) -> None:
        """See BaseVectorStoreClient."""
        async with self._lock:
            self._vectors.clear()
            self._metadatas.clear()
            self._next_id = 0

# --- SemanticRouter Core Logic ---

class Route(BaseModel):
    """
    Represents a specific routing path or intent within the framework.

    Attributes:
        name: A unique identifier for the route (e.g., "qa_chain", "customer_support_agent").
        description: A human-readable description of what this route handles.
                     Crucial for the LLM to understand and for debuggability.
        examples: A list of example user queries that should map to this route.
                  These examples are embedded and used for semantic matching.
        target: An identifier or configuration for the downstream component
                that handles this route (e.g., a function name, a chain ID,
                a tool name, or a complex dictionary). This is what the router
                will ultimately return.
    """
    name: str = Field(..., description="Unique name for the route.")
    description: str = Field(..., description="Human-readable description of the route's purpose.")
    examples: List[str] = Field(default_factory=list, description="Example queries that should match this route.")
    target: Any = Field(..., description="The identifier or configuration for the downstream component.")
    _vector_ids: List[str] = PrivateAttr(default_factory=list) # Internal: IDs of example embeddings in the vector store

class RouteMatch(BaseModel):
    """
    Represents the result of a semantic routing operation.

    Attributes:
        route: The matched Route object.
        score: The similarity score of the match, typically cosine similarity.
        query: The original query that was routed.
    """
    route: Route = Field(..., description="The matched route.")
    score: float = Field(..., description="The similarity score of the match.")
    query: str = Field(..., description="The original query that was routed.")

class SemanticRouter:
    """
    Vishustra's core semantic routing component.

    This class takes a user query, embeds it, and uses a vector store to find the
    most semantically similar predefined `Route`. It allows for flexible,
    context-aware dispatching of queries to different framework components.

    Attributes:
        embedding_model: An instance of a BaseEmbeddingModel implementation.
        vector_store: An instance of a BaseVectorStoreClient implementation.
        routes: A dictionary mapping route names to Route objects.
        min_similarity_threshold: The minimum cosine similarity score required
                                  for a query to be considered a match for a route.
                                  If no route meets this threshold, None is returned.
    """

    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        vector_store: BaseVectorStoreClient,
        routes: Optional[List[Route]] = None,
        min_similarity_threshold: float = 0.75, # A common starting point for cosine similarity
    ):
        """
        Initializes the SemanticRouter.

        Args:
            embedding_model: The embedding model to use for generating vector representations.
            vector_store: The vector store client to use for similarity search.
            routes: An initial list of Route objects to register.
            min_similarity_threshold: Minimum similarity score (0.0 to 1.0) for a route to be considered a match.
        """
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.min_similarity_threshold = min_similarity_threshold
        self._routes: Dict[str, Route] = {}

        if routes:
            # Store initial routes to be added during asynchronous initialization
            self._initial_routes = routes
        else:
            self._initial_routes = []

        self._initialized = False # Flag to ensure async setup is done before use
        self._initialization_lock = asyncio.Lock() # Prevent race conditions during initialization

    async def _initialize(self) -> None:
        """
        Asynchronously initializes the router by indexing all initial routes.
        This method ensures that routes are added to the vector store only once.
        It should be awaited before `route_query` is called for the first time
        or explicitly if dynamic route management is expected after initial setup.
        """
        async with self._initialization_lock:
            if self._initialized:
                return

            print(f"Initializing SemanticRouter with {len(self._initial_routes)} initial routes...")
            # Clear vector store for a clean start with initial routes
            await self.vector_store.clear()
            for route in self._initial_routes:
                # Add routes one by one, which handles embedding and indexing
                await self.add_route(route)
            self._initialized = True
            self._initial_routes = [] # Clear temporary storage
            print("SemanticRouter initialization complete.")

    async def add_route(self, route: Route) -> None:
        """
        Adds a new route to the router. Its examples will be embedded and indexed
        in the configured vector store.

        Args:
            route: The `Route` object to add.

        Raises:
            ValueError: If a route with the same name already exists.
        """
        if route.name in self._routes:
            raise ValueError(f"Route with name '{route.name}' already exists.")

        if not route.examples:
            print(f"Warning: Route '{route.name}' has no examples. It will not be searchable by query.")
            self._routes[route.name] = route
            return

        embeddings = await self.embedding_model.embed_documents(route.examples)
        metadatas = [{"route_name": route.name, "example_text": ex} for ex in route.examples]

        # Store embeddings in the vector store and get their assigned unique IDs
        vector_ids = await self.vector_store.add_vectors(embeddings, metadatas=metadatas)
        # Store these IDs within the route object for future removal if needed
        route._vector_ids = vector_ids
        self._routes[route.name] = route
        print(f"Route '{route.name}' added with {len(vector_ids)} examples indexed.")

    async def remove_route(self, route_name: str) -> None:
        """
        Removes a route from the router and deletes its associated embeddings
        from the underlying vector store.

        Args:
            route_name: The name of the route to remove.
        """
        route = self._routes.pop(route_name, None)
        if route:
            if route._vector_ids:
                # Delete associated vectors from the store
                await self.vector_store.delete_vectors(route._vector_ids)
                print(f"Route '{route_name}' and its {len(route._vector_ids)} indexed examples removed.")
            else:
                print(f"Route '{route_name}' removed (no examples were indexed).")
        else:
            print(f"Warning: Route '{route_name}' not found, nothing to remove.")

    async def route_query(self, query: str) -> Optional[RouteMatch]:
        """
        Asynchronously routes an incoming user query to the most semantically
        similar registered route.

        The process involves:
        1. Ensuring the router is initialized (all initial routes indexed).
        2. Embedding the incoming query.
        3. Performing a similarity search in the vector store against all
           indexed route examples.
        4. Identifying the top-matching route based on the search results.
        5. Returning a `RouteMatch` object if the best match meets the
           `min_similarity_threshold`, otherwise `None`.

        Args:
            query: The user's input query string.

        Returns:
            A `RouteMatch` object containing the matched route, score, and original query
            if a suitable route is found above the `min_similarity_threshold`,
            otherwise `None`.
        """
        if not self._initialized:
            # Ensure initialization happens on first use, but only once
            await self._initialize()

        if not self._routes:
            print("Warning: No routes registered in the router. Cannot route query.")
            return None

        query_embedding = await self.embedding_model.embed_query(query)

        # Search the vector store for the single most similar route example.
        # The associated metadata will allow us to identify the parent route.
        search_results = await self.vector_store.search(query_embedding, k=1)

        if not search_results:
            print(f"No semantic matches found in vector store for query: '{query}'")
            return None

        top_id, top_score, top_metadata = search_results[0]
        matched_route_name = top_metadata.get("route_name")

        if not matched_route_name:
            print(f"Error: Vector store returned a match (ID: {top_id}) without 'route_name' metadata. This indicates a data integrity issue.")
            return None

        if top_score >= self.min_similarity_threshold:
            matched_route = self._routes.get(matched_route_name)
            if matched_route:
                print(f"Query '{query}' routed to '{matched_route.name}' with score: {top_score:.3f}")
                return RouteMatch(route=matched_route, score=top_score, query=query)
            else:
                print(f"Warning: Matched route name '{matched_route_name}' found in vector store but not in internal _routes dictionary. This implies a synchronization issue.")
                return None
        else:
            print(f"No route met the similarity threshold ({self.min_similarity_threshold:.2f}) for query: '{query}' (best score: {top_score:.3f})")
            return None

# --- Example Usage (for internal testing/demonstration, typically removed or moved to a separate test file in production) ---

async def main():
    print("Vishustra SemanticRouter Demo")

    # 1. Initialize Core Components: Embedding Model and Vector Store
    try:
        embedding_model = SentenceTransformerEmbeddingModel()
    except ImportError as e:
        print(f"Cannot run demo: {e}. Please install 'sentence-transformers' (`pip install sentence-transformers`).")
        return

    vector_store = InMemoryVectorStoreClient() # Using in-memory for simple demo

    # 2. Define Example Routes
    routes = [
        Route(
            name="customer_support",
            description="Handles general customer support inquiries, account issues, billing questions.",
            examples=[
                "I need help with my account.",
                "How do I reset my password?",
                "What's my current bill?",
                "My payment failed.",
                "I want to talk to customer service.",
            ],
            target={"type": "agent", "id": "customer_support_agent"} # Example target configuration
        ),
        Route(
            name="product_qa",
            description="Answers questions about product features, specifications, and usage.",
            examples=[
                "What are the features of product X?",
                "How do I use feature Y?",
                "Tell me about the specifications of your new device.",
                "Does it support Z?",
                "Technical questions about the product.",
            ],
            target={"type": "chain", "id": "product_qa_chain"}
        ),
        Route(
            name="order_status",
            description="Provides information about order tracking and delivery status.",
            examples=[
                "Where is my order?",
                "What's the status of my recent purchase?",
                "When will my package arrive?",
                "Track my delivery.",
                "I want to check my order.",
            ],
            target={"type": "tool", "name": "order_tracker_tool"}
        ),
        Route(
            name="greetings",
            description="Handles polite greetings and small talk.",
            examples=[
                "Hello", "Hi there", "Good morning", "How are you?", "Hey"
            ],
            target={"type": "response", "message": "Hello! How can I assist you today?"}
        ),
    ]

    # 3. Instantiate SemanticRouter
    router = SemanticRouter(
        embedding_model=embedding_model,
        vector_store=vector_store,
        routes=routes,
        min_similarity_threshold=0.65 # Adjust threshold based on embedding model and desired strictness
    )

    # The router will implicitly call _initialize() on the first route_query,
    # but explicitly calling it here ensures all routes are ready before any queries.
    await router._initialize()

    # 4. Test Routing with various queries
    print("\n--- Testing Routing Queries ---")
    test_queries = [
        "I need help with my bill.",                      # -> customer_support
        "What are the features of the new phone?",        # -> product_qa
        "When will my package get here?",                 # -> order_status
        "Just saying hi!",                                # -> greetings
        "I can't log into my account.",                   # -> customer_support
        "Tell me about the software's capabilities.",     # -> product_qa
        "I want to know where my last order is.",         # -> order_status
        "How's the weather today?",                       # -> (Likely No match)
        "Can you help me with my subscription?",          # -> customer_support
        "Hi."                                             # -> greetings
    ]

    for query in test_queries:
        print(f"\nRouting query: '{query}'")
        match = await router.route_query(query)
        if match:
            print(f"  MATCH: Route='{match.route.name}' (Score: {match.score:.3f}, Target: {match.route.target})")
        else:
            print("  NO MATCH found above threshold.")

    # 5. Demonstrate Dynamic Route Management: Add a new route
    print("\n--- Dynamic Route Management: Adding a new route ---")
    new_route = Route(
        name="feedback_collection",
        description="Collects user feedback and suggestions.",
        examples=[
            "I have some feedback.",
            "I want to provide a suggestion.",
            "How can I give feedback?",
            "What do you think of this idea?",
            "I want to report an issue."
        ],
        target={"type": "tool", "name": "feedback_form_tool"}
    )
    await router.add_route(new_route)

    print("\nRouting a new query to the dynamically added feedback route:")
    match_feedback = await router.route_query("I have some suggestions for your service.")
    if match_feedback:
        print(f"  MATCH: Route='{match_feedback.route.name}' (Score: {match_feedback.score:.3f})")
    else:
        print("  NO MATCH for feedback query.")

    # 6. Demonstrate Dynamic Route Management: Remove an existing route
    print("\n--- Dynamic Route Management: Removing an existing route ---")
    await router.remove_route("product_qa")

    print("\nRouting a query that previously matched 'product_qa' after its removal:")
    match_removed = await router.route_query("Tell me about the software's capabilities.")
    if match_removed:
        print(f"  MATCH: Route='{match_removed.route.name}' (Score: {match_removed.score:.3f})")
    else:
        print("  NO MATCH for query that previously matched 'product_qa' (as expected).")

    print("\nVishustra SemanticRouter Demo Complete.")


if __name__ == "__main__":
    # Run the asynchronous main function
    asyncio.run(main())