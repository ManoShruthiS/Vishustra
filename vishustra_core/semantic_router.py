import abc
import asyncio
import logging
from typing import (
    List,
    Dict,
    Any,
    Optional,
    Awaitable,
    Tuple,
    TypeVar,
    Generic,
)
from uuid import uuid4

from pydantic import BaseModel, Field, PrivateAttr, ValidationError

# Initialize logging for the module
logger = logging.getLogger(__name__)


# --- Core Interfaces (Abstract Base Classes) ---

class EmbeddingModel(abc.ABC):
    """
    Abstract Base Class for an embedding model.

    This interface defines the contract for any embedding model used within Vishustra.
    Implementations must provide asynchronous methods for embedding text.
    """

    @abc.abstractmethod
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Asynchronously embeds a list of texts into vector representations.

        Args:
            texts: A list of strings to embed.

        Returns:
            A list of embedding vectors, where each vector is a list of floats.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def dimension(self) -> int:
        """
        The dimension of the embedding vectors produced by this model.
        """
        raise NotImplementedError


class VectorStoreResult(BaseModel):
    """
    Represents a single result from a vector store search.
    """
    id: str = Field(description="Unique identifier for the stored vector.")
    vector: List[float] = Field(description="The embedding vector itself.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata associated with the vector.")
    score: float = Field(description="Similarity score of this result relative to the query.")


class VectorStore(abc.ABC):
    """
    Abstract Base Class for a vector store.

    This interface defines the contract for any vector database used within Vishustra
    to store and retrieve embeddings.
    """

    @abc.abstractmethod
    async def add(
        self,
        vectors: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Asynchronously adds vectors to the vector store.

        Args:
            vectors: A list of embedding vectors to add.
            metadatas: Optional. A list of dictionaries, where each dictionary
                       contains metadata corresponding to a vector.
            ids: Optional. A list of unique identifiers for the vectors. If not
                 provided, the vector store should generate them.

        Returns:
            A list of IDs for the added vectors.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def search(
        self, query_vector: List[float], top_k: int
    ) -> List[VectorStoreResult]:
        """
        Asynchronously searches the vector store for the most similar vectors to a query.

        Args:
            query_vector: The embedding vector of the query.
            top_k: The number of top similar results to return.

        Returns:
            A list of `VectorStoreResult` objects, sorted by similarity score in
            descending order.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def delete(self, ids: List[str]) -> None:
        """
        Asynchronously deletes vectors from the vector store by their IDs.

        Args:
            ids: A list of IDs of vectors to delete.
        """
        raise NotImplementedError

# --- Data Models for Routing ---

class Route(BaseModel):
    """
    Represents a specific route or action within the framework.

    Each route has a name, a descriptive text, a target (e.g., a tool name,
    function ID, or endpoint), and optional metadata. The description is used
    for semantic matching.
    """
    name: str = Field(description="A unique, human-readable name for the route.")
    description: str = Field(description="A detailed description of what this route does or represents. This text will be embedded and used for semantic matching.")
    target: str = Field(description="The identifier of the target component (e.g., tool name, agent ID, function name, API endpoint) to be invoked when this route is matched.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary additional data associated with this route.")

    # Private attribute to store the UUID generated for this route instance
    _id: str = PrivateAttr(default_factory=lambda: str(uuid4()))

    def __hash__(self) -> int:
        return hash((self.name, self.description, self.target, frozenset(self.metadata.items())))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Route):
            return NotImplemented
        return (
            self.name == other.name and
            self.description == other.description and
            self.target == other.target and
            self.metadata == other.metadata
        )

class RouteMatch(BaseModel):
    """
    Represents a matched route along with its similarity score.
    """
    route: Route = Field(description="The matched route object.")
    score: float = Field(description="The similarity score (e.g., cosine similarity) of the match.")


# --- Custom Exceptions ---

class RouterError(Exception):
    """Base exception for semantic router operations."""
    pass


class RouterInitializationError(RouterError):
    """Raised when the semantic router fails to initialize its routes."""
    pass


class RoutingError(RouterError):
    """Raised when an error occurs during the routing process."""
    pass


# --- Semantic Router Implementation ---

class SemanticRouter:
    """
    A highly modular and asynchronous semantic router for Vishustra.

    This router uses an embedding model and a vector store to semantically
    match incoming queries to predefined routes. It's designed for use in
    LLM orchestration frameworks to intelligently dispatch requests to
    appropriate tools, agents, or functions based on query meaning.

    Attributes:
        _embedding_model: The `EmbeddingModel` instance used for generating embeddings.
        _vector_store: The `VectorStore` instance used for storing and searching route embeddings.
        _routes_by_id: A mapping from vector store ID to the original `Route` object.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        routes: List[Route],
        min_score_threshold: float = 0.7,
    ):
        """
        Initializes the SemanticRouter with an embedding model, vector store, and routes.

        Args:
            embedding_model: An instance of a class implementing the `EmbeddingModel` interface.
            vector_store: An instance of a class implementing the `VectorStore` interface.
            routes: A list of `Route` objects that the router can match against.
            min_score_threshold: The minimum similarity score required for a route to be
                                 considered a match. Defaults to 0.7.
        
        Raises:
            RouterInitializationError: If routes cannot be initialized in the vector store.
            ValueError: If no routes are provided.
        """
        if not routes:
            raise ValueError("SemanticRouter requires at least one route to be initialized.")
        if not isinstance(embedding_model, EmbeddingModel):
            raise TypeError("embedding_model must be an instance of EmbeddingModel.")
        if not isinstance(vector_store, VectorStore):
            raise TypeError("vector_store must be an instance of VectorStore.")

        self._embedding_model: EmbeddingModel = embedding_model
        self._vector_store: VectorStore = vector_store
        self._routes: List[Route] = routes
        self._routes_by_id: Dict[str, Route] = {route._id: route for route in routes}
        self._min_score_threshold: float = min_score_threshold

        logger.info(f"Initializing SemanticRouter with {len(routes)} routes.")
        # We don't await initialization here as __init__ cannot be async.
        # The user of the router must await `await router.initialize()` explicitly
        # to ensure routes are loaded before routing queries.
        self._initialization_task: Optional[asyncio.Task] = None

    async def initialize(self) -> None:
        """
        Asynchronously initializes the router by embedding all routes and adding them
        to the vector store. This method must be called after instantiation and awaited
        before the router can be used for `route` calls.

        This method is idempotent; calling it multiple times will not re-initialize if
        it has already successfully completed.
        """
        if self._initialization_task and self._initialization_task.done():
            # If the task completed without error, we're already initialized.
            # If it completed with an exception, re-raise it for the caller.
            await self._initialization_task
            return 
        elif self._initialization_task:
            # If the task is running, await it.
            await self._initialization_task
            return

        # If no task, create and run it.
        self._initialization_task = asyncio.create_task(self._add_routes_to_vector_store())
        await self._initialization_task
        logger.info("SemanticRouter initialization complete.")


    async def _add_routes_to_vector_store(self) -> None:
        """
        Embeds route descriptions and adds them to the vector store.
        Handles deletion of existing routes with the same IDs to ensure idempotency.
        """
        descriptions = [route.description for route in self._routes]
        route_ids = [route._id for route in self._routes]
        metadatas = [
            {
                "route_name": route.name,
                "route_target": route.target,
                "original_route_id": route._id, # Store our internal ID
                **route.metadata,
            }
            for route in self._routes
        ]

        try:
            # First, delete any existing routes with these IDs to ensure idempotency.
            # This is important if routes might be updated or the router re-initialized.
            await self._vector_store.delete(route_ids)
            logger.debug(f"Deleted {len(route_ids)} potential pre-existing routes from vector store.")

            embeddings = await self._embedding_model.embed(descriptions)
            if len(embeddings) != len(descriptions):
                raise RouterInitializationError(
                    f"Embedding model returned {len(embeddings)} embeddings for {len(descriptions)} descriptions."
                )

            # Check embedding dimension
            if embeddings and len(embeddings[0]) != self._embedding_model.dimension:
                raise RouterInitializationError(
                    f"Embedding dimension mismatch: model reports {self._embedding_model.dimension}, "
                    f"but returned {len(embeddings[0])} for first embedding."
                )

            added_ids = await self._vector_store.add(
                vectors=embeddings,
                metadatas=metadatas,
                ids=route_ids,
            )
            if len(added_ids) != len(route_ids):
                raise RouterInitializationError(
                    f"Vector store added {len(added_ids)} vectors, expected {len(route_ids)}."
                )
            logger.info(f"Successfully added {len(added_ids)} routes to vector store.")

        except Exception as e:
            logger.error(f"Failed to initialize routes in vector store: {e}", exc_info=True)
            raise RouterInitializationError(f"Error during route initialization: {e}") from e

    async def route(self, query: str, top_k: int = 1) -> List[RouteMatch]:
        """
        Asynchronously routes an incoming query to the most semantically relevant routes.

        Args:
            query: The user query or intent string to route.
            top_k: The maximum number of top matching routes to return.

        Returns:
            A list of `RouteMatch` objects, sorted by score in descending order,
            and filtered by `min_score_threshold`. An empty list if no matches
            are found above the threshold.

        Raises:
            RoutingError: If embedding the query or searching the vector store fails,
                          or if the router has not been initialized.
        """
        if not self._initialization_task or not self._initialization_task.done():
            raise RoutingError(
                "SemanticRouter has not been initialized or initialization is still pending. Call `await router.initialize()` first."
            )
        try:
            # Ensure the initialization task completed successfully.
            # If it failed, this 'await' will re-raise the InitializationError.
            await self._initialization_task 
        except RouterInitializationError as e:
            raise RoutingError("Router initialization failed previously.") from e

        try:
            query_embedding_list = await self._embedding_model.embed([query])
            if not query_embedding_list:
                raise RoutingError("Embedding model returned empty embedding for query.")
            query_embedding = query_embedding_list[0]

            if len(query_embedding) != self._embedding_model.dimension:
                raise RoutingError(
                    f"Query embedding dimension mismatch: expected {self._embedding_model.dimension}, "
                    f"got {len(query_embedding)}."
                )

            search_results = await self._vector_store.search(query_embedding, top_k=top_k)
            logger.debug(f"Vector store search returned {len(search_results)} results for query.")

            matches: List[RouteMatch] = []
            for result in search_results:
                if result.score >= self._min_score_threshold:
                    original_route_id = result.metadata.get("original_route_id")
                    if original_route_id and original_route_id in self._routes_by_id:
                        route = self._routes_by_id[original_route_id]
                        matches.append(RouteMatch(route=route, score=result.score))
                        logger.debug(f"Matched route '{route.name}' with score {result.score:.4f}")
                    else:
                        logger.warning(
                            f"Vector store result with ID '{result.id}' has missing or unknown "
                            f"original_route_id '{original_route_id}'. Skipping."
                        )
                else:
                    logger.debug(
                        f"Skipping route match below threshold: ID='{result.id}', "
                        f"Score={result.score:.4f} < Threshold={self._min_score_threshold:.4f}"
                    )

            # Sort by score in descending order, although vector store should usually return them sorted
            matches.sort(key=lambda m: m.score, reverse=True)
            return matches

        except ValidationError as ve:
            logger.error(f"Pydantic validation error during routing: {ve}", exc_info=True)
            raise RoutingError(f"Invalid data encountered during routing: {ve}") from ve
        except Exception as e:
            logger.error(f"Failed to route query: {e}", exc_info=True)
            raise RoutingError(f"Error during query routing: {e}") from e


# --- Example Implementations (for demonstration, not part of the core framework) ---
# These would typically be in separate files like `openai_embedding.py` or `qdrant_vector_store.py`

class MockEmbeddingModel(EmbeddingModel):
    """A mock embedding model for testing purposes."""
    def __init__(self, dimension: int = 1536):
        self._dimension = dimension
        logger.warning("Using MockEmbeddingModel. This is for demonstration/testing only.")

    async def embed(self, texts: List[str]) -> List[List[float]]:
        # Simulate an API call with a small delay
        await asyncio.sleep(0.01)
        # Create distinct but somewhat predictable vectors for demonstration
        return [[float(hash(text + str(i)) % 1000) / 1000.0 for i in range(self._dimension)] for text in texts]

    @property
    def dimension(self) -> int:
        return self._dimension

class MockVectorStore(VectorStore):
    """A simple in-memory mock vector store for testing purposes."""
    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}
        logger.warning("Using MockVectorStore. This is for demonstration/testing only.")

    async def add(
        self,
        vectors: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        await asyncio.sleep(0.005) # Simulate async I/O
        added_ids = []
        for i, vec in enumerate(vectors):
            _id = ids[i] if ids and ids[i] else str(uuid4())
            self._store[_id] = {
                "vector": vec,
                "metadata": metadatas[i] if metadatas else {},
            }
            added_ids.append(_id)
        logger.debug(f"MockVectorStore: Added {len(added_ids)} vectors.")
        return added_ids

    async def search(
        self, query_vector: List[float], top_k: int
    ) -> List[VectorStoreResult]:
        await asyncio.sleep(0.005) # Simulate async I/O
        results: List[Tuple[float, str]] = []
        
        # Simple dot product for similarity (cosine if normalized)
        query_norm = (sum(q*q for q in query_vector)**0.5)
        if query_norm == 0:
            query_norm = 1e-8 # Avoid division by zero for zero vectors

        for _id, data in self._store.items():
            stored_vector = data["vector"]
            stored_norm = (sum(s*s for s in stored_vector)**0.5)
            if stored_norm == 0:
                stored_norm = 1e-8 # Avoid division by zero

            score = sum(q * s for q, s in zip(query_vector, stored_vector)) / (query_norm * stored_norm)
            results.append((score, _id))

        results.sort(key=lambda x: x[0], reverse=True)
        
        vector_store_results = []
        for score, _id in results[:top_k]:
            data = self._store[_id]
            vector_store_results.append(
                VectorStoreResult(id=_id, vector=data["vector"], metadata=data["metadata"], score=score)
            )
        logger.debug(f"MockVectorStore: Searched, returning {len(vector_store_results)} results.")
        return vector_store_results
    
    async def delete(self, ids: List[str]) -> None:
        await asyncio.sleep(0.001) # Simulate async I/O
        deleted_count = 0
        for _id in ids:
            if _id in self._store:
                del self._store[_id]
                deleted_count += 1
        logger.debug(f"MockVectorStore: Deleted {deleted_count} vectors.")


# --- Example Usage (for demonstration) ---
async def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.setLevel(logging.DEBUG) # Set this module's logger to debug for more detail

    # 1. Instantiate the mock components
    embedding_model = MockEmbeddingModel(dimension=8) # Lower dimension for easier mock
    vector_store = MockVectorStore()

    # 2. Define your application's routes
    routes = [
        Route(
            name="customer_support_query",
            description="Handles questions about product features, technical issues, order status, or general customer service inquiries.",
            target="customer_support_agent",
            metadata={"priority": "high", "department": "support"},
        ),
        Route(
            name="sales_inquiry",
            description="Routes requests related to purchasing new products, pricing, discounts, or partnership opportunities.",
            target="sales_team_webhook",
            metadata={"priority": "medium", "department": "sales"},
        ),
        Route(
            name="knowledge_base_search",
            description="Finds information in the internal knowledge base or documentation.",
            target="knowledge_base_tool",
            metadata={"source": "internal_docs"},
        ),
        Route(
            name="developer_api_docs",
            description="Provides access to API documentation, SDKs, and developer guides for integrating with Vishustra.",
            target="developer_portal_tool",
            metadata={"audience": "developers"},
        ),
        Route(
            name="feedback_submission",
            description="Collects user feedback, bug reports, or feature requests.",
            target="feedback_system_api",
            metadata={"type": "feedback"},
        ),
    ]

    # 3. Instantiate the SemanticRouter
    router = SemanticRouter(
        embedding_model=embedding_model,
        vector_store=vector_store,
        routes=routes,
        min_score_threshold=0.6, # Adjust threshold as needed
    )

    # 4. Initialize the router (important step to load routes into the vector store)
    logger.info("Starting router initialization...")
    await router.initialize()
    logger.info("Router initialized.")

    # 5. Route some queries
    queries = [
        "I need help with my account and a bug in the software.",
        "How much does your enterprise plan cost?",
        "Where can I find details about your REST API endpoints?",
        "I want to report a new feature idea.",
        "Tell me about the product's main capabilities.",
        "What is the capital of France?", # Should ideally not match strongly
        "My order #12345 is delayed, can you check it?"
    ]

    print("\n--- Routing Queries ---")
    for i, query in enumerate(queries):
        print(f"\nQuery {i+1}: '{query}'")
        try:
            matches = await router.route(query, top_k=2)
            if matches:
                for match in matches:
                    print(f"  -> Matched Route: '{match.route.name}' (Score: {match.score:.4f})")
                    print(f"     Target: {match.route.target}")
                    print(f"     Description: {match.route.description[:100]}...")
            else:
                print("  -> No strong match found.")
        except RouterError as e:
            print(f"  -> ERROR during routing: {e}")

if __name__ == "__main__":
    asyncio.run(main())