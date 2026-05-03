"""
semantic_router.py

This module provides a robust and highly configurable semantic routing mechanism for
the Vishustra LLM orchestration framework. It allows incoming queries or user prompts
to be dynamically directed to the most appropriate backend chain, agent, or tool
based on their semantic meaning.

The router leverages an embedder to convert natural language descriptions of routes
and incoming queries into vector representations. It then uses similarity metrics
(e.g., cosine similarity) to determine the best match, enabling intelligent dispatch
and enhancing the modularity and efficiency of complex LLM applications.

Key Features:
- Asynchronous operation for embedding and routing.
- Pluggable `AbstractEmbedder` interface for various embedding models.
- Pydantic models for declarative route and router configuration.
- Configurable similarity threshold and fallback mechanisms.
- Dynamic route management (add, remove routes).
- Efficient pre-embedding of route descriptions during initialization.
"""

from __future__ import annotations # For postponed evaluation of type annotations
import abc
import asyncio
from typing import Any, List, Optional, Tuple, Type, TypeVar

import numpy as np
from pydantic import BaseModel, Field, PrivateAttr, ValidationError, model_validator


# --- Type Definitions ---
Embedding = List[float]
Vector = np.ndarray # For internal numpy operations


# --- Abstract Base Classes ---

class AbstractEmbedder(abc.ABC):
    """
    Abstract base class for all embedders used within Vishustra.
    Concrete implementations must provide methods to convert text into vector embeddings.
    """

    @abc.abstractmethod
    async def embed(self, text: str) -> Embedding:
        """
        Embeds a single string of text into a vector.

        Args:
            text: The text string to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[Embedding]:
        """
        Embeds a batch of text strings into a list of vectors.

        Args:
            texts: A list of text strings to embed.

        Returns:
            A list of lists of floats, where each inner list is an embedding vector.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def dimensions(self) -> int:
        """
        Returns the dimension of the embeddings produced by this embedder.

        Returns:
            An integer representing the embedding dimension.
        """
        raise NotImplementedError


# --- Concrete Embedder Example (for demonstration/testing) ---
# In a real Vishustra setup, this would be in a separate `embedders` module
# and could wrap OpenAI, Cohere, local models, etc.

class DummyEmbedder(AbstractEmbedder):
    """
    A dummy embedder for demonstration and testing purposes.
    It generates fixed-size random embeddings.
    DO NOT USE IN PRODUCTION. This implementation assumes a fixed output dimension.
    """
    def __init__(self, dimension: int = 1536):
        self._dimension = dimension
        np.random.seed(42) # For reproducible dummy embeddings across calls

    async def embed(self, text: str) -> Embedding:
        """Generates a dummy embedding for a single text."""
        # Simple hash-based seeding for diverse dummy embeddings based on input text
        seed = sum(ord(c) for c in text) % 1000
        rng = np.random.default_rng(seed)
        return rng.rand(self._dimension).tolist()

    async def embed_batch(self, texts: List[str]) -> List[Embedding]:
        """Generates dummy embeddings for a batch of texts."""
        # In a real scenario, this would be optimized for batch API calls
        return [await self.embed(text) for text in texts]

    async def dimensions(self) -> int:
        """Returns the configured dummy embedding dimension."""
        return self._dimension


# --- Data Models ---

class Route(BaseModel):
    """
    Represents a specific routing destination within Vishustra.
    Each route has a name, a descriptive text, and a target identifier.
    The descriptive text is used to semantically match incoming queries.
    """
    name: str = Field(..., description="A unique name for this route.")
    description: str = Field(..., description="A natural language description of what this route handles. This text will be embedded for semantic matching.")
    target_id: str = Field(..., description="The identifier of the Vishustra component (e.g., chain ID, agent ID, tool name) to activate if this route is chosen.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata associated with this route.")

    @model_validator(mode='after')
    def validate_name_and_target_id(self) -> 'Route':
        if not self.name.strip():
            raise ValueError("Route name cannot be empty or just whitespace.")
        if not self.target_id.strip():
            raise ValueError("Route target_id cannot be empty or just whitespace.")
        if not self.description.strip():
            raise ValueError("Route description cannot be empty or just whitespace, as it's used for semantic matching.")
        return self


class SemanticRouterConfig(BaseModel):
    """
    Configuration model for the SemanticRouter.
    Defines the available routes, similarity threshold, and fallback behavior.
    """
    routes: List[Route] = Field(..., min_length=1, description="A list of Route objects that the router will consider.")
    similarity_threshold: float = Field(0.75, ge=0.0, le=1.0, description="Minimum cosine similarity required for a query to match a route. Values closer to 1.0 indicate stricter matching.")
    fallback_target_id: Optional[str] = Field(None, description="The 'target_id' to return if no route meets the similarity threshold. If None, the router will return None in such cases.")

    @model_validator(mode='after')
    def validate_routes_unique_names(self) -> 'SemanticRouterConfig':
        names = [route.name for route in self.routes]
        if len(names) != len(set(names)):
            raise ValueError("All route names must be unique.")
        return self


# --- Utility Functions ---

def cosine_similarity(vec1: Vector, vec2: Vector) -> float:
    """
    Calculates the cosine similarity between two vectors.
    Returns 0.0 if either vector is a zero vector to avoid division by zero.

    Args:
        vec1: The first vector (numpy array).
        vec2: The second vector (numpy array).

    Returns:
        The cosine similarity as a float between -1.0 and 1.0.
    """
    if vec1.shape != vec2.shape:
        raise ValueError(f"Vector dimensions must match. Got {vec1.shape} and {vec2.shape}")

    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0  # Avoid division by zero for zero vectors

    return dot_product / (norm_vec1 * norm_vec2)


# --- Main Router Class ---

class SemanticRouter:
    """
    The core Semantic Router for Vishustra.
    It takes an incoming query, embeds it, and compares it semantically
    to a set of pre-defined routes to determine the best destination.

    This class is designed to be asynchronous and requires explicit initialization
    via `initialize()` before routing operations.
    """
    def __init__(self, config: SemanticRouterConfig, embedder: AbstractEmbedder):
        """
        Initializes the SemanticRouter.

        Args:
            config: A SemanticRouterConfig object defining the routes and router behavior.
            embedder: An instance of an AbstractEmbedder to use for generating embeddings.

        Raises:
            TypeError: If config or embedder are not of the expected types.
        """
        if not isinstance(config, SemanticRouterConfig):
            raise TypeError("config must be an instance of SemanticRouterConfig")
        if not isinstance(embedder, AbstractEmbedder):
            raise TypeError("embedder must be an instance of AbstractEmbedder")

        self._config = config
        self._embedder = embedder
        # Internal storage for routes, mapped to their pre-computed embeddings
        # Stored as a list of tuples: (Route object, numpy array embedding)
        self._routed_items: List[Tuple[Route, Vector]] = []
        self._is_initialized = False # To ensure async init is awaited

    async def initialize(self) -> None:
        """
        Asynchronously initializes the router by pre-embedding all route descriptions.
        This method must be called exactly once before calling the `route` method
        or dynamically adding/removing routes.

        Raises:
            RuntimeError: If initialization fails (e.g., due to embedder issues).
            ValueError: If embedding dimensions are inconsistent.
        """
        if self._is_initialized:
            print("Vishustra SemanticRouter: Router already initialized.")
            return

        print(f"Vishustra SemanticRouter: Initializing with {len(self._config.routes)} routes...")
        route_descriptions = [r.description for r in self._config.routes]
        
        try:
            route_embeddings = await self._embedder.embed_batch(route_descriptions)
            if len(route_embeddings) != len(route_descriptions):
                raise ValueError("Embedder did not return an embedding for each route description.")
            
            # Verify embedding dimensions for consistency
            expected_dim = await self._embedder.dimensions()
            for i, emb in enumerate(route_embeddings):
                if len(emb) != expected_dim:
                    raise ValueError(f"Embedding for route '{self._config.routes[i].name}' has incorrect dimension {len(emb)}, expected {expected_dim}.")

            self._routed_items = [
                (self._config.routes[i], np.array(embedding))
                for i, embedding in enumerate(route_embeddings)
            ]
            self._is_initialized = True
            print("Vishustra SemanticRouter: Initialization complete.")
        except Exception as e:
            # Catching a broad exception here to log and re-raise as a specific RuntimeError
            # for router initialization failures.
            print(f"Vishustra SemanticRouter: Error during initialization: {e}")
            raise RuntimeError(f"SemanticRouter initialization failed: {e}") from e

    async def route(self, query: str) -> Optional[Route]:
        """
        Routes an incoming query to the most semantically similar pre-defined route.

        Args:
            query: The incoming user query or prompt string.

        Returns:
            The best-matching `Route` object if a match above the similarity threshold
            is found. If no match is found and `fallback_target_id` is configured,
            a synthetic `Route` for the fallback is returned. Otherwise, returns `None`.

        Raises:
            RuntimeError: If the router has not been initialized.
        """
        if not self._is_initialized:
            raise RuntimeError("SemanticRouter must be initialized by calling .initialize() before routing queries.")
        if not query or not query.strip():
            print("Vishustra SemanticRouter: Received empty or whitespace-only query, returning fallback or None.")
            return self._get_fallback_route()

        query_embedding = np.array(await self._embedder.embed(query))

        best_match_route: Optional[Route] = None
        max_similarity = -1.0

        for route, route_embedding in self._routed_items:
            try:
                # Ensure query embedding and route embedding have same dimensions for comparison
                if query_embedding.shape != route_embedding.shape:
                    print(f"Vishustra SemanticRouter: Dimension mismatch for route '{route.name}'. Query: {query_embedding.shape}, Route: {route_embedding.shape}. Skipping.")
                    continue

                similarity = cosine_similarity(query_embedding, route_embedding)
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match_route = route
            except ValueError as e:
                print(f"Vishustra SemanticRouter: Error calculating similarity for route '{route.name}': {e}")
                continue # Skip this route and try others

        if best_match_route and max_similarity >= self._config.similarity_threshold:
            print(f"Vishustra SemanticRouter: Query '{query[:50]}...' matched route '{best_match_route.name}' with similarity {max_similarity:.2f}")
            return best_match_route
        else:
            print(f"Vishustra SemanticRouter: No route found above threshold {self._config.similarity_threshold:.2f} (best: {max_similarity:.2f}).")
            return self._get_fallback_route()

    async def add_route(self, route: Route) -> None:
        """
        Adds a new route to the router dynamically.
        The new route's description will be embedded and added to the internal state.

        Args:
            route: The new `Route` object to add.

        Raises:
            RuntimeError: If the router has not been initialized.
            ValueError: If a route with the same name already exists, or embedding fails.
        """
        if not self._is_initialized:
            raise RuntimeError("SemanticRouter must be initialized before adding routes.")
        if any(item[0].name == route.name for item in self._routed_items):
            raise ValueError(f"Route with name '{route.name}' already exists.")

        print(f"Vishustra SemanticRouter: Adding new route '{route.name}'...")
        try:
            embedding = np.array(await self._embedder.embed(route.description))
            expected_dim = await self._embedder.dimensions()
            if len(embedding) != expected_dim:
                raise ValueError(f"Embedding for new route '{route.name}' has incorrect dimension {len(embedding)}, expected {expected_dim}.")

            self._routed_items.append((route, embedding))
            print(f"Vishustra SemanticRouter: Route '{route.name}' added successfully.")
        except Exception as e:
            print(f"Vishustra SemanticRouter: Failed to add route '{route.name}': {e}")
            raise ValueError(f"Failed to add route '{route.name}': {e}") from e

    async def remove_route(self, route_name: str) -> bool:
        """
        Removes a route from the router by its name.

        Args:
            route_name: The name of the route to remove.

        Returns:
            True if the route was found and removed, False otherwise.

        Raises:
            RuntimeError: If the router has not been initialized.
        """
        if not self._is_initialized:
            raise RuntimeError("SemanticRouter must be initialized before removing routes.")

        original_count = len(self._routed_items)
        self._routed_items = [item for item in self._routed_items if item[0].name != route_name]
        if len(self._routed_items) < original_count:
            print(f"Vishustra SemanticRouter: Route '{route_name}' removed.")
            return True
        print(f"Vishustra SemanticRouter: Route '{route_name}' not found for removal.")
        return False

    def _get_fallback_route(self) -> Optional[Route]:
        """Helper to construct and return the fallback route if configured."""
        if self._config.fallback_target_id:
            return Route(
                name="FALLBACK_ROUTE",
                description="Default fallback route when no specific route matches thresholds.",
                target_id=self._config.fallback_target_id,
                metadata={"reason": "no_match_above_threshold"}
            )
        return None


# --- Example Usage (for demonstrating functionality, typically in tests or example scripts) ---
async def _example_usage():
    print("\n--- Vishustra SemanticRouter Example ---")

    # 1. Define routes
    search_route = Route(
        name="search_query",
        description="User wants to search for information, ask questions, or retrieve facts.",
        target_id="search_agent_id"
    )
    order_route = Route(
        name="place_order",
        description="User wants to order a product, check order status, or modify an existing order.",
        target_id="e_commerce_agent_id"
    )
    support_route = Route(
        name="customer_support",
        description="User needs help, technical assistance, or has a complaint about a product or service.",
        target_id="helpdesk_chain_id"
    )
    admin_route = Route(
        name="admin_tasks",
        description="User is an administrator performing system operations, monitoring, or configuration tasks.",
        target_id="admin_tool_executor_id"
    )

    # 2. Configure the router
    router_config = SemanticRouterConfig(
        routes=[search_route, order_route, support_route, admin_route],
        similarity_threshold=0.7,
        fallback_target_id="default_llm_chain_id" # This will be returned if no route matches above threshold
    )

    # 3. Instantiate embedder and router
    embedder = DummyEmbedder(dimension=128) # Using a smaller dimension for quicker local testing
    router = SemanticRouter(config=router_config, embedder=embedder)

    # 4. Initialize the router (critical async step, must be awaited)
    await router.initialize()

    # 5. Test routing with various queries
    print("\n--- Testing Routing ---")
    queries = [
        "What is the capital of France?",
        "I want to buy a new laptop.",
        "My internet is not working.",
        "How can I reset my password?",
        "Show me all active users.",
        "Where is my package?",
        "Tell me a joke.", # Should go to fallback as it doesn't strongly match any defined route
        "", # Empty query test
        "     " # Whitespace query test
    ]

    for query in queries:
        print(f"\nQuery: '{query}'")
        matched_route = await router.route(query)
        if matched_route:
            print(f"  --> Matched Route: '{matched_route.name}' (Target: '{matched_route.target_id}')")
        else:
            print("  --> No route matched, and no fallback configured. Result: None")

    # 6. Test dynamic route addition
    print("\n--- Testing Dynamic Route Addition ---")
    new_route = Route(
        name="feedback_collection",
        description="User wants to give feedback, report a bug, or suggest a new feature for the platform.",
        target_id="feedback_system_api_id"
    )
    await router.add_route(new_route)

    query_feedback = "I have a suggestion for your product."
    print(f"\nQuery: '{query_feedback}' (after adding 'feedback_collection' route)")
    matched_route = await router.route(query_feedback)
    if matched_route:
        print(f"  --> Matched Route: '{matched_route.name}' (Target: '{matched_route.target_id}')")

    # 7. Test route removal
    print("\n--- Testing Route Removal ---")
    removed = await router.remove_route("search_query")
    print(f"Was 'search_query' route removed? {removed}")

    query_search_again = "Who invented the light bulb?"
    print(f"\nQuery: '{query_search_again}' (after removing 'search_query' route)")
    matched_route = await router.route(query_search_again)
    if matched_route:
        print(f"  --> Matched Route: '{matched_route.name}' (Target: '{matched_route.target_id}')")
    else:
        print("  --> No route matched, fallback activated or None returned.")


if __name__ == "__main__":
    # This block allows running the example directly for demonstration
    # It catches common exceptions during configuration or runtime.
    try:
        asyncio.run(_example_usage())
    except ValidationError as e:
        print(f"\nCRITICAL CONFIGURATION ERROR: {e}")
    except RuntimeError as e:
        print(f"\nCRITICAL RUNTIME ERROR: {e}")
    except Exception as e:
        print(f"\nAN UNEXPECTED ERROR OCCURRED: {e}")