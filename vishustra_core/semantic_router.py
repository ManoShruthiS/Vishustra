import abc
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Configure logging for the module
logger = logging.getLogger(__name__)

# --- Abstract Base Class for Embedding Providers ---
class EmbeddingProvider(abc.ABC):
    """
    Abstract Base Class defining the interface for any embedding model used within Vishustra.
    This allows for interchangeable embedding backend implementations (e.g., OpenAI, Cohere, local models).
    """

    @abc.abstractmethod
    async def embed_query(self, text: str) -> np.ndarray:
        """
        Asynchronously embeds a single query string.

        Args:
            text: The string to embed.

        Returns:
            A 1D numpy array representing the embedding vector.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def embed_documents(self, texts: List[str]) -> List[np.ndarray]:
        """
        Asynchronously embeds a list of document strings.

        Args:
            texts: A list of strings to embed.

        Returns:
            A list of 1D numpy arrays, each representing an embedding vector.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def dimensionality(self) -> int:
        """
        Returns the dimensionality of the embeddings produced by this provider.
        """
        raise NotImplementedError

# --- Data Class for a Routing Destination ---
@dataclass(frozen=True)
class Route:
    """
    Represents a single routing destination or topic within the SemanticRouter.

    Attributes:
        name: A unique identifier for the route (e.g., "customer_support", "product_info").
        description: A brief explanation of what this route handles.
        example_queries: A list of example natural language queries that should route here.
                         These are used to generate the route's representative embedding.
        handler: The actual object or identifier that this route dispatches to.
                 This could be a chain name, a tool identifier, a callable function, etc.
        metadata: An optional dictionary for additional, custom data associated with the route.
    """
    name: str
    description: str
    example_queries: List[str] = field(default_factory=list)
    handler: Any
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.name:
            raise ValueError("Route 'name' cannot be empty.")
        if not self.example_queries:
            logger.warning(
                f"Route '{self.name}' has no example queries. "
                "It might not be effectively routable via semantic similarity unless a default route is heavily relied upon."
            )

# --- Helper Function for Cosine Similarity ---
def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Computes the cosine similarity between two numpy vectors.

    Args:
        vec1: The first numpy vector (1D array).
        vec2: The second numpy vector (1D array).

    Returns:
        The cosine similarity as a float between -1.0 and 1.0.
        Returns 0.0 if either vector has zero norm to prevent division by zero.
    """
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product / (norm_a * norm_b)

# --- Main Semantic Router Class ---
class SemanticRouter:
    """
    An intelligent router that dispatches incoming queries to the most semantically relevant `Route`.
    It uses an `EmbeddingProvider` to embed queries and route examples, then performs
    cosine similarity matching.

    The router computes a representative embedding for each route by taking the mean of its
    example queries' embeddings. It then compares incoming query embeddings against these
    route embeddings to find the best match.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        routes: List[Route],
        similarity_threshold: float = 0.75,
        default_route: Optional[Route] = None
    ):
        """
        Initializes the SemanticRouter.

        Args:
            embedding_provider: An instance of an EmbeddingProvider to generate embeddings.
            routes: A list of `Route` objects defining the available routing destinations.
            similarity_threshold: The minimum cosine similarity required for a query to be
                                  considered a match for a route. If no route exceeds this
                                  threshold, the `default_route` is returned, or None.
                                  Must be between 0.0 and 1.0.
            default_route: An optional `Route` to fall back to if no route exceeds the
                           `similarity_threshold`.
        """
        if not isinstance(embedding_provider, EmbeddingProvider):
            raise TypeError("embedding_provider must be an instance of EmbeddingProvider.")
        if not all(isinstance(r, Route) for r in routes):
            raise TypeError("All items in 'routes' must be instances of Route.")
        if not (0.0 <= similarity_threshold <= 1.0):
            raise ValueError("similarity_threshold must be between 0.0 and 1.0.")
        if default_route and not isinstance(default_route, Route):
            raise TypeError("default_route must be an instance of Route if provided.")

        self.embedding_provider = embedding_provider
        self.routes = routes
        self.similarity_threshold = similarity_threshold
        self.default_route = default_route

        self._route_embeddings: List[Tuple[Route, np.ndarray]] = []
        self._initialized = False

    async def ainitialize(self) -> None:
        """
        Asynchronously computes and stores the representative embedding for each route.
        Each route's embedding is the mean of its example queries' embeddings.
        This method can be called upfront to pre-warm the router and avoid lazy
        initialization on the first `route` call.
        """
        if self._initialized:
            return

        logger.info(f"Initializing embeddings for {len(self.routes)} routes...")
        all_example_queries = []
        # Store tuples of (Route, List[str]) to map embedded examples back to routes
        routes_with_examples: List[Tuple[Route, List[str]]] = []

        for route in self.routes:
            if route.example_queries:
                all_example_queries.extend(route.example_queries)
                routes_with_examples.append((route, route.example_queries))
            else:
                logger.info(f"Route '{route.name}' has no example queries. Skipping embedding computation for this route.")

        if not all_example_queries:
            logger.warning("No example queries provided across all routes. SemanticRouter will rely solely on default_route if set.")
            self._initialized = True
            return

        # Embed all example queries in a batch for efficiency
        try:
            batched_embeddings = await self.embedding_provider.embed_documents(all_example_queries)
            if not batched_embeddings or len(batched_embeddings) != len(all_example_queries):
                raise ValueError("Embedding provider returned an inconsistent number of embeddings.")
        except Exception as e:
            logger.error(f"Failed to embed example queries: {e}")
            raise RuntimeError("Embedding provider failed to process example queries during initialization.") from e

        # Reconstruct route embeddings from the batched embeddings
        current_idx = 0
        for route, examples in routes_with_examples:
            embeddings_for_route = batched_embeddings[current_idx : current_idx + len(examples)]
            if embeddings_for_route:
                # Calculate the mean embedding for the route
                mean_embedding = np.mean(embeddings_for_route, axis=0)
                self._route_embeddings.append((route, mean_embedding))
            else:
                logger.warning(f"Could not compute mean embedding for route '{route.name}' as its example queries resulted in no valid embeddings.")
            current_idx += len(examples)

        if not self._route_embeddings and not self.default_route:
            logger.warning("No routable embeddings were computed and no default route is set. Router may always return None.")

        self._initialized = True
        logger.info(f"SemanticRouter initialized with {len(self._route_embeddings)} routable routes.")

    async def route(self, query: str) -> Optional[Route]:
        """
        Asynchronously routes an incoming query to the most semantically similar `Route`.

        Args:
            query: The natural language query string to route.

        Returns:
            The best matching `Route` object, or the `default_route` if no match exceeds
            the `similarity_threshold`, or `None` if no match and no `default_route`.
        """
        if not query:
            logger.debug("Received empty query. Returning default_route if available.")
            return self.default_route

        if not self._initialized:
            await self.ainitialize()

        if not self._route_embeddings:
            logger.debug("No routable routes initialized or available. Returning default_route.")
            return self.default_route

        try:
            query_embedding = await self.embedding_provider.embed_query(query)
            if query_embedding is None or query_embedding.ndim != 1:
                raise ValueError("Embedding provider returned an invalid query embedding.")
        except Exception as e:
            logger.error(f"Failed to embed query '{query[:50]}...': {e}")
            # Fallback to default route if embedding fails
            return self.default_route

        best_match_route: Optional[Route] = None
        max_similarity: float = -1.0 # Similarity scores are [-1, 1], so -1 is a good starting min

        for route, route_embed in self._route_embeddings:
            similarity = _cosine_similarity(query_embedding, route_embed)
            logger.debug(f"Query '{query[:20]}...' vs Route '{route.name}': Similarity = {similarity:.4f}")
            if similarity > max_similarity:
                max_similarity = similarity
                best_match_route = route

        if best_match_route and max_similarity >= self.similarity_threshold:
            logger.info(
                f"Query '{query[:50]}...' routed to '{best_match_route.name}' "
                f"with similarity {max_similarity:.4f} (threshold {self.similarity_threshold:.2f})."
            )
            return best_match_route
        else:
            if best_match_route:
                logger.info(
                    f"No route met similarity threshold ({self.similarity_threshold:.2f}). "
                    f"Best match was '{best_match_route.name}' with {max_similarity:.4f}. "
                    "Returning default_route."
                )
            else:
                logger.info(
                    "No routable routes or best_match_route was None. "
                    "Returning default_route."
                )
            return self.default_route