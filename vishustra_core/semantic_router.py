import asyncio
import logging
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field, PrivateAttr, ValidationError, model_validator

# Assume these core Vishustra components and interfaces exist in the framework.
# They are included here as minimal definitions for context.

# --- Vishustra Core Base Components (e.g., from vishustra.core.base) ---
class VishustraComponent:
    """Base class for all Vishustra components, providing common utilities."""
    def __init__(self, component_id: Optional[str] = None, **kwargs: Any):
        self.component_id = component_id or self.__class__.__name__
        self._logger = logging.getLogger(f"{self.__class__.__name__}.{self.component_id}")
        self._logger.debug(f"Initialized {self.__class__.__name__} with ID: {self.component_id}")

# --- Vishustra Embedding Interfaces (e.g., from vishustra.embeddings.base) ---
class Embeddings(BaseModel):
    """
    Represents embedding vectors.

    Attributes:
        data: A numpy array where each row is an embedding vector.
              Expected shape (N, D) where N is the number of texts embedded,
              and D is the embedding dimension.
        model_name: Optional name of the embedding model that generated these.
    """
    data: np.ndarray = Field(..., description="Numpy array representing the embedding vectors.")
    model_name: Optional[str] = Field(None, description="Name of the embedding model used.")

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            np.ndarray: lambda v: v.tolist()  # For JSON serialization if needed
        }

class EmbeddingModel(Protocol):
    """Protocol for synchronous embedding models."""
    def embed_documents(self, texts: List[str]) -> Embeddings:
        """Embeds a list of documents/texts."""
        ...
    def embed_query(self, text: str) -> Embeddings:
        """Embeds a single query string."""
        ...

class AsyncEmbeddingModel(Protocol):
    """Protocol for asynchronous embedding models."""
    async def aembed_documents(self, texts: List[str]) -> Embeddings:
        """Asynchronously embeds a list of documents/texts."""
        ...
    async def aembed_query(self, text: str) -> Embeddings:
        """Asynchronously embeds a single query string."""
        ...

_logger = logging.getLogger(__name__)

class Route(BaseModel):
    """
    Represents a distinct processing route or chain within Vishustra.
    Each route is defined by a name, a description, and example queries
    that should semantically map to this route.
    """
    name: str = Field(..., description="Unique identifier for the route.")
    description: str = Field(..., description="A brief description of what this route handles or represents.")
    examples: List[str] = Field(
        default_factory=list,
        description="Example queries or prompts that should semantically map to this route. "
                    "These examples are used to compute the route's semantic embedding."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary key-value metadata associated with this route. "
                    "Can be used to store target chain IDs, function names, etc."
    )

    @model_validator(mode='after')
    def validate_examples_exist(self) -> 'Route':
        """Ensures that routes have examples, or logs a warning if they don't."""
        if not self.examples:
            _logger.warning(
                f"Route '{self.name}' has no examples. "
                "It will not be effectively routable by the SemanticRouter."
            )
        return self

class SemanticRouterConfig(BaseModel):
    """
    Configuration model for the SemanticRouter component.
    """
    default_similarity_threshold: float = Field(
        0.75,
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold. Routes with a similarity score below "
                    "this value will not be considered a match, resulting in an 'unmatched' state."
    )
    embedding_batch_size: int = Field(
        32,
        gt=0,
        description="Number of texts to embed in a single batch. Larger batches can improve "
                    "throughput for embedding models that support it, but may increase memory usage."
    )

class SemanticRouter(VishustraComponent):
    """
    A sophisticated semantic router that directs incoming queries to the most
    semantically relevant predefined 'route' within the Vishustra framework.

    This component leverages an embedding model to vectorize user queries and
    compares them against pre-computed aggregate embeddings of example queries
    for each registered route. The route with the highest cosine similarity
    score above a configured threshold is selected.

    Supports both synchronous and asynchronous embedding models and operations.
    """

    _routes: Dict[str, Route] = PrivateAttr(default_factory=dict)
    # Stores the aggregate embedding for each route's examples. Key: route_name.
    _route_embeddings: Dict[str, Embeddings] = PrivateAttr(default_factory=dict)

    def __init__(
        self,
        embedding_model: Union[EmbeddingModel, AsyncEmbeddingModel],
        routes: Optional[List[Route]] = None,
        config: Optional[SemanticRouterConfig] = None,
        **kwargs: Any
    ):
        """
        Initializes the SemanticRouter.

        Args:
            embedding_model: An instance of an embedding model (sync or async)
                             conforming to Vishustra's `EmbeddingModel` or
                             `AsyncEmbeddingModel` protocols.
            routes: An optional list of initial `Route` objects to register upon initialization.
            config: Optional `SemanticRouterConfig` for customizing router behavior.
            **kwargs: Arbitrary keyword arguments passed to the base `VishustraComponent`.

        Raises:
            TypeError: If `embedding_model` does not conform to the expected types.
        """
        super().__init__(**kwargs)
        if not isinstance(embedding_model, (EmbeddingModel, AsyncEmbeddingModel)):
            raise TypeError(
                "embedding_model must be an instance conforming to "
                "vishustra.embeddings.base.EmbeddingModel or AsyncEmbeddingModel."
            )

        self._embedding_model = embedding_model
        self.config = config or SemanticRouterConfig()
        self._is_async_embedding_model = isinstance(self._embedding_model, AsyncEmbeddingModel)
        self._logger.info(
            f"SemanticRouter initialized with {'async' if self._is_async_embedding_model else 'sync'} "
            f"embedding model. Default threshold: {self.config.default_similarity_threshold}"
        )

        if routes:
            if self._is_async_embedding_model:
                # If the embedding model is async, we need to run add_routes in an async context.
                # Since __init__ is sync, we can't directly await.
                # For initial setup, we might force sync embedding if the model provides it,
                # or require async init if many routes.
                # For simplicity here, we'll log a warning and recommend async setup for many routes.
                self._logger.warning(
                    "Attempting to add initial routes synchronously with an async embedding model. "
                    "Consider using `aadd_routes` after initialization for full async benefits, "
                    "or ensuring your async model has a synchronous fallback for initial setup."
                )
                # Fallback to sync for initialization if the async model supports it (not strictly covered by protocol)
                # or just process them with individual async calls if within an event loop
                # For this specific __init__, we need a sync way. Let's assume a limited set or defer.
                # Better approach: make init `async` or provide a `setup` method.
                # For current context, we use the `add_route` which handles its own sync/async logic.
                asyncio.run(self.aadd_routes(routes))
            else:
                self.add_routes(routes)

    def _embed_texts_sync(self, texts: List[str]) -> Embeddings:
        """Synchronously embeds a list of texts using the configured embedding model."""
        if self._is_async_embedding_model:
            raise RuntimeError(
                "Attempted to call synchronous embedding on an `AsyncEmbeddingModel`. "
                "Use `_embed_texts_async` instead."
            )
        return self._embedding_model.embed_documents(texts)

    async def _embed_texts_async(self, texts: List[str]) -> Embeddings:
        """Asynchronously embeds a list of texts using the configured embedding model."""
        if not self._is_async_embedding_model:
            raise RuntimeError(
                "Attempted to call asynchronous embedding on a `EmbeddingModel` (synchronous). "
                "Use `_embed_texts_sync` instead."
            )
        return await self._embedding_model.aembed_documents(texts)

    def _compute_route_embeddings(self, route: Route) -> Optional[Embeddings]:
        """
        Synchronously computes the aggregate embedding for a single route's examples.
        If the route has no examples, returns None.
        The aggregate embedding is typically the mean of all example embeddings.
        """
        if not route.examples:
            return None

        self._logger.debug(f"Computing embeddings for route '{route.name}' with {len(route.examples)} examples.")
        all_example_embeddings = []
        for i in range(0, len(route.examples), self.config.embedding_batch_size):
            batch = route.examples[i : i + self.config.embedding_batch_size]
            try:
                batch_embeddings = self._embed_texts_sync(batch)
                all_example_embeddings.append(batch_embeddings.data)
            except Exception as e:
                self._logger.error(f"Failed to embed batch for route '{route.name}': {e}", exc_info=True)
                raise  # Re-raise to indicate a critical failure

        if not all_example_embeddings:
            return None

        # Aggregate by taking the mean of all example embeddings for a single route vector
        aggregated_embedding = np.mean(np.vstack(all_example_embeddings), axis=0)
        return Embeddings(data=aggregated_embedding.reshape(1, -1), model_name=self._embedding_model.model_name if hasattr(self._embedding_model, 'model_name') else None)

    async def _acompute_route_embeddings(self, route: Route) -> Optional[Embeddings]:
        """
        Asynchronously computes the aggregate embedding for a single route's examples.
        If the route has no examples, returns None.
        """
        if not route.examples:
            return None

        self._logger.debug(f"Asynchronously computing embeddings for route '{route.name}' with {len(route.examples)} examples.")
        all_example_embeddings = []
        for i in range(0, len(route.examples), self.config.embedding_batch_size):
            batch = route.examples[i : i + self.config.embedding_batch_size]
            try:
                batch_embeddings = await self._embed_texts_async(batch)
                all_example_embeddings.append(batch_embeddings.data)
            except Exception as e:
                self._logger.error(f"Failed to async embed batch for route '{route.name}': {e}", exc_info=True)
                raise

        if not all_example_embeddings:
            return None

        aggregated_embedding = np.mean(np.vstack(all_example_embeddings), axis=0)
        return Embeddings(data=aggregated_embedding.reshape(1, -1), model_name=self._embedding_model.model_name if hasattr(self._embedding_model, 'model_name') else None)

    def add_route(self, route: Route) -> None:
        """
        Synchronously adds a single route to the router.
        The route's example embeddings are computed and stored.
        If a route with the same name exists, it will be overwritten.
        """
        if self._is_async_embedding_model:
            raise RuntimeError(
                "Cannot add route synchronously with an asynchronous embedding model. "
                "Use `aadd_route` instead or ensure synchronous setup."
            )

        if route.name in self._routes:
            self._logger.warning(f"Route '{route.name}' already exists. Overwriting existing route.")

        self._routes[route.name] = route
        route_embeds = self._compute_route_embeddings(route)
        if route_embeds is not None:
            self._route_embeddings[route.name] = route_embeds
            self._logger.info(f"Route '{route.name}' added successfully.")
        else:
            self._route_embeddings.pop(route.name, None) # Ensure no stale entries
            self._logger.warning(f"Route '{route.name}' added, but has no examples/embeddings computed.")

    async def aadd_route(self, route: Route) -> None:
        """
        Asynchronously adds a single route to the router.
        The route's example embeddings are computed and stored.
        If a route with the same name exists, it will be overwritten.
        """
        if not self._is_async_embedding_model:
            raise RuntimeError(
                "Cannot add route asynchronously with a synchronous embedding model. "
                "Use `add_route` instead."
            )

        if route.name in self._routes:
            self._logger.warning(f"Route '{route.name}' already exists. Overwriting existing route.")

        self._routes[route.name] = route
        route_embeds = await self._acompute_route_embeddings(route)
        if route_embeds is not None:
            self._route_embeddings[route.name] = route_embeds
            self._logger.info(f"Async route '{route.name}' added successfully.")
        else:
            self._route_embeddings.pop(route.name, None)
            self._logger.warning(f"Async route '{route.name}' added, but has no examples/embeddings computed.")

    def add_routes(self, routes: List[Route]) -> None:
        """
        Synchronously adds multiple routes to the router.
        """
        for route in routes:
            self.add_route(route)
        self._logger.info(f"Added {len(routes)} routes synchronously.")

    async def aadd_routes(self, routes: List[Route]) -> None:
        """
        Asynchronously adds multiple routes to the router.
        This method uses `asyncio.gather` for concurrent processing of routes.
        """
        await asyncio.gather(*[self.aadd_route(route) for route in routes])
        self._logger.info(f"Added {len(routes)} routes asynchronously.")

    def remove_route(self, route_name: str) -> None:
        """
        Removes a route by its name.
        """
        if route_name not in self._routes:
            self._logger.warning(f"Route '{route_name}' not found. No action taken.")
            return

        del self._routes[route_name]
        self._route_embeddings.pop(route_name, None)
        self._logger.info(f"Route '{route_name}' and its embeddings removed.")

    def _calculate_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculates cosine similarity between two 1D numpy arrays (vectors).
        Assumes vectors are already normalized for efficiency, or normalizes them.
        """
        # Ensure vectors are 1D
        vec1 = vec1.flatten()
        vec2 = vec2.flatten()

        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)

        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0 # Handle cases where a vector is zero (no information)

        return dot_product / (norm_vec1 * norm_vec2)

    def route(self, query: str, threshold: Optional[float] = None) -> Optional[Tuple[Route, float]]:
        """
        Routes a given query to the most suitable route synchronously.

        Args:
            query: The incoming user query or prompt string.
            threshold: An optional override for the default similarity threshold defined in config.

        Returns:
            A tuple containing the best matching `Route` object and its cosine similarity score,
            or `None` if no route meets the similarity threshold or no routes are registered.

        Raises:
            RuntimeError: If the router is configured with an `AsyncEmbeddingModel`.
            Exception: For failures during query embedding.
        """
        if not self._routes:
            self._logger.warning("No routes registered in the router. Cannot route query.")
            return None
        if self._is_async_embedding_model:
            raise RuntimeError(
                "Cannot use synchronous `route()` method with an `AsyncEmbeddingModel`. "
                "Use `aroute()` instead."
            )

        _threshold = threshold if threshold is not None else self.config.default_similarity_threshold

        try:
            # Assuming embed_query returns Embeddings object with a single vector in data
            query_embedding_obj = self._embedding_model.embed_query(query)
            query_embedding = query_embedding_obj.data.flatten()
        except Exception as e:
            self._logger.error(f"Failed to embed query '{query[:100]}...': {e}", exc_info=True)
            return None

        best_route: Optional[Route] = None
        highest_similarity: float = -1.0

        for route_name, route_embed_obj in self._route_embeddings.items():
            route_embedding = route_embed_obj.data.flatten()
            similarity = self._calculate_cosine_similarity(query_embedding, route_embedding)

            self._logger.debug(
                f"Query '{query[:50]}...' vs Route '{route_name}': Similarity = {similarity:.4f}"
            )

            if similarity > highest_similarity:
                highest_similarity = similarity
                best_route = self._routes[route_name] # Retrieve original Route object

        if best_route and highest_similarity >= _threshold:
            self._logger.info(
                f"Query '{query[:50]}...' routed to '{best_route.name}' "
                f"with similarity {highest_similarity:.4f} (Threshold: {_threshold:.2f})"
            )
            return best_route, highest_similarity
        else:
            self._logger.info(
                f"No route found above threshold {_threshold:.2f} for query '{query[:50]}...' "
                f"(Best match: {best_route.name if best_route else 'N/A'}, Similarity: {highest_similarity:.4f})"
            )
            return None

    async def aroute(self, query: str, threshold: Optional[float] = None) -> Optional[Tuple[Route, float]]:
        """
        Asynchronously routes a given query to the most suitable route.

        Args:
            query: The incoming user query or prompt string.
            threshold: An optional override for the default similarity threshold defined in config.

        Returns:
            A tuple containing the best matching `Route` object and its cosine similarity score,
            or `None` if no route meets the similarity threshold or no routes are registered.

        Raises:
            RuntimeError: If the router is configured with a synchronous `EmbeddingModel`.
            Exception: For failures during query embedding.
        """
        if not self._routes:
            self._logger.warning("No routes registered in the router. Cannot route query.")
            return None
        if not self._is_async_embedding_model:
            raise RuntimeError(
                "Cannot use asynchronous `aroute()` method with a synchronous `EmbeddingModel`. "
                "Use `route()` instead."
            )

        _threshold = threshold if threshold is not None else self.config.default_similarity_threshold

        try:
            query_embedding_obj = await self._embedding_model.aembed_query(query)
            query_embedding = query_embedding_obj.data.flatten()
        except Exception as e:
            self._logger.error(f"Failed to async embed query '{query[:100]}...': {e}", exc_info=True)
            return None

        best_route: Optional[Route] = None
        highest_similarity: float = -1.0

        # Note: If the number of routes becomes extremely large, one might consider a
        # vector database for efficient similarity search instead of in-memory iteration.
        for route_name, route_embed_obj in self._route_embeddings.items():
            route_embedding = route_embed_obj.data.flatten()
            similarity = self._calculate_cosine_similarity(query_embedding, route_embedding)

            self._logger.debug(
                f"Async Query '{query[:50]}...' vs Route '{route_name}': Similarity = {similarity:.4f}"
            )

            if similarity > highest_similarity:
                highest_similarity = similarity
                best_route = self._routes[route_name]

        if best_route and highest_similarity >= _threshold:
            self._logger.info(
                f"Async query '{query[:50]}...' routed to '{best_route.name}' "
                f"with similarity {highest_similarity:.4f} (Threshold: {_threshold:.2f})"
            )
            return best_route, highest_similarity
        else:
            self._logger.info(
                f"No async route found above threshold {_threshold:.2f} for query '{query[:50]}...' "
                f"(Best match: {best_route.name if best_route else 'N/A'}, Similarity: {highest_similarity:.4f})"
            )
            return None

    def get_routes(self) -> List[Route]:
        """Returns a list of all currently registered `Route` objects."""
        return list(self._routes.values())

    def get_route_by_name(self, name: str) -> Optional[Route]:
        """Returns a `Route` object by its unique name, or `None` if not found."""
        return self._routes.get(name)

    def get_registered_route_names(self) -> List[str]:
        """Returns a list of names for all registered routes."""
        return list(self._routes.keys())