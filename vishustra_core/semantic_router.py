import asyncio
import logging
from typing import List, Dict, Any, Optional

import numpy as np
from pydantic import BaseModel, Field, ValidationError

# Assume these are core Vishustra components.
# In a real Vishustra setup, these would be imported from the framework's core modules.
# For this exercise, we define minimal interfaces or assume their existence.

# --- Vishustra Core Assumptions (for context and integration points) ---
class BaseComponent(BaseModel):
    """
    Base class for all Vishustra pluggable components.
    Provides a standardized way to identify components within the framework.
    """
    component_id: str = Field(..., description="Unique identifier for the component.")

    class Config:
        arbitrary_types_allowed = True # Allows non-Pydantic types like EmbeddingsProvider


class EmbeddingsProvider:
    """
    Simulated service for generating text embeddings within the Vishustra framework.
    In a production system, this would abstract various embedding models (e.g., OpenAI, HuggingFace, local).
    """
    def __init__(self, service_url: str = "http://embeddings-service.vishustra.local"):
        self._service_url = service_url
        logging.getLogger(__name__).info(f"Initialized EmbeddingsProvider connecting to {service_url}")

    async def get_embeddings(self, texts: List[str], model_id: str) -> List[List[float]]:
        """
        Asynchronously retrieves embeddings for a list of texts using a specified model ID.

        Args:
            texts: A list of strings for which to generate embeddings.
            model_id: The identifier for the embedding model to use (e.g., "openai-ada-002").

        Returns:
            A list of lists of floats, where each inner list is an embedding vector
            corresponding to the input text at the same index.
        """
        logger.debug(f"Getting embeddings for {len(texts)} texts using model '{model_id}'...")
        # Simulate network delay and actual embedding generation.
        # In a real system, this would involve an RPC call, HTTP request, or direct model inference.
        await asyncio.sleep(0.02 * len(texts))
        # Return dummy embeddings for demonstration purposes.
        dummy_dim = 768 # Common embedding dimension
        return [[np.random.rand() * 2 - 1 for _ in range(dummy_dim)] for _ in texts] # Random values between -1 and 1

# Configure a Vishustra-wide logger instance.
logger = logging.getLogger("vishustra")
if not logger.handlers:
    # Basic configuration if no handlers are already set (e.g., for standalone testing).
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.setLevel(logging.INFO)


class RoutingError(Exception):
    """Custom exception raised for errors encountered during semantic routing."""
    pass
# --- End Vishustra Core Assumptions ---


class RouteDestination(BaseModel):
    """
    Defines a potential destination for the SemanticRouter.

    Each destination is characterized by descriptive patterns (phrases, example queries)
    that semantically indicate when this route should be chosen.
    """
    destination_id: str = Field(..., description="Unique identifier for this specific route destination.")
    name: str = Field(..., description="A human-readable name for the destination (e.g., 'Knowledge Base Q&A').")
    description: str = Field(
        ...,
        max_length=500,
        description="A brief, comprehensive description of what this route handles and its purpose."
    )
    patterns: List[str] = Field(
        ...,
        min_items=1,
        description="A list of example phrases or detailed descriptions that semantically define this route. "
                    "These patterns are used to generate embeddings for similarity comparison with incoming queries."
    )
    target_module_id: Optional[str] = Field(
        None,
        description="The `component_id` of the Vishustra module or component to route to if this destination is chosen. "
                    "If `None`, it implies this route is a conceptual destination or requires further interpretation "
                    "by a subsequent routing stage."
    )
    metadata: Dict[str, Any] = Field({}, description="Additional arbitrary metadata associated with this destination.")

    def __hash__(self):
        """Enable hashing for use in sets/dicts, based on `destination_id`."""
        return hash(self.destination_id)

    def __eq__(self, other):
        """Enable equality comparison based on `destination_id`."""
        if not isinstance(other, RouteDestination):
            return NotImplemented
        return self.destination_id == other.destination_id


class SemanticRouterConfig(BaseModel):
    """
    Configuration model for the SemanticRouter.
    Defines parameters controlling the behavior of the semantic routing mechanism.
    """
    embedding_model_id: str = Field(
        ...,
        description="The ID of the embedding model to use for semantic similarity checks. "
                    "This ID must be recognized and supported by the Vishustra `EmbeddingsProvider`."
    )
    similarity_threshold: float = Field(
        0.75,
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold. A route's similarity score must be "
                    "equal to or exceed this value to be considered a valid match."
    )
    top_k_routes: int = Field(
        1,
        ge=1,
        description="The maximum number of top semantically similar routes to return. "
                    "If `top_k_routes` > 1, the router may return multiple potential destinations, "
                    "sorted by similarity, that meet the `similarity_threshold`."
    )
    fallback_module_id: Optional[str] = Field(
        None,
        description="Optional `component_id` of a Vishustra module to route to if no "
                    "suitable semantic route is found (i.e., no route meets the threshold). "
                    "If `None`, a `RoutingError` will be raised if no match is found."
    )


class SemanticRouter(BaseComponent):
    """
    The SemanticRouter intelligently directs incoming natural language queries or requests
    to the most semantically relevant Vishustra module(s) or destination(s).

    It operates by computing the embedding of the input query and comparing it against
    pre-computed embeddings of defined route destinations. This enables highly flexible,
    content-aware routing that is resilient to variations in user phrasing, without
    requiring explicit keyword matching or complex rule sets.

    Usage:
    1. Instantiate with `component_id`, `config`, `destinations`, and an `EmbeddingsProvider`.
    2. Call `await initialize()` once to pre-compute destination embeddings.
    3. Call `await route(query)` for each incoming request.
    """
    config: SemanticRouterConfig = Field(..., description="Configuration settings for this router instance.")
    destinations: List[RouteDestination] = Field(..., min_items=1, description="List of all possible route destinations.")

    # Internal state, marked as exclude=True so Pydantic does not attempt to serialize these.
    _embeddings_provider: EmbeddingsProvider
    _destination_embeddings: Dict[str, np.ndarray] = Field(default_factory=dict, exclude=True)
    _is_initialized: bool = Field(False, exclude=True)

    def __init__(self, _embeddings_provider: EmbeddingsProvider, **data: Any):
        """
        Initializes the SemanticRouter.

        Args:
            _embeddings_provider: An instance of `EmbeddingsProvider` used to generate text embeddings.
                                  This is explicitly passed for dependency injection.
            **data: Additional keyword arguments passed to the `BaseComponent` and `BaseModel` constructor,
                    typically including `component_id`, `config`, and `destinations`.
        """
        if not isinstance(_embeddings_provider, EmbeddingsProvider):
            raise TypeError("`_embeddings_provider` must be an instance of `EmbeddingsProvider`.")
        self._embeddings_provider = _embeddings_provider
        super().__init__(**data)

    async def initialize(self) -> None:
        """
        Asynchronously initializes the router by pre-computing and storing embeddings
        for all defined route destinations. This method should be called once after
        instantiation and before any routing operations.
        """
        if self._is_initialized:
            logger.warning(f"SemanticRouter '{self.component_id}' is already initialized. Skipping re-initialization.")
            return

        logger.info(f"Initializing SemanticRouter '{self.component_id}' with {len(self.destinations)} destinations...")
        all_patterns_to_embed = []
        # Map each pattern back to its original destination_id.
        # This handles cases where multiple patterns belong to the same destination.
        pattern_to_destination_id_map = {}

        for dest in self.destinations:
            for pattern in dest.patterns:
                all_patterns_to_embed.append(pattern)
                pattern_to_destination_id_map[pattern] = dest.destination_id

        if not all_patterns_to_embed:
            raise RoutingError(
                f"SemanticRouter '{self.component_id}' cannot initialize: no patterns found across all destinations."
            )

        try:
            # Generate embeddings for all unique patterns from all destinations.
            pattern_embeddings_list = await self._embeddings_provider.get_embeddings(
                texts=all_patterns_to_embed,
                model_id=self.config.embedding_model_id
            )
            if len(pattern_embeddings_list) != len(all_patterns_to_embed):
                raise RoutingError(
                    f"Mismatch in number of patterns ({len(all_patterns_to_embed)}) and "
                    f"generated embeddings ({len(pattern_embeddings_list)})."
                )

            # Aggregate and average embeddings per destination_id for robustness.
            # Using the average embedding for a destination's patterns creates a more general
            # representation of that destination's semantic space.
            destination_raw_embeddings: Dict[str, List[np.ndarray]] = {}
            for i, pattern_embedding in enumerate(pattern_embeddings_list):
                original_pattern = all_patterns_to_embed[i]
                destination_id = pattern_to_destination_id_map[original_pattern]
                if destination_id not in destination_raw_embeddings:
                    destination_raw_embeddings[destination_id] = []
                destination_raw_embeddings[destination_id].append(np.array(pattern_embedding))

            for dest_id, embs in destination_raw_embeddings.items():
                # Compute the mean embedding for each destination.
                mean_embedding = np.mean(embs, axis=0)
                # Normalize the mean embedding to unit length for cosine similarity.
                self._destination_embeddings[dest_id] = mean_embedding / np.linalg.norm(mean_embedding)

            logger.info(
                f"SemanticRouter '{self.component_id}' initialized successfully. "
                f"{len(self._destination_embeddings)} unique destination embeddings computed."
            )
            self._is_initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize SemanticRouter '{self.component_id}': {e}", exc_info=True)
            raise RoutingError(f"Initialization failed for router '{self.component_id}'.") from e

    async def route(self, query: str) -> List[Dict[str, Any]]:
        """
        Asynchronously routes an incoming natural language query to the most semantically
        similar destination(s) based on pre-computed destination embeddings.

        Args:
            query: The input query string from the user or another component.

        Returns:
            A list of dictionaries, where each dictionary represents a potential route.
            Each route dictionary includes:
            - "destination_id": The unique ID of the chosen `RouteDestination`.
            - "target_module_id": The `component_id` of the target Vishustra module (if specified).
            - "similarity": The cosine similarity score between the query and the destination.
            - "metadata": Any additional metadata associated with the `RouteDestination`.
            The list is sorted by similarity in descending order, up to `top_k_routes`.

        Raises:
            RoutingError: If the router is not initialized, if the query is empty and no
                          fallback is defined, or if no suitable route is found and no
                          fallback module is configured.
        """
        if not self._is_initialized:
            raise RoutingError(
                f"SemanticRouter '{self.component_id}' has not been initialized. "
                "Call `.initialize()` method before attempting to route queries."
            )
        if not query or not query.strip():
            logger.warning("Received an empty or whitespace-only query for routing. Returning no match.")
            if self.config.fallback_module_id:
                return [{
                    "destination_id": "fallback_empty_query",
                    "target_module_id": self.config.fallback_module_id,
                    "similarity": 0.0,
                    "metadata": {"reason": "empty_query"}
                }]
            raise RoutingError("Cannot route an empty query and no fallback defined for router "
                               f"'{self.component_id}'.")

        logger.debug(f"Attempting to route query: '{query[:100]}{'...' if len(query) > 100 else ''}'")

        try:
            # Get the embedding for the incoming query.
            query_embedding_list = await self._embeddings_provider.get_embeddings(
                texts=[query],
                model_id=self.config.embedding_model_id
            )
            if not query_embedding_list or len(query_embedding_list[0]) == 0:
                raise RoutingError("Failed to obtain a valid embedding for the query.")

            query_embedding = np.array(query_embedding_list[0])
            # Normalize the query embedding for accurate cosine similarity calculation.
            query_embedding = query_embedding / np.linalg.norm(query_embedding)

            similarity_scores = []
            for dest in self.destinations:
                dest_emb = self._destination_embeddings.get(dest.destination_id)
                if dest_emb is None:
                    logger.warning(f"Embedding for destination '{dest.destination_id}' not found. Skipping.")
                    continue

                # Calculate cosine similarity between query and destination embedding.
                similarity = np.dot(query_embedding, dest_emb)
                similarity_scores.append((dest.destination_id, similarity))

            # Sort destinations by similarity score in descending order.
            similarity_scores.sort(key=lambda x: x[1], reverse=True)

            potential_routes: List[Dict[str, Any]] = []
            # Keep track of destination_ids already added to `potential_routes` to avoid duplicates.
            matched_destination_ids = set()

            for dest_id, score in similarity_scores:
                # Only consider routes that meet or exceed the similarity threshold.
                if score >= self.config.similarity_threshold:
                    if len(potential_routes) < self.config.top_k_routes:
                        # Retrieve the full RouteDestination object to get all its properties.
                        dest_obj = next((d for d in self.destinations if d.destination_id == dest_id), None)
                        if dest_obj and dest_id not in matched_destination_ids:
                            potential_routes.append({
                                "destination_id": dest_obj.destination_id,
                                "target_module_id": dest_obj.target_module_id,
                                "similarity": float(score), # Ensure JSON serializable
                                "metadata": dest_obj.metadata
                            })
                            matched_destination_ids.add(dest_id)
                # Stop if we've found enough top K routes that also meet the threshold.
                if len(potential_routes) >= self.config.top_k_routes:
                    break

            if not potential_routes:
                logger.info(
                    f"No suitable semantic route found for query: '{query[:100]}...' "
                    f"(threshold: {self.config.similarity_threshold})."
                )
                if self.config.fallback_module_id:
                    logger.info(f"Falling back to module '{self.config.fallback_module_id}'.")
                    return [{
                        "destination_id": "fallback",
                        "target_module_id": self.config.fallback_module_id,
                        "similarity": 0.0,
                        "metadata": {"reason": "no_semantic_match"}
                    }]
                else:
                    raise RoutingError(
                        f"No semantic route found for query and no fallback module configured for router "
                        f"'{self.component_id}': '{query}'"
                    )

            logger.info(
                f"Routed query to {len(potential_routes)} destination(s) for '{query[:100]}...': "
                f"{', '.join([r['destination_id'] for r in potential_routes])}"
            )
            return potential_routes

        except ValidationError as e:
            logger.error(
                f"Pydantic validation error encountered during routing for '{self.component_id}': {e}",
                exc_info=True
            )
            raise RoutingError(
                f"Configuration or data validation error during routing for '{self.component_id}'."
            ) from e
        except Exception as e:
            logger.error(
                f"An unexpected error occurred during routing for '{self.component_id}': {e}",
                exc_info=True
            )
            raise RoutingError(
                f"An unexpected error occurred during routing for '{self.component_id}'."
            ) from e