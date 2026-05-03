import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

class AbstractEmbeddingModel(ABC):
    """
    Abstract base class for embedding models.
    Vishustra components requiring embeddings should depend on this interface.
    """
    @abstractmethod
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Asynchronously embeds a list of texts into a list of embedding vectors.

        Args:
            texts: A list of strings to embed.

        Returns:
            A list of lists of floats, where each inner list represents an embedding
            vector for the corresponding input text.
        """
        raise NotImplementedError

class AbstractVectorStore(ABC):
    """
    Abstract base class for vector stores.
    Vishustra components requiring vector storage and retrieval should depend on this interface.
    """
    @abstractmethod
    async def add_vectors(self, vectors: List[List[float]], metadatas: List[Dict[str, Any]]) -> None:
        """
        Asynchronously adds a batch of vectors with associated metadata to the vector store.

        Args:
            vectors: A list of embedding vectors to add.
            metadatas: A list of dictionaries, where each dictionary contains metadata
                       corresponding to the vector at the same index.
        """
        raise NotImplementedError

    @abstractmethod
    async def search(self, query_vector: List[float], k: int = 1) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Asynchronously searches the vector store for the `k` most similar vectors
        to the given query vector.

        Args:
            query_vector: The embedding vector to query with.
            k: The number of top similar results to retrieve.

        Returns:
            A list of tuples, where each tuple contains (similarity_score, metadata_dict).
            The list is ordered by similarity score in descending order.
        """
        raise NotImplementedError

@dataclass(frozen=True)
class Route:
    """
    Represents a specific route within the semantic router.

    A route defines a target (e.g., a specific tool, agent, or processing pipeline)
    and provides examples of user utterances that should map to this target.

    Attributes:
        name: A unique identifier for the route (e.g., "customer_service_query").
        description: A brief explanation of what this route handles.
        utterances: A list of example phrases or sentences that should trigger this route.
        target: The name or identifier of the component/function this route dispatches to.
                This could be a function name, an agent ID, a topic string, etc.
    """
    name: str
    description: str
    utterances: List[str]
    target: str

    def __post_init__(self):
        if not self.name or not isinstance(self.name, str):
            raise ValueError("Route name must be a non-empty string.")
        if not self.description or not isinstance(self.description, str):
            raise ValueError("Route description must be a non-empty string.")
        if not self.utterances or not all(isinstance(u, str) for u in self.utterances):
            raise ValueError("Route utterances must be a non-empty list of strings.")
        if not self.target or not isinstance(self.target, str):
            raise ValueError("Route target must be a non-empty string.")


@dataclass
class SemanticRouterConfig:
    """
    Configuration options for the SemanticRouter.

    Attributes:
        top_k: The number of top similar vectors to retrieve from the vector store
               during a search operation.
        similarity_threshold: The minimum similarity score required for a route
                              to be considered a match. Routes with scores below
                              this threshold will be ignored.
        embedding_batch_size: The number of texts to embed in a single batch.
                              Adjusting this can impact performance and memory usage
                              depending on the embedding model.
        fallback_route_name: The name of a predefined route to use if no other
                             route meets the similarity threshold. If None,
                             the router will return None if no match is found.
    """
    top_k: int = 3
    similarity_threshold: float = 0.75
    embedding_batch_size: int = 32
    fallback_route_name: Optional[str] = None

class SemanticRouter:
    """
    A highly modular and asynchronous semantic router for Vishustra.

    This router dispatches incoming text queries to predefined routes based on
    the semantic similarity between the query and example utterances associated
    with each route. It leverages an `AbstractEmbeddingModel` for text embeddings
    and an `AbstractVectorStore` for efficient similarity search.

    The router's initialization is asynchronous, performing all embedding and
    vector store population in a background task, ensuring the router is ready
    before handling requests.

    Args:
        embedding_model: An instance of `AbstractEmbeddingModel` to generate embeddings.
        vector_store: An instance of `AbstractVectorStore` to store and search
                      route utterance embeddings.
        routes: A list of `Route` objects defining the available routing paths.
        config: An optional `SemanticRouterConfig` instance to customize router behavior.
                If not provided, a default configuration will be used.

    Raises:
        ValueError: If `routes` is empty or invalid, or if `fallback_route_name`
                    is provided but doesn't exist in `routes`.
    """
    def __init__(
        self,
        embedding_model: AbstractEmbeddingModel,
        vector_store: AbstractVectorStore,
        routes: List[Route],
        config: Optional[SemanticRouterConfig] = None,
    ):
        if not routes:
            raise ValueError("SemanticRouter must be initialized with at least one route.")

        self._embedding_model = embedding_model
        self._vector_store = vector_store
        self._routes = routes
        self.config = config if config else SemanticRouterConfig()

        self._route_map: Dict[str, Route] = {route.name: route for route in self._routes}
        self._vector_store_ready = asyncio.Event()

        if self.config.fallback_route_name and self.config.fallback_route_name not in self._route_map:
            raise ValueError(
                f"Fallback route '{self.config.fallback_route_name}' not found in provided routes."
            )

        # Start the asynchronous initialization in a background task
        asyncio.create_task(self._init_vector_store())
        logger.info("SemanticRouter initialized. Populating vector store in background...")

    async def _init_vector_store(self) -> None:
        """
        Asynchronously initializes the vector store by embedding all route utterances
        and adding them to the store.
        """
        all_utterances: List[str] = []
        utterance_metadatas: List[Dict[str, str]] = []

        for route in self._routes:
            for utterance in route.utterances:
                all_utterances.append(utterance)
                utterance_metadatas.append({"route_name": route.name, "original_utterance": utterance})

        try:
            # Embed utterances in batches
            vectors: List[List[float]] = []
            for i in range(0, len(all_utterances), self.config.embedding_batch_size):
                batch_texts = all_utterances[i : i + self.config.embedding_batch_size]
                batch_vectors = await self._embedding_model.embed(batch_texts)
                vectors.extend(batch_vectors)
                logger.debug(f"Embedded batch {i // self.config.embedding_batch_size + 1} of {len(all_utterances) // self.config.embedding_batch_size + 1}")

            if vectors:
                await self._vector_store.add_vectors(vectors, utterance_metadatas)
                logger.info(f"Successfully populated vector store with {len(vectors)} embeddings for {len(self._routes)} routes.")
            else:
                logger.warning("No utterances found to embed for routes.")

        except Exception as e:
            logger.error(f"Failed to initialize SemanticRouter vector store: {e}")
        finally:
            self._vector_store_ready.set() # Signal that initialization is complete (or failed)

    async def _embed_query(self, query: str) -> List[float]:
        """
        Asynchronously embeds a single query string.

        Args:
            query: The text query to embed.

        Returns:
            The embedding vector for the query.
        """
        embeddings = await self._embedding_model.embed([query])
        return embeddings[0]

    async def _find_best_route(self, query_embedding: List[float]) -> Optional[Route]:
        """
        Asynchronously searches the vector store for the best matching route
        based on the query embedding and router configuration.

        Args:
            query_embedding: The embedding vector of the input query.

        Returns:
            The `Route` object that best matches the query, or None if no route
            meets the `similarity_threshold`.
        """
        search_results = await self._vector_store.search(query_embedding, k=self.config.top_k)
        logger.debug(f"Vector store search results for query: {search_results}")

        best_match: Optional[Tuple[float, Route]] = None

        for score, metadata in search_results:
            if score >= self.config.similarity_threshold:
                route_name = metadata.get("route_name")
                if route_name and route_name in self._route_map:
                    if best_match is None or score > best_match[0]:
                        best_match = (score, self._route_map[route_name])
                    logger.debug(f"Potential match: Route '{route_name}' with score {score:.2f} (above threshold {self.config.similarity_threshold:.2f})")
                else:
                    logger.warning(f"Search result returned unknown route_name: {route_name} or missing in map. Metadata: {metadata}")
            else:
                logger.debug(f"Discarding match with score {score:.2f} (below threshold {self.config.similarity_threshold:.2f})")
                # Since results are ordered, we can stop if we hit below threshold
                break

        if best_match:
            logger.info(f"Found best matching route '{best_match[1].name}' with score {best_match[0]:.2f}")
            return best_match[1]
        else:
            logger.info("No route found above the similarity threshold.")
            return None

    async def route(self, query: str) -> Optional[Route]:
        """
        Asynchronously routes an input query string to the most appropriate `Route`.

        This is the primary method to use for dispatching requests. It first waits
        for the internal vector store to be fully initialized.

        Args:
            query: The input text query to be routed.

        Returns:
            The `Route` object that the query is routed to, or None if no suitable
            route is found and no fallback is configured.
        """
        await self._vector_store_ready.wait() # Ensure the vector store is populated

        if not query.strip():
            logger.warning("Received empty query, returning None.")
            return None

        try:
            query_embedding = await self._embed_query(query)
            matched_route = await self._find_best_route(query_embedding)

            if matched_route:
                return matched_route
            elif self.config.fallback_route_name:
                fallback_route = self._route_map.get(self.config.fallback_route_name)
                if fallback_route:
                    logger.info(f"No direct match found, falling back to route: '{fallback_route.name}'")
                    return fallback_route
                else:
                    logger.error(f"Fallback route '{self.config.fallback_route_name}' configured but not found in map. This should not happen if `__init__` checks are correct.")
                    return None
            else:
                logger.info("No route found and no fallback route configured.")
                return None
        except Exception as e:
            logger.error(f"Error during semantic routing for query '{query}': {e}")
            return None

# --- Example Usage (for demonstration within the framework context) ---
# This part would typically be in a separate `examples/` or `tests/` directory
# but is included here to show how the component would be used.
#
# class MockEmbeddingModel(AbstractEmbeddingModel):
#     """A simple mock embedding model for testing."""
#     async def embed(self, texts: List[str]) -> List[List[float]]:
#         logger.debug(f"MockEmbeddingModel embedding {len(texts)} texts.")
#         # Return dummy embeddings (e.g., based on text length for simplistic tests)
#         return [[float(ord(c)) for c in text[:8].ljust(8, ' ')] for text in texts]
#
# class MockVectorStore(AbstractVectorStore):
#     """A simple in-memory mock vector store."""
#     def __init__(self):
#         self._store: List[Tuple[List[float], Dict[str, Any]]] = []
#
#     async def add_vectors(self, vectors: List[List[float]], metadatas: List[Dict[str, Any]]) -> None:
#         logger.debug(f"MockVectorStore adding {len(vectors)} vectors.")
#         for vec, meta in zip(vectors, metadatas):
#             self._store.append((vec, meta))
#
#     async def search(self, query_vector: List[float], k: int = 1) -> List[Tuple[float, Dict[str, Any]]]:
#         logger.debug(f"MockVectorStore searching for {k} results.")
#         if not self._store:
#             return []
#
#         # Simple dot product for similarity
#         def dot_product(vec1, vec2):
#             return sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
#
#         results = []
#         for stored_vec, metadata in self._store:
#             # Ensure vectors have same dimension for dot product
#             dim = min(len(query_vector), len(stored_vec))
#             score = dot_product(query_vector[:dim], stored_vec[:dim])
#             results.append((score, metadata))
#
#         results.sort(key=lambda x: x[0], reverse=True)
#         return results[:k]
#
# async def run_example():
#     logger.add("file.log", rotation="500 MB")
#     logger.level("DEBUG")
#     logger.info("Starting SemanticRouter example...")
#
#     # 1. Define routes
#     customer_service_route = Route(
#         name="customer_service",
#         description="Handles general customer service inquiries.",
#         utterances=["I have a problem with my order", "My product is broken", "I need help", "customer support"],
#         target="handle_customer_service"
#     )
#     sales_route = Route(
#         name="sales_inquiry",
#         description="Handles inquiries about purchasing products or services.",
#         utterances=["I want to buy something", "What are your prices?", "Can I get a quote?", "how much does it cost"],
#         target="handle_sales_inquiry"
#     )
#     general_chat_route = Route(
#         name="general_chat",
#         description="Handles casual greetings and non-specific conversations.",
#         utterances=["hello", "hi there", "how are you", "what's up"],
#         target="engage_general_chat"
#     )
#     unknown_route = Route(
#         name="unknown_intent",
#         description="Fallback route for unhandled queries.",
#         utterances=[], # Fallback routes typically don't need utterances
#         target="escalate_to_human"
#     )
#
#     all_routes = [customer_service_route, sales_route, general_chat_route, unknown_route]
#
#     # 2. Instantiate dependencies
#     embedding_model = MockEmbeddingModel()
#     vector_store = MockVectorStore()
#     config = SemanticRouterConfig(similarity_threshold=20.0, fallback_route_name="unknown_intent")
#
#     # 3. Instantiate the router
#     router = SemanticRouter(embedding_model, vector_store, all_routes, config)
#
#     # Wait for router to be ready
#     await router._vector_store_ready.wait()
#     logger.info("SemanticRouter is ready to process queries.")
#
#     # 4. Test queries
#     queries = [
#         "My item arrived damaged, I need assistance.",          # -> customer_service
#         "How can I purchase your premium subscription?",        # -> sales_inquiry
#         "Good morning, how's your day?",                        # -> general_chat
#         "What is the capital of France?",                       # -> unknown_intent
#         "I have a query about an existing service.",            # -> customer_service (less direct)
#         "Tell me about your latest offers."                     # -> sales_inquiry (less direct)
#     ]
#
#     for q in queries:
#         routed_to = await router.route(q)
#         target_name = routed_to.name if routed_to else "NO_ROUTE_FOUND"
#         target_action = routed_to.target if routed_to else "N/A"
#         logger.info(f"Query: '{q}' -> Routed to: '{target_name}' (Action: '{target_action}')")
#
#     logger.info("SemanticRouter example finished.")
#
# if __name__ == "__main__":
#     # To run the example, uncomment the example usage section above.
#     # asyncio.run(run_example())
#     print("Uncomment the 'Example Usage' section and 'if __name__ == \"__main__\":' block to run the demo.")
#     print("This file defines the SemanticRouter component for Vishustra.")