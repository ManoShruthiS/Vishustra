import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

from pydantic import BaseModel, Field, ValidationError, validate_call

# Initialize logging for the module
logger = logging.getLogger(__name__)

T = TypeVar('T')

class RouteDefinition(BaseModel):
    """
    Defines a potential route or target within the Vishustra framework.
    This includes a unique identifier, a descriptive text for semantic matching,
    and optional metadata.
    """
    identifier: str = Field(..., description="A unique identifier for this route (e.g., 'finance_analyzer', 'customer_support_bot').")
    description: str = Field(..., description="A detailed natural language description of what this route does or when it should be used. This text is used for embedding and semantic matching.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional arbitrary metadata associated with the route.")

    class Config:
        frozen = True # Route definitions should be immutable after creation

class RouteMatch(BaseModel):
    """
    Represents a potential match found by the SemanticRouter, including the
    matched route definition, its similarity score to the query, and a derived confidence.
    """
    route: RouteDefinition = Field(..., description="The matched RouteDefinition.")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="The similarity score (e.g., cosine similarity) between the query and the route's description.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="A derived confidence score, potentially scaled or transformed from the similarity score.")

    class Config:
        frozen = True # Route matches should be immutable

class SemanticRouterError(Exception):
    """Base exception for errors in the SemanticRouter module."""
    pass

class InitializationError(SemanticRouterError):
    """Raised when the SemanticRouter encounters an error during initialization."""
    pass

class RoutingError(SemanticRouterError):
    """Raised when the SemanticRouter encounters an error during routing."""
    pass

class BaseEmbeddingModel(ABC):
    """
    Abstract Base Class for embedding models.
    Vishustra components requiring text embeddings should depend on this interface.
    """
    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """
        Generates a single embedding vector for the given text.

        Args:
            text: The input text to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        raise NotImplementedError

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embedding vectors for a batch of texts.

        Args:
            texts: A list of input texts to embed.

        Returns:
            A list of embedding vectors, where each inner list corresponds to an input text.
        """
        raise NotImplementedError

    @abstractmethod
    async def dimensions(self) -> int:
        """
        Returns the dimension of the embedding vectors produced by this model.
        """
        raise NotImplementedError

class BaseVectorStore(ABC):
    """
    Abstract Base Class for vector databases or similarity search indexes.
    Vishustra components requiring vector storage and retrieval should depend on this interface.
    """
    @abstractmethod
    async def add_vectors(self,
                          vectors: List[List[float]],
                          payloads: List[Dict[str, Any]],
                          ids: Optional[List[str]] = None) -> List[str]:
        """
        Adds vectors to the store.

        Args:
            vectors: A list of embedding vectors to add.
            payloads: A list of dictionaries, where each dictionary is metadata
                      associated with the corresponding vector.
                      Each payload must contain 'identifier' and 'description' fields.
            ids: Optional list of explicit identifiers for the vectors. If None,
                 the store should generate them.

        Returns:
            A list of identifiers for the added vectors.
        """
        raise NotImplementedError

    @abstractmethod
    async def search(self,
                     query_vector: List[float],
                     top_k: int = 5,
                     filter: Optional[Dict[str, Any]] = None) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Searches the vector store for the most similar vectors to the query vector.

        Args:
            query_vector: The embedding vector of the query.
            top_k: The number of top similar results to return.
            filter: Optional dictionary to filter results based on metadata.

        Returns:
            A list of tuples, where each tuple contains:
            (vector_id: str, similarity_score: float, associated_payload: Dict[str, Any]).
            The payload will contain the 'identifier' and 'description' of the original RouteDefinition.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_vectors(self, ids: List[str]) -> None:
        """
        Deletes vectors from the store by their IDs.

        Args:
            ids: A list of vector IDs to delete.
        """
        raise NotImplementedError

    @abstractmethod
    async def count_vectors(self) -> int:
        """
        Returns the total number of vectors currently in the store.
        """
        raise NotImplementedError

class SemanticRouter:
    """
    The Vishustra SemanticRouter routes incoming natural language queries to
    the most semantically relevant predefined routes. It leverages an embedding model
    and a vector store to perform similarity search against route descriptions.

    This enables dynamic and intelligent orchestration of LLM chains, tools,
    or agents based on the user's intent.
    """
    @validate_call
    def __init__(self,
                 embedding_model: BaseEmbeddingModel,
                 vector_store: BaseVectorStore,
                 similarity_to_confidence_threshold: float = 0.75):
        """
        Initializes the SemanticRouter.

        Args:
            embedding_model: An instance of a class implementing BaseEmbeddingModel,
                             used to convert text into vector embeddings.
            vector_store: An instance of a class implementing BaseVectorStore,
                          used to store and search for route embeddings.
            similarity_to_confidence_threshold: The similarity score at which
                                                confidence starts scaling linearly
                                                from 0 up to 1. Scores below this
                                                threshold result in 0 confidence.
                                                Must be between 0 and 1.
        """
        if not (0.0 <= similarity_to_confidence_threshold <= 1.0):
            raise ValueError("similarity_to_confidence_threshold must be between 0 and 1.")

        self._embedding_model = embedding_model
        self._vector_store = vector_store
        self._similarity_to_confidence_threshold = similarity_to_confidence_threshold
        self._initialized = False
        self._route_id_to_vector_id: Dict[str, str] = {} # Maps RouteDefinition.identifier to vector_store_id

    async def init(self, initial_routes: List[RouteDefinition], overwrite_existing: bool = False) -> None:
        """
        Initializes the router by embedding the provided initial routes and adding them to the
        vector store. This method must be called before calling `route()`.

        Args:
            initial_routes: A list of RouteDefinition objects to initially populate the router.
            overwrite_existing: If True, existing routes in the vector store will be cleared
                                before adding the new ones. Use with caution.

        Raises:
            InitializationError: If an error occurs during embedding or vector store operations.
        """
        if self._initialized and not overwrite_existing:
            logger.warning("SemanticRouter already initialized. Call with overwrite_existing=True to re-initialize.")
            return

        logger.info("Initializing SemanticRouter...")
        try:
            if overwrite_existing:
                logger.info("Overwrite_existing is True. Deleting all existing vectors from the store.")
                # This requires an advanced vector store that can delete all or by a broad query.
                # For simplicity, if we don't have a direct 'clear_all' method, we might need to
                # query all IDs first, then delete them. Assuming `delete_vectors` can handle this.
                # A more robust implementation would rely on the specific vector store's API.
                # For now, let's assume `_route_id_to_vector_id` is the source of truth for IDs.
                if self._route_id_to_vector_id:
                    await self._vector_store.delete_vectors(list(self._route_id_to_vector_id.values()))
                self._route_id_to_vector_id.clear()


            await self.add_routes(initial_routes)
            self._initialized = True
            logger.info(f"SemanticRouter initialized with {await self._vector_store.count_vectors()} routes.")
        except Exception as e:
            logger.exception("Failed to initialize SemanticRouter.")
            self._initialized = False # Ensure _initialized is false if setup fails
            raise InitializationError(f"Error during SemanticRouter initialization: {e}") from e

    def _ensure_initialized(self):
        """Internal helper to ensure the router has been initialized."""
        if not self._initialized:
            raise RoutingError("SemanticRouter has not been initialized. Call init() first.")

    @validate_call
    async def add_routes(self, routes: List[RouteDefinition]) -> None:
        """
        Adds new routes to the semantic router. The router will embed these
        route descriptions and store them in the underlying vector store.

        Args:
            routes: A list of RouteDefinition objects to add.

        Raises:
            RoutingError: If routes cannot be added (e.g., embedding failure, vector store error).
            ValueError: If a route with the same identifier already exists.
        """
        if not routes:
            return

        new_route_identifiers = [r.identifier for r in routes]
        if any(ident in self._route_id_to_vector_id for ident in new_route_identifiers):
            # This check is basic; for large scale, might need vector store's 'get by ID'
            raise ValueError("One or more route identifiers already exist. Use update_routes or ensure unique IDs.")

        descriptions = [route.description for route in routes]
        logger.debug(f"Adding {len(routes)} new routes to the router...")

        try:
            embeddings = await self._embedding_model.embed_batch(descriptions)
            payloads = []
            for route in routes:
                payloads.append({
                    "identifier": route.identifier,
                    "description": route.description,
                    "metadata": route.metadata
                })

            vector_ids = await self._vector_store.add_vectors(
                vectors=embeddings,
                payloads=payloads,
                ids=[route.identifier for route in routes] # Use route identifier as vector ID for easy mapping
            )
            for i, route in enumerate(routes):
                self._route_id_to_vector_id[route.identifier] = vector_ids[i]

            logger.info(f"Successfully added {len(routes)} routes.")
        except Exception as e:
            logger.exception("Failed to add routes to SemanticRouter.")
            raise RoutingError(f"Error adding routes: {e}") from e

    @validate_call
    async def remove_routes(self, identifiers: List[str]) -> None:
        """
        Removes routes from the semantic router by their identifiers.

        Args:
            identifiers: A list of route identifiers to remove.

        Raises:
            RoutingError: If routes cannot be removed (e.g., vector store error).
        """
        if not identifiers:
            return

        vector_ids_to_delete = []
        for ident in identifiers:
            if ident in self._route_id_to_vector_id:
                vector_ids_to_delete.append(self._route_id_to_vector_id[ident])
            else:
                logger.warning(f"Attempted to remove non-existent route identifier: {ident}")

        if not vector_ids_to_delete:
            logger.info("No routes found to remove based on provided identifiers.")
            return

        logger.debug(f"Removing {len(vector_ids_to_delete)} routes from the router...")
        try:
            await self._vector_store.delete_vectors(vector_ids_to_delete)
            for ident in identifiers:
                self._route_id_to_vector_id.pop(ident, None) # Remove gracefully if not found
            logger.info(f"Successfully removed {len(vector_ids_to_delete)} routes.")
        except Exception as e:
            logger.exception("Failed to remove routes from SemanticRouter.")
            raise RoutingError(f"Error removing routes: {e}") from e

    @validate_call
    async def route(self,
                    query: str,
                    top_k: int = 1,
                    min_similarity: float = 0.7) -> List[RouteMatch]:
        """
        Routes an incoming natural language query to the most semantically relevant
        predefined routes.

        Args:
            query: The natural language query string.
            top_k: The maximum number of top matching routes to return.
            min_similarity: The minimum similarity score a route must have to be considered a match.
                            Scores below this threshold will not be returned.

        Returns:
            A sorted list of `RouteMatch` objects, ordered by confidence (descending).
            Returns an empty list if no suitable routes are found or if the query is empty.

        Raises:
            RoutingError: If the router is not initialized or an error occurs during routing.
            ValueError: If `query` is empty or `min_similarity` is out of range.
        """
        self._ensure_initialized()

        if not query.strip():
            logger.warning("Empty query provided to SemanticRouter. Returning empty list.")
            return []
        if not (0.0 <= min_similarity <= 1.0):
            raise ValueError("min_similarity must be between 0 and 1.")
        if top_k <= 0:
            logger.warning("top_k must be greater than 0. Returning empty list.")
            return []

        try:
            query_embedding = await self._embedding_model.embed(query)
            if not query_embedding:
                raise RoutingError("Embedding model returned an empty vector for the query.")

            search_results = await self._vector_store.search(
                query_vector=query_embedding,
                top_k=top_k
            )

            matched_routes: List[RouteMatch] = []
            for vector_id, score, payload in search_results:
                if score >= min_similarity:
                    try:
                        # Reconstruct RouteDefinition from payload
                        route_def = RouteDefinition(
                            identifier=payload["identifier"],
                            description=payload["description"],
                            metadata=payload.get("metadata", {})
                        )
                        confidence = self._calculate_confidence(score)
                        matched_routes.append(
                            RouteMatch(route=route_def, similarity_score=score, confidence=confidence)
                        )
                    except ValidationError as e:
                        logger.error(f"Failed to validate RouteDefinition from vector store payload (ID: {vector_id}): {e}")
                    except KeyError as e:
                        logger.error(f"Missing expected key in vector store payload for ID {vector_id}: {e}. Payload: {payload}")


            # Sort by confidence descending
            matched_routes.sort(key=lambda match: match.confidence, reverse=True)
            logger.debug(f"Routed query '{query[:50]}...' to {len(matched_routes)} potential routes.")
            return matched_routes

        except Exception as e:
            logger.exception(f"Error during semantic routing for query: '{query[:100]}...'")
            raise RoutingError(f"Failed to route query: {e}") from e

    def _calculate_confidence(self, similarity_score: float) -> float:
        """
        Calculates a confidence score based on the similarity score and the
        router's configured threshold.

        Args:
            similarity_score: The raw similarity score (0.0 to 1.0).

        Returns:
            A confidence score (0.0 to 1.0).
        """
        if similarity_score < self._similarity_to_confidence_threshold:
            return 0.0
        # Linearly scale confidence from 0 to 1 as similarity goes from threshold to 1
        scaling_range = 1.0 - self._similarity_to_confidence_threshold
        if scaling_range <= 0: # Avoid division by zero if threshold is 1.0
             return 1.0 if similarity_score >= 1.0 else 0.0

        scaled_score = (similarity_score - self._similarity_to_confidence_threshold) / scaling_range
        return min(max(scaled_score, 0.0), 1.0) # Ensure it's within [0, 1]

    async def close(self) -> None:
        """
        Cleans up resources if necessary. This method should be called when the router
        is no longer needed to ensure proper resource release.
        """
        logger.info("Closing SemanticRouter resources...")
        # Currently, BaseEmbeddingModel and BaseVectorStore don't define a close() method.
        # If they did, it would be called here.
        # Example:
        # if hasattr(self._embedding_model, 'close') and callable(self._embedding_model.close):
        #     await self._embedding_model.close()
        # if hasattr(self._vector_store, 'close') and callable(self._vector_store.close):
        #     await self._vector_store.close()
        self._initialized = False
        logger.info("SemanticRouter closed.")

# Example Usage (not part of the module, but for demonstration)
# class MockEmbeddingModel(BaseEmbeddingModel):
#     async def embed(self, text: str) -> List[float]:
#         # Simple hash-based mock embedding for demonstration
#         return [float(ord(c)) / 100 for c in text[:16].ljust(16)] # Fixed size 16
#
#     async def embed_batch(self, texts: List[str]) -> List[List[float]]:
#         return [await self.embed(text) for text in texts]
#
#     async def dimensions(self) -> int:
#         return 16
#
# class MockVectorStore(BaseVectorStore):
#     def __init__(self):
#         self._store: Dict[str, Tuple[List[float], Dict[str, Any]]] = {}
#
#     async def add_vectors(self, vectors: List[List[float]], payloads: List[Dict[str, Any]], ids: Optional[List[str]] = None) -> List[str]:
#         generated_ids = []
#         for i, vec in enumerate(vectors):
#             _id = ids[i] if ids and ids[i] else f"vec_{len(self._store)}_{payloads[i]['identifier']}"
#             self._store[_id] = (vec, payloads[i])
#             generated_ids.append(_id)
#         return generated_ids
#
#     async def search(self, query_vector: List[float], top_k: int = 5, filter: Optional[Dict[str, Any]] = None) -> List[Tuple[str, float, Dict[str, Any]]]:
#         results = []
#         for _id, (vec, payload) in self._store.items():
#             # Simple dot product similarity for mock
#             similarity = sum(qv * vv for qv, vv in zip(query_vector, vec)) / (sum(x*x for x in query_vector)**0.5 * sum(x*x for x in vec)**0.5 + 1e-9)
#             results.append((_id, similarity, payload))
#
#         results.sort(key=lambda x: x[1], reverse=True)
#         return results[:top_k]
#
#     async def delete_vectors(self, ids: List[str]) -> None:
#         for _id in ids:
#             self._store.pop(_id, None)
#
#     async def count_vectors(self) -> int:
#         return len(self._store)
#
# async def main():
#     logging.basicConfig(level=logging.INFO)
#
#     embedding_model = MockEmbeddingModel()
#     vector_store = MockVectorStore()
#     router = SemanticRouter(embedding_model=embedding_model, vector_store=vector_store, similarity_to_confidence_threshold=0.7)
#
#     routes = [
#         RouteDefinition(identifier="weather_tool", description="Provides current weather information for a given city."),
#         RouteDefinition(identifier="stock_analyzer", description="Analyzes stock prices, company financials, and market trends."),
#         RouteDefinition(identifier="customer_support_qa", description="Answers common customer support questions about products and orders."),
#         RouteDefinition(identifier="joke_generator", description="Generates humorous jokes based on various topics.")
#     ]
#
#     await router.init(initial_routes=routes)
#
#     queries = [
#         "What's the weather like in London tomorrow?",
#         "Tell me a funny story.",
#         "Analyze Tesla's recent stock performance.",
#         "How do I return a product?",
#         "What are the capital cities of Europe?", # Should not match well
#         "" # Empty query
#     ]
#
#     for q in queries:
#         print(f"\nQuery: '{q}'")
#         try:
#             matches = await router.route(q, top_k=3, min_similarity=0.6)
#             if matches:
#                 for match in matches:
#                     print(f"  Match: {match.route.identifier}, Similarity: {match.similarity_score:.2f}, Confidence: {match.confidence:.2f}")
#             else:
#                 print("  No suitable routes found.")
#         except SemanticRouterError as e:
#             print(f"  Routing Error: {e}")
#
#     await router.add_routes([
#         RouteDefinition(identifier="news_summarizer", description="Summarizes recent news articles on a given topic.")
#     ])
#
#     print("\nQuery after adding new route: 'Summarize today's headlines.'")
#     matches = await router.route("Summarize today's headlines.", top_k=1)
#     if matches:
#         for match in matches:
#             print(f"  Match: {match.route.identifier}, Similarity: {match.similarity_score:.2f}, Confidence: {match.confidence:.2f}")
#
#     await router.remove_routes(["joke_generator"])
#     print("\nQuery after removing 'joke_generator': 'Tell me a joke.'")
#     matches = await router.route("Tell me a joke.", top_k=1)
#     if matches:
#         for match in matches:
#             print(f"  Match: {match.route.identifier}, Similarity: {match.similarity_score:.2f}, Confidence: {match.confidence:.2f}")
#     else:
#         print("  No suitable routes found (joke_generator removed).")
#
#     await router.close()
#
# if __name__ == "__main__":
#     asyncio.run(main())