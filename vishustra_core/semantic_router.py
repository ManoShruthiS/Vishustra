import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Dict, Any, Union
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

class EmbeddingModel(ABC):
    """
    Abstract Base Class for any embedding model used by Vishustra.
    Concrete implementations should inherit from this class and provide
    async and sync methods for text embedding.
    """
    @abstractmethod
    async def aembed(self, texts: List[str]) -> List[List[float]]:
        """
        Asynchronously embeds a list of texts into dense vector representations.

        Args:
            texts: A list of strings to be embedded.

        Returns:
            A list of lists of floats, where each inner list is the embedding
            vector for the corresponding text.
        """
        pass

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Synchronously embeds a list of texts into dense vector representations.

        Args:
            texts: A list of strings to be embedded.

        Returns:
            A list of lists of floats, where each inner list is the embedding
            vector for the corresponding text.
        """
        pass

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """
        Returns the dimension of the embedding vectors produced by this model.
        """
        pass

class VectorStore(ABC):
    """
    Abstract Base Class for any vector store used by Vishustra.
    Concrete implementations should provide methods for adding, querying,
    and managing vectors, potentially within namespaces.
    """
    @abstractmethod
    async def aadd_vectors(self, vectors: List[List[float]], metadatas: List[Dict[str, Any]], namespace: Optional[str] = None):
        """
        Asynchronously adds vectors with associated metadata to the store.

        Args:
            vectors: A list of embedding vectors.
            metadatas: A list of dictionaries, where each dictionary contains
                       metadata for the corresponding vector.
            namespace: An optional string to partition vectors within the store.
        """
        pass

    @abstractmethod
    def add_vectors(self, vectors: List[List[float]], metadatas: List[Dict[str, Any]], namespace: Optional[str] = None):
        """
        Synchronously adds vectors with associated metadata to the store.

        Args:
            vectors: A list of embedding vectors.
            metadatas: A list of dictionaries, where each dictionary contains
                       metadata for the corresponding vector.
            namespace: An optional string to partition vectors within the store.
        """
        pass

    @abstractmethod
    async def aquery_vectors(self, query_vector: List[float], top_k: int = 1, namespace: Optional[str] = None) -> List[Tuple[Dict[str, Any], float]]:
        """
        Asynchronously queries the store for the most similar vectors to a given
        query vector.

        Args:
            query_vector: The embedding vector for the query.
            top_k: The number of top similar results to return.
            namespace: An optional string to limit the query to a specific partition.

        Returns:
            A list of tuples, where each tuple contains (metadata, similarity_score).
        """
        pass

    @abstractmethod
    def query_vectors(self, query_vector: List[float], top_k: int = 1, namespace: Optional[str] = None) -> List[Tuple[Dict[str, Any], float]]:
        """
        Synchronously queries the store for the most similar vectors to a given
        query vector.

        Args:
            query_vector: The embedding vector for the query.
            top_k: The number of top similar results to return.
            namespace: An optional string to limit the query to a specific partition.

        Returns:
            A list of tuples, where each tuple contains (metadata, similarity_score).
        """
        pass

    @abstractmethod
    async def adelete_namespace(self, namespace: str):
        """
        Asynchronously deletes all vectors within a specific namespace.

        Args:
            namespace: The namespace to delete.
        """
        pass

    @abstractmethod
    def delete_namespace(self, namespace: str):
        """
        Synchronously deletes all vectors within a specific namespace.

        Args:
            namespace: The namespace to delete.
        """
        pass

class Route(BaseModel):
    """
    Represents a specific routing target within Vishustra, defined by a name,
    a description, and a set of example queries that should map to it.
    """
    name: str = Field(..., description="A unique identifier for this route.")
    description: str = Field(..., description="A brief, human-readable description of what this route handles.")
    examples: List[str] = Field(..., min_length=1, description="A list of example queries or phrases that semantically map to this route.")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional arbitrary metadata associated with the route.")

    class Config:
        frozen = True # Routes are immutable once defined for stability

class SemanticRouter:
    """
    The SemanticRouter component intelligently routes incoming natural language
    queries to predefined 'routes' (e.g., specific tools, agents, or chains)
    based on their semantic similarity.

    It leverages an `EmbeddingModel` to vectorize queries and route examples,
    and a `VectorStore` to perform efficient similarity searches. This allows
    for flexible, intent-based routing without explicit keyword matching.
    """
    def __init__(self,
                 embedding_model: EmbeddingModel,
                 vector_store: VectorStore,
                 similarity_threshold: float = 0.75,
                 route_namespace: str = "vishustra-semantic-routes"):
        """
        Initializes the SemanticRouter.

        Args:
            embedding_model: An instance of an `EmbeddingModel` implementation
                             responsible for creating vector embeddings.
            vector_store: An instance of a `VectorStore` implementation where
                          route example embeddings are stored and queried.
            similarity_threshold: The minimum cosine similarity score required
                                  for a query to be confidently assigned to a route.
                                  Queries below this threshold will result in no route.
            route_namespace: A unique string to identify and isolate route examples
                             within the `VectorStore`. Useful if the vector store
                             is shared across multiple components.

        Raises:
            TypeError: If `embedding_model` or `vector_store` are not instances
                       of their respective abstract base classes.
        """
        if not isinstance(embedding_model, EmbeddingModel):
            raise TypeError("`embedding_model` must be an instance of `EmbeddingModel`.")
        if not isinstance(vector_store, VectorStore):
            raise TypeError("`vector_store` must be an instance of `VectorStore`.")
        if not (0.0 <= similarity_threshold <= 1.0):
            raise ValueError("`similarity_threshold` must be between 0.0 and 1.0.")
        if not isinstance(route_namespace, str) or not route_namespace.strip():
            raise ValueError("`route_namespace` must be a non-empty string.")

        self._embedding_model = embedding_model
        self._vector_store = vector_store
        self._similarity_threshold = similarity_threshold
        self._route_namespace = route_namespace
        self._registered_routes: Dict[str, Route] = {} # Caches Route objects by name

        logger.info(f"SemanticRouter initialized. Namespace: '{self._route_namespace}', "
                    f"Similarity Threshold: {self._similarity_threshold:.2f}")

    async def aadd_route(self, route: Route):
        """
        Asynchronously adds a new `Route` to the router.
        This involves embedding its example queries and storing them in the
        configured `VectorStore`. If a route with the same name already exists,
        it will be overwritten.

        Args:
            route: The `Route` object to add.

        Raises:
            TypeError: If the provided `route` is not an instance of `Route`.
            Exception: Propagates exceptions from embedding model or vector store
                       operations (e.g., API errors, network issues).
        """
        if not isinstance(route, Route):
            raise TypeError("`route` must be an instance of `Route`.")
        if not route.examples:
            logger.warning(f"Route '{route.name}' has no examples. It will not be routable.")
            self._registered_routes[route.name] = route
            return

        if route.name in self._registered_routes:
            logger.warning(f"Route '{route.name}' already exists. Its examples will be updated "
                           f"and any prior examples for this route might persist if not explicitly cleaned.")
            # In a real-world scenario, you might want to delete previous vectors for this route
            # or ensure the vector store supports upsert based on a unique ID.
            # For simplicity, we add new vectors and rely on subsequent queries finding the latest.

        try:
            example_embeddings = await self._embedding_model.aembed(route.examples)
            metadatas = [
                {"route_name": route.name, "original_example": example_text, "route_metadata": route.metadata}
                for example_text in route.examples
            ]

            await self._vector_store.aadd_vectors(
                vectors=example_embeddings,
                metadatas=metadatas,
                namespace=self._route_namespace
            )
            self._registered_routes[route.name] = route
            logger.info(f"Route '{route.name}' added/updated successfully with {len(route.examples)} examples.")
        except Exception as e:
            logger.error(f"Failed to add route '{route.name}': {e}", exc_info=True)
            raise

    def add_route(self, route: Route):
        """
        Synchronously adds a new `Route` to the router.
        This involves embedding its example queries and storing them in the
        configured `VectorStore`. If a route with the same name already exists,
        it will be overwritten.

        Args:
            route: The `Route` object to add.

        Raises:
            TypeError: If the provided `route` is not an instance of `Route`.
            Exception: Propagates exceptions from embedding model or vector store
                       operations.
        """
        if not isinstance(route, Route):
            raise TypeError("`route` must be an instance of `Route`.")
        if not route.examples:
            logger.warning(f"Route '{route.name}' has no examples. It will not be routable.")
            self._registered_routes[route.name] = route
            return

        if route.name in self._registered_routes:
            logger.warning(f"Route '{route.name}' already exists. Its examples will be updated "
                           f"and any prior examples for this route might persist if not explicitly cleaned.")

        try:
            example_embeddings = self._embedding_model.embed(route.examples)
            metadatas = [
                {"route_name": route.name, "original_example": example_text, "route_metadata": route.metadata}
                for example_text in route.examples
            ]

            self._vector_store.add_vectors(
                vectors=example_embeddings,
                metadatas=metadatas,
                namespace=self._route_namespace
            )
            self._registered_routes[route.name] = route
            logger.info(f"Route '{route.name}' added/updated successfully with {len(route.examples)} examples.")
        except Exception as e:
            logger.error(f"Failed to add route '{route.name}': {e}", exc_info=True)
            raise

    async def aroute_query(self, query: str) -> Optional[Tuple[Route, float]]:
        """
        Asynchronously routes an incoming natural language query to the most
        semantically similar registered route.

        The query is embedded, and a similarity search is performed against
        all stored route examples. If the highest similarity score meets or
        exceeds the `similarity_threshold`, the corresponding `Route` and
        score are returned. Otherwise, `None` is returned.

        Args:
            query: The input query string to route.

        Returns:
            A tuple of (`Route`, `similarity_score`) if a confident route is found,
            otherwise `None`.
        """
        if not query or not query.strip():
            logger.warning("Attempted to route an empty or whitespace-only query.")
            return None

        try:
            # Embed the query
            query_embedding = (await self._embedding_model.aembed([query]))[0]
            if not query_embedding:
                logger.error("Embedding model returned empty embedding for query, cannot route.")
                return None

            # Query the vector store for the top match
            results = await self._vector_store.aquery_vectors(
                query_vector=query_embedding,
                top_k=1,
                namespace=self._route_namespace
            )

            if not results:
                logger.debug(f"No semantic routes found in vector store for query: '{query[:100]}...'")
                return None

            # Evaluate the best match
            most_similar_metadata, similarity = results[0]
            if similarity >= self._similarity_threshold:
                route_name = most_similar_metadata.get("route_name")
                if route_name and route_name in self._registered_routes:
                    routed_route = self._registered_routes[route_name]
                    logger.debug(f"Query '{query[:100]}...' confidently routed to '{route_name}' "
                                 f"with similarity {similarity:.4f} (threshold: {self._similarity_threshold:.2f}).")
                    return routed_route, similarity
                else:
                    logger.warning(f"Vector store returned an unregistered route_name '{route_name}'. "
                                   f"Similarity: {similarity:.4f}. Route examples might be stale.")
            else:
                logger.debug(f"Query '{query[:100]}...' did not meet similarity threshold "
                             f"({self._similarity_threshold:.2f}). Best similarity: {similarity:.4f}.")
            return None

        except Exception as e:
            logger.error(f"Error during asynchronous query routing for '{query[:100]}...': {e}", exc_info=True)
            return None

    def route_query(self, query: str) -> Optional[Tuple[Route, float]]:
        """
        Synchronously routes an incoming natural language query to the most
        semantically similar registered route.

        The query is embedded, and a similarity search is performed against
        all stored route examples. If the highest similarity score meets or
        exceeds the `similarity_threshold`, the corresponding `Route` and
        score are returned. Otherwise, `None` is returned.

        Args:
            query: The input query string to route.

        Returns:
            A tuple of (`Route`, `similarity_score`) if a confident route is found,
            otherwise `None`.
        """
        if not query or not query.strip():
            logger.warning("Attempted to route an empty or whitespace-only query.")
            return None

        try:
            # Embed the query
            query_embedding = (self._embedding_model.embed([query]))[0]
            if not query_embedding:
                logger.error("Embedding model returned empty embedding for query, cannot route.")
                return None

            # Query the vector store for the top match
            results = self._vector_store.query_vectors(
                query_vector=query_embedding,
                top_k=1,
                namespace=self._route_namespace
            )

            if not results:
                logger.debug(f"No semantic routes found in vector store for query: '{query[:100]}...'")
                return None

            # Evaluate the best match
            most_similar_metadata, similarity = results[0]
            if similarity >= self._similarity_threshold:
                route_name = most_similar_metadata.get("route_name")
                if route_name and route_name in self._registered_routes:
                    routed_route = self._registered_routes[route_name]
                    logger.debug(f"Query '{query[:100]}...' confidently routed to '{route_name}' "
                                 f"with similarity {similarity:.4f} (threshold: {self._similarity_threshold:.2f}).")
                    return routed_route, similarity
                else:
                    logger.warning(f"Vector store returned an unregistered route_name '{route_name}'. "
                                   f"Similarity: {similarity:.4f}. Route examples might be stale.")
            else:
                logger.debug(f"Query '{query[:100]}...' did not meet similarity threshold "
                             f"({self._similarity_threshold:.2f}). Best similarity: {similarity:.4f}.")
            return None

        except Exception as e:
            logger.error(f"Error during synchronous query routing for '{query[:100]}...': {e}", exc_info=True)
            return None

    async def aclear_routes(self):
        """
        Asynchronously clears all registered routes and their corresponding
        examples from the `VectorStore` within the configured namespace.
        Also clears the in-memory cache of registered routes.

        Raises:
            Exception: Propagates exceptions from vector store operations.
        """
        try:
            await self._vector_store.adelete_namespace(self._route_namespace)
            self._registered_routes.clear()
            logger.info(f"All routes and their examples cleared from namespace '{self._route_namespace}'.")
        except Exception as e:
            logger.error(f"Failed to clear routes from namespace '{self._route_namespace}': {e}", exc_info=True)
            raise

    def clear_routes(self):
        """
        Synchronously clears all registered routes and their corresponding
        examples from the `VectorStore` within the configured namespace.
        Also clears the in-memory cache of registered routes.

        Raises:
            Exception: Propagates exceptions from vector store operations.
        """
        try:
            self._vector_store.delete_namespace(self._route_namespace)
            self._registered_routes.clear()
            logger.info(f"All routes and their examples cleared from namespace '{self._route_namespace}'.")
        except Exception as e:
            logger.error(f"Failed to clear routes from namespace '{self._route_namespace}': {e}", exc_info=True)
            raise

    @property
    def registered_route_names(self) -> List[str]:
        """
        Returns a list of the names of all currently registered routes.
        """
        return list(self._registered_routes.keys())

    @property
    def registered_routes(self) -> Dict[str, Route]:
        """
        Returns a dictionary mapping route names to their `Route` objects.
        """
        return self._registered_routes.copy() # Return a copy to prevent external modification

    def get_route(self, route_name: str) -> Optional[Route]:
        """
        Retrieves a registered `Route` object by its name.

        Args:
            route_name: The name of the route to retrieve.

        Returns:
            The `Route` object if found, otherwise `None`.
        """
        return self._registered_routes.get(route_name)