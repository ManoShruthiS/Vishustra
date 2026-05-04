import uuid
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cosine

class EmbeddingModel(ABC):
    """
    Abstract base class for embedding models used by Vishustra.
    This interface ensures that the router can work with any compatible embedding service.
    """
    @abstractmethod
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Asynchronously embeds a list of documents into vector representations.

        Args:
            texts: A list of strings, where each string is a document to embed.

        Returns:
            A list of lists of floats, where each inner list is the embedding
            vector for the corresponding document.
        """
        pass

    @abstractmethod
    async def embed_query(self, text: str) -> List[float]:
        """
        Asynchronously embeds a single query string into a vector representation.

        Args:
            text: The query string to embed.

        Returns:
            A list of floats representing the embedding vector for the query.
        """
        pass

    @abstractmethod
    def dimensionality(self) -> int:
        """
        Returns the dimensionality of the embedding vectors produced by this model.
        """
        pass

class VectorStoreClient(ABC):
    """
    Abstract base class for vector store clients.
    This interface allows the SemanticRouter to operate with various vector database
    implementations (e.g., Pinecone, Qdrant, Faiss, ChromaDB) by abstracting
    away their specific APIs.
    """
    @abstractmethod
    async def add_vectors(self, vectors: List[List[float]], metadatas: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """
        Asynchronously adds vectors to the vector store.

        Args:
            vectors: A list of embedding vectors to add.
            metadatas: An optional list of dictionaries, where each dictionary
                       contains metadata associated with the corresponding vector.
                       Metadata is crucial for retrieving route information after a search.

        Returns:
            A list of unique IDs assigned to the added vectors by the vector store.
        """
        pass

    @abstractmethod
    async def similarity_search_with_score(self, query_vector: List[float], k: int = 1) -> List[Tuple[Dict[str, Any], float]]:
        """
        Asynchronously performs a similarity search in the vector store using a query vector.

        Args:
            query_vector: The embedding vector of the query.
            k: The number of top similar results to retrieve.

        Returns:
            A list of tuples, where each tuple contains:
            - A dictionary of metadata for a matched vector.
            - The similarity score between the query vector and the matched vector.
            The list is sorted by similarity score in descending order.
        """
        pass

    @abstractmethod
    async def delete_vectors(self, ids: List[str]):
        """
        Asynchronously deletes vectors from the store by their unique IDs.

        Args:
            ids: A list of vector IDs to delete.
        """
        pass

    @abstractmethod
    async def clear(self):
        """
        Asynchronously clears all vectors and associated data from the vector store.
        """
        pass

class InMemoryEmbeddingModel(EmbeddingModel):
    """
    A lightweight, in-memory dummy embedding model for demonstration and testing purposes.
    Generates random vectors of a specified dimensionality.
    In a production environment, this would be replaced by actual LLM embedding providers
    like OpenAI, Cohere, HuggingFace, etc.
    """
    def __init__(self, dim: int = 768):
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError("Dimensionality must be a positive integer.")
        self._dim = dim

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generates random embeddings for a list of documents."""
        return [self._generate_embedding() for _ in texts]

    async def embed_query(self, text: str) -> List[float]:
        """Generates a random embedding for a single query."""
        return self._generate_embedding()

    def dimensionality(self) -> int:
        """Returns the configured dimensionality."""
        return self._dim

    def _generate_embedding(self) -> List[float]:
        """Helper to generate a random normalized embedding vector."""
        vec = np.random.rand(self._dim)
        return (vec / np.linalg.norm(vec)).tolist() # Normalize to unit vector

class InMemoryVectorStoreClient(VectorStoreClient):
    """
    A simple, in-memory vector store client for demonstration and testing.
    It uses a dictionary to store vectors and performs cosine similarity search.
    This is not suitable for large-scale production use due to lack of indexing
    and persistence.
    """
    def __init__(self):
        self._store: Dict[str, Tuple[List[float], Dict[str, Any]]] = {} # {id: (vector, metadata)}

    async def add_vectors(self, vectors: List[List[float]], metadatas: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """Adds vectors to the in-memory store."""
        ids = []
        _metadatas = metadatas if metadatas is not None else [{} for _ in vectors]
        if len(vectors) != len(_metadatas):
            raise ValueError("Length of vectors and metadatas must be the same.")

        for i, vector in enumerate(vectors):
            new_id = str(uuid.uuid4())
            self._store[new_id] = (vector, _metadatas[i])
            ids.append(new_id)
        return ids

    async def similarity_search_with_score(self, query_vector: List[float], k: int = 1) -> List[Tuple[Dict[str, Any], float]]:
        """
        Performs a cosine similarity search against all stored vectors.
        """
        if not self._store:
            return []

        results = []
        query_np = np.array(query_vector)

        for _id, (vec, meta) in self._store.items():
            vec_np = np.array(vec)
            # Cosine similarity: 1 - cosine_distance. Higher is better.
            similarity = 1 - cosine(query_np, vec_np)
            results.append((meta, similarity))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    async def delete_vectors(self, ids: List[str]):
        """Deletes vectors by their IDs."""
        for _id in ids:
            self._store.pop(_id, None)

    async def clear(self):
        """Clears all vectors from the store."""
        self._store.clear()

class Route:
    """
    Represents a specific routing path within the Vishustra framework.
    Each route is defined by a unique name, a natural language description,
    and a target (which could be an identifier for a chain, tool, agent, or a callable).
    """
    def __init__(self, name: str, description: str, target: Any, metadata: Optional[Dict[str, Any]] = None):
        """
        Initializes a Route instance.

        Args:
            name: A unique, human-readable name for the route (e.g., "sales_inquiry_chain").
            description: A detailed natural language description of what this route handles.
                         This description is used for semantic matching.
            target: The actual destination for this route. This can be any type:
                    - A string identifier (e.g., "sales_chain_id", "calculator_tool").
                    - A callable function or method.
                    - An instance of a chain or agent class.
            metadata: Optional, additional key-value pairs associated with the route.
        """
        if not name:
            raise ValueError("Route name cannot be empty.")
        if not description:
            raise ValueError("Route description cannot be empty.")

        self.id: str = str(uuid.uuid4()) # Internal unique identifier for vector store management
        self.name: str = name
        self.description: str = description
        self.target: Any = target
        self.metadata: Dict[str, Any] = metadata if metadata is not None else {}

    def __repr__(self) -> str:
        return f"Route(name='{self.name}', description='{self.description[:50]}...', target='{self.target}')"

class SemanticRouter:
    """
    The Vishustra SemanticRouter dynamically directs incoming queries or LLM outputs
    to the most semantically relevant route (e.g., a specific tool, processing chain,
    or agent) based on their meaning.

    It leverages an embedding model to convert route descriptions and input queries
    into dense vector representations, and a vector store to efficiently find
    the best matching route.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store_client: VectorStoreClient,
        similarity_threshold: float = 0.75,
        top_k: int = 1,
    ):
        """
        Initializes the SemanticRouter.

        Args:
            embedding_model: An instance of an `EmbeddingModel` to generate vector
                             representations for route descriptions and queries.
            vector_store_client: An instance of a `VectorStoreClient` to store and
                                 retrieve route embeddings efficiently.
            similarity_threshold: The minimum cosine similarity score required for a
                                  route to be considered a valid match. If the highest
                                  matching route's score falls below this threshold,
                                  `route()` will return None. Defaults to 0.75.
            top_k: The number of top similar routes to retrieve from the vector store
                   before applying the similarity threshold. For basic routing, `top_k=1`
                   is usually sufficient. Defaults to 1.
        """
        if not isinstance(embedding_model, EmbeddingModel):
            raise TypeError("embedding_model must be an instance of EmbeddingModel.")
        if not isinstance(vector_store_client, VectorStoreClient):
            raise TypeError("vector_store_client must be an instance of VectorStoreClient.")
        if not 0 <= similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1.")
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        self._embedding_model = embedding_model
        self._vector_store_client = vector_store_client
        self._similarity_threshold = similarity_threshold
        self._top_k = top_k
        self._routes: Dict[str, Route] = {} # Internal cache of Route objects by their unique ID

    async def add_route(self, route: Route):
        """
        Adds a new route to the router.
        The route's description is embedded and stored in the vector store.
        If a route with the same unique `id` already exists, it will be updated
        (its old embedding removed and a new one added). If a route with the
        same `name` but different `id` is added, it's treated as a new route.
        To explicitly update by name, first `remove_route` then `add_route`.

        Args:
            route: The `Route` object to add.
        """
        existing_route_id = None
        for r_id, r in self._routes.items():
            if r.name == route.name:
                existing_route_id = r_id
                break

        if existing_route_id:
            # If a route with this ID or name already exists, remove it first
            await self._vector_store_client.delete_vectors(ids=[existing_route_id])
            del self._routes[existing_route_id]
            print(f"Removed existing route '{route.name}' (ID: {existing_route_id}) before adding new version.")

        embedding = await self._embedding_model.embed_query(route.description)
        
        # Metadata stored in vector store must be serializable.
        # Store route's ID, name, description, and string representation of target.
        vector_metadata = {
            "route_id": route.id,
            "route_name": route.name,
            "route_description": route.description,
            "target_repr": str(route.target), # Store a string representation of target
            **route.metadata # Include any additional metadata
        }
        
        await self._vector_store_client.add_vectors(
            vectors=[embedding],
            metadatas=[vector_metadata]
        )
        self._routes[route.id] = route
        print(f"Route '{route.name}' (ID: {route.id}) added successfully.")

    async def remove_route(self, route_name: str) -> bool:
        """
        Removes a route by its name.
        This operation also removes the associated embedding from the vector store.

        Args:
            route_name: The `name` of the route to remove.

        Returns:
            True if the route was found and successfully removed, False otherwise.
        """
        route_to_remove: Optional[Route] = None
        for _id, route in self._routes.items():
            if route.name == route_name:
                route_to_remove = route
                break

        if route_to_remove:
            await self._vector_store_client.delete_vectors(ids=[route_to_remove.id])
            del self._routes[route_to_remove.id]
            print(f"Route '{route_name}' (ID: {route_to_remove.id}) removed successfully.")
            return True
        print(f"Route '{route_name}' not found. No route removed.")
        return False

    async def route(self, query: str) -> Optional[Tuple[Route, float]]:
        """
        Determines the most semantically relevant route for a given input query.

        Args:
            query: The input query string to route.

        Returns:
            A tuple containing the best matching `Route` object and its similarity score,
            or `None` if no route meets the configured similarity threshold.
        """
        if not self._routes:
            print("No routes defined in the router. Cannot perform routing.")
            return None

        query_embedding = await self._embedding_model.embed_query(query)
        search_results = await self._vector_store_client.similarity_search_with_score(
            query_vector=query_embedding,
            k=self._top_k
        )

        if not search_results:
            print("No similarity search results found from vector store.")
            return None

        # Take the top match as per _top_k (which is usually 1)
        best_match_meta, best_match_score = search_results[0]
        
        if best_match_score >= self._similarity_threshold:
            route_id = best_match_meta.get("route_id")
            if route_id and route_id in self._routes:
                matched_route = self._routes[route_id]
                print(f"Query '{query[:30]}...' matched route '{matched_route.name}' "
                      f"with score: {best_match_score:.4f} (Threshold: {self._similarity_threshold:.2f}).")
                return matched_route, best_match_score
            else:
                # This indicates a potential data inconsistency between _routes cache and vector store
                print(f"Warning: Matched vector found with ID '{route_id}', but no corresponding Route object in cache. "
                      "This might indicate an issue or stale vector store entry.")
                return None
        else:
            print(f"No route found above similarity threshold ({self._similarity_threshold:.2f}). "
                  f"Best match score: {best_match_score:.4f} for route '{best_match_meta.get('route_name', 'N/A')}'.")
            return None

    def get_all_routes(self) -> List[Route]:
        """
        Returns a list of all currently registered `Route` objects in the router.
        """
        return list(self._routes.values())

    async def clear_all_routes(self):
        """
        Clears all registered routes from the router and removes their
        embeddings from the underlying vector store.
        """
        await self._vector_store_client.clear()
        self._routes.clear()
        print("All routes cleared from router and vector store.")