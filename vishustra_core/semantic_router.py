import numpy as np
from typing import List, Dict, Any, Optional, Protocol, Tuple
from dataclasses import dataclass
import asyncio

# --- Interfaces (Protocols) for extensibility ---

class EmbeddingModel(Protocol):
    """
    Protocol for an embedding model.

    Implementations should convert text into dense vector representations.
    Vectors should ideally be normalized for cosine similarity calculations.
    """
    def embed_text(self, text: str) -> np.ndarray:
        """Embeds a single text string into a vector."""
        ...
    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Embeds a list of text strings into a list of vectors."""
        ...

class VectorStore(Protocol):
    """
    Protocol for a vector store.

    Implementations should manage storage and retrieval of vectors, typically
    facilitating similarity search operations.
    """
    def add_vectors(self, ids: List[str], vectors: List[np.ndarray], metadata: List[Dict[str, Any]]):
        """
        Adds vectors to the store.

        Args:
            ids: Unique identifiers for each vector.
            vectors: The actual vector embeddings.
            metadata: Associated metadata for each vector, useful for filtering or retrieval.
        """
        ...
    def search(self, query_vector: np.ndarray, top_k: int = 1) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Performs a similarity search against the stored vectors.

        Args:
            query_vector: The vector to search with. Should typically be normalized.
            top_k: The number of top similar results to return.

        Returns:
            A list of tuples, each containing (id, similarity_score, metadata) for the
            most similar vectors, sorted by similarity in descending order.
        """
        ...
    def get_by_id(self, id: str) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Retrieves a vector and its metadata by its unique ID.

        Args:
            id: The unique identifier of the vector.

        Returns:
            A tuple of (vector, metadata) or None if not found.
        """
        ...

# --- Data Models for Router Components ---

@dataclass(frozen=True)
class RouteDefinition:
    """
    Defines a routable destination within the Vishustra framework.

    Attributes:
        name: A unique, human-readable name for the route (e.g., "CustomerSupportAgent").
        description: A detailed description of what this route handles. This text
                     will be embedded and used for semantic matching.
        target_identifier: The actual identifier used by the framework to invoke
                           this route (e.g., an agent's ID, a chain's name, a tool's function).
    """
    name: str
    description: str
    target_identifier: str

@dataclass
class InferredRoute:
    """
    Represents a potential route inferred by the SemanticRouter for a given query.

    Attributes:
        name: The name of the matched route.
        description: The description of the matched route.
        target_identifier: The identifier to use for invoking the matched route.
        score: The similarity score (e.g., cosine similarity) of the match,
               indicating confidence.
    """
    name: str
    description: str
    target_identifier: str
    score: float

# --- Mock Implementations for Demonstration/Testing ---

class MockEmbeddingModel:
    """
    A simple mock embedding model for testing purposes.

    Generates deterministic, normalized random vectors.
    """
    _DIMENSION = 128

    def embed_text(self, text: str) -> np.ndarray:
        """
        Embeds a single text string by generating a deterministic normalized random vector.
        """
        # Use hash for reproducibility and determinism
        hash_val = hash(text)
        # Ensure seed is within NumPy's 32-bit integer range
        np.random.seed(hash_val % (2**32 - 1))
        vector = np.random.rand(self._DIMENSION).astype(np.float32)
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else np.zeros_like(vector)
    
    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Embeds a list of text strings."""
        return [self.embed_text(text) for text in texts]

class InMemoryVectorStore:
    """
    A simple in-memory vector store implementation for demonstration purposes.

    It stores vectors and their metadata in a dictionary and performs
    cosine similarity search.
    """
    def __init__(self):
        self._store: Dict[str, Tuple[np.ndarray, Dict[str, Any]]] = {}

    def add_vectors(self, ids: List[str], vectors: List[np.ndarray], metadata: List[Dict[str, Any]]):
        """Adds vectors and their metadata to the in-memory store."""
        if not (len(ids) == len(vectors) == len(metadata)):
            raise ValueError("Lengths of ids, vectors, and metadata must match.")
        for i, _id in enumerate(ids):
            self._store[_id] = (vectors[i], metadata[i])

    def search(self, query_vector: np.ndarray, top_k: int = 1) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Performs a cosine similarity search against stored vectors.

        Assumes query_vector and stored vectors are normalized.
        """
        if not self._store:
            return []

        query_vector_normalized = query_vector / np.linalg.norm(query_vector) \
                                  if np.linalg.norm(query_vector) > 0 else np.zeros_like(query_vector)

        results = []
        for _id, (stored_vector, metadata) in self._store.items():
            # For normalized vectors, dot product is cosine similarity
            similarity = np.dot(query_vector_normalized, stored_vector)
            results.append((_id, float(similarity), metadata))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def get_by_id(self, id: str) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Retrieves a vector and its metadata by ID."""
        return self._store.get(id)

# --- Main Semantic Router Implementation ---

class SemanticRouter:
    """
    Routes an incoming query to the most semantically relevant target within the framework.

    This router leverages an `EmbeddingModel` to vectorize queries and `RouteDefinition` descriptions,
    and a `VectorStore` to perform efficient similarity searches, identifying the best-fit route.
    It's designed for high modularity and performance in LLM orchestration.
    """
    
    def __init__(self, 
                 embedding_model: EmbeddingModel, 
                 vector_store: VectorStore, 
                 routes: Optional[List[RouteDefinition]] = None, 
                 similarity_threshold: float = 0.75):
        """
        Initializes the SemanticRouter.

        Args:
            embedding_model: An instance adhering to the `EmbeddingModel` protocol.
            vector_store: An instance adhering to the `VectorStore` protocol.
            routes: An optional list of initial `RouteDefinition` objects to register upon initialization.
            similarity_threshold: The minimum cosine similarity score required for a route
                                  to be considered a confident match. Matches below this
                                  threshold will result in `None` being returned by `route_query`.
                                  Expected to be between 0.0 and 1.0.
        """
        if not (0.0 <= similarity_threshold <= 1.0):
            raise ValueError("similarity_threshold must be between 0.0 and 1.0.")

        self._embedding_model = embedding_model
        self._vector_store = vector_store
        self._similarity_threshold = similarity_threshold
        # Stores the actual RouteDefinition objects, keyed by name for quick lookup after search
        self._registered_routes: Dict[str, RouteDefinition] = {} 

        if routes:
            self.add_routes(routes)

    def add_routes(self, routes: List[RouteDefinition]):
        """
        Registers multiple new routes with the router.

        Embeds each route's description and adds it to the underlying vector store.
        If a route with the same name already exists, it will be skipped.

        Args:
            routes: A list of `RouteDefinition` objects to add.
        """
        new_route_ids = []
        new_route_descriptions = []
        new_route_metadatas = []

        for route in routes:
            if route.name in self._registered_routes:
                # In a production system, one might log a warning or replace the existing route.
                # For this example, we simply skip.
                continue 
            
            self._registered_routes[route.name] = route
            new_route_ids.append(route.name) # Use route name as ID in vector store
            new_route_descriptions.append(route.description)
            new_route_metadatas.append({
                "name": route.name, 
                "description": route.description, 
                "target_identifier": route.target_identifier
            })
        
        if new_route_descriptions:
            route_embeddings = self._embedding_model.embed_texts(new_route_descriptions)
            self._vector_store.add_vectors(new_route_ids, route_embeddings, new_route_metadatas)
            
    def add_route(self, route: RouteDefinition):
        """
        Registers a single new route with the router.

        Args:
            route: The `RouteDefinition` object to add.
        """
        self.add_routes([route])

    async def route_query(self, query: str) -> Optional[InferredRoute]:
        """
        Asynchronously routes an incoming query to the most appropriate target based on semantic similarity.

        Args:
            query: The incoming query string from the user or another system component.

        Returns:
            An `InferredRoute` object containing the best matching route's details and
            its similarity score, or `None` if no route meets the configured `similarity_threshold`.
        """
        if not query:
            return None

        query_embedding = self._embedding_model.embed_text(query)
        
        # Perform a similarity search for the top route
        search_results = self._vector_store.search(query_embedding, top_k=1)
        
        if not search_results:
            return None # No routes registered or vector store is empty

        best_match_id, best_score, _ = search_results[0]
        
        if best_score >= self._similarity_threshold:
            # Retrieve the full RouteDefinition object for the best match using the stored ID
            best_route_def = self._registered_routes.get(best_match_id)
            if best_route_def:
                return InferredRoute(
                    name=best_route_def.name,
                    description=best_route_def.description,
                    target_identifier=best_route_def.target_identifier,
                    score=best_score
                )
            else:
                # This scenario indicates an inconsistency (route ID in vector store but not in _registered_routes)
                # Should ideally not happen if add_routes is robust.
                return None
        else:
            return None # No confident match found above the similarity threshold

# Example Usage (optional, for demonstration of how it would be used)
async def _example_usage():
    print("--- SemanticRouter Example Usage ---")

    # 1. Initialize components
    embedding_model = MockEmbeddingModel()
    vector_store = InMemoryVectorStore()
    router = SemanticRouter(
        embedding_model=embedding_model,
        vector_store=vector_store,
        similarity_threshold=0.78 # Adjust threshold for stricter or looser matching
    )

    # 2. Define and add routes
    customer_support_route = RouteDefinition(
        name="CustomerSupport",
        description="Handles inquiries related to product issues, refunds, order status, or technical support.",
        target_identifier="CustomerSupportAgent"
    )
    sales_inquiry_route = RouteDefinition(
        name="SalesInquiry",
        description="Routes questions about new product features, pricing, purchasing, or partnership opportunities.",
        target_identifier="SalesAgent"
    )
    general_chat_route = RouteDefinition(
        name="GeneralChat",
        description="For casual conversation, greetings, or off-topic discussions not covered by other agents.",
        target_identifier="GeneralChatbot"
    )
    router.add_routes([customer_support_route, sales_inquiry_route, general_chat_route])
    print(f"Registered {len(router._registered_routes)} routes.")

    # 3. Test routing with various queries
    queries = [
        "My order #12345 hasn't arrived yet. Can you help?",
        "Tell me more about the new AI features roadmap.",
        "Hello, how are you today?",
        "I need help setting up my account.",
        "What's the price for enterprise license?",
        "When is the next full moon?", # Should not confidently match
        "How do I request a refund?",
    ]

    for query in queries:
        print(f"\nRouting query: '{query}'")
        inferred_route = await router.route_query(query)
        if inferred_route:
            print(f"  -> Matched Route: {inferred_route.name}")
            print(f"     Target: {inferred_route.target_identifier}")
            print(f"     Score: {inferred_route.score:.4f}")
        else:
            print("  -> No confident route found.")

if __name__ == "__main__":
    asyncio.run(_example_usage())