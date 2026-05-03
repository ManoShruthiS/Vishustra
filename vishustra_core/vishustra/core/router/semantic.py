"""
Vishustra's SemanticRouter module for intelligent request dispatch based on semantic similarity.

This module provides a robust, asynchronous semantic routing mechanism for 'Vishustra',
allowing incoming LLM queries to be directed to different handling components
(e.g., specific agents, tool chains, or prompt templates) based on their
semantic meaning rather than rigid keyword matching.

The `SemanticRouter` leverages an embedding model to convert both the input query
and predefined route descriptions into high-dimensional vectors. It then calculates
the cosine similarity between the query vector and all route vectors to determine
the most relevant route. This enables more flexible and context-aware routing decisions.

Key features:
- Asynchronous operations for non-blocking I/O with embedding models.
- Support for precomputed route embeddings for performance optimization.
- Configurable similarity threshold for routing decisions.
- Clear separation of concerns between router logic and embedding model implementation.
- Highly modular and extensible for various routing strategies.
"""

import asyncio
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Union

# --- Placeholder for Vishustra's core components ---
# In a real framework, these would be defined in their respective files.

class VishustraRouterException(Exception):
    """Base exception for errors within the Vishustra router module."""
    pass

class BaseEmbeddingModel(ABC):
    """
    Abstract Base Class for Vishustra's embedding models.

    All embedding models used by components like the SemanticRouter must
    implement this interface to ensure compatibility.
    """

    @abstractmethod
    async def aembed(self, texts: List[str]) -> List[List[float]]:
        """
        Asynchronously embeds a list of texts.

        Args:
            texts: A list of strings to embed.

        Returns:
            A list of embeddings, where each embedding is a list of floats.
        """
        pass

    async def aembed_query(self, text: str) -> List[float]:
        """
        Asynchronously embeds a single query text.

        This is a convenience method that calls `aembed` with a single item list
        and returns the first (and only) embedding.

        Args:
            text: The single string to embed.

        Returns:
            A single embedding as a list of floats.
        """
        results = await self.aembed([text])
        if not results:
            raise VishustraRouterException("Embedding model returned empty results for single query.")
        return results[0]

# --- Core SemanticRouter Implementation ---

@dataclass(frozen=True)
class RouterRoute:
    """
    Represents a single routing destination for the SemanticRouter.

    Each route defines a name, a descriptive explanation, and either example phrases
    or a precomputed embedding vector that semantically represents this route's intent.

    Attributes:
        name: A unique identifier for the route (e.g., "customer_service", "data_retrieval").
        description: A human-readable description of what this route handles.
        example_phrases: Optional. A list of example phrases that are characteristic
                         of queries meant for this route. If `precomputed_embedding`
                         is not provided, the first phrase will be used to generate
                         the route's embedding during initialization.
        precomputed_embedding: Optional. A precomputed embedding vector (numpy array)
                               for this route. Providing this can speed up
                               initialization for static routes.
    """
    name: str
    description: str
    example_phrases: Optional[List[str]] = None
    precomputed_embedding: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.precomputed_embedding is not None and not isinstance(self.precomputed_embedding, np.ndarray):
            raise TypeError("precomputed_embedding must be a numpy array.")
        if self.example_phrases is None and self.precomputed_embedding is None:
            raise ValueError("RouterRoute must have either 'example_phrases' or 'precomputed_embedding'.")
        if self.example_phrases and not all(isinstance(p, str) for p in self.example_phrases):
            raise TypeError("All items in 'example_phrases' must be strings.")


class SemanticRouter:
    """
    Intelligent router that dispatches queries based on semantic similarity.

    The router uses an embedding model to compare incoming query embeddings with
    predefined route embeddings, selecting the most semantically similar route.

    Requires initialization via `ainitialize()` after instantiation to compute
    or load route embeddings.
    """

    def __init__(self, embedding_model: BaseEmbeddingModel, routes: List[RouterRoute]):
        """
        Initializes the SemanticRouter.

        Args:
            embedding_model: An instance of `BaseEmbeddingModel` to generate embeddings.
            routes: A list of `RouterRoute` objects defining the available routes.
        """
        if not isinstance(embedding_model, BaseEmbeddingModel):
            raise TypeError("embedding_model must be an instance of BaseEmbeddingModel.")
        if not all(isinstance(r, RouterRoute) for r in routes):
            raise TypeError("All items in 'routes' must be instances of RouterRoute.")

        self._embedding_model = embedding_model
        self._routes = routes
        self._route_embeddings: Optional[np.ndarray] = None  # Stores stacked embeddings of routes
        self._route_names: List[str] = [route.name for route in routes]

    async def ainitialize(self):
        """
        Asynchronously initializes the router by computing embeddings for routes
        that do not have `precomputed_embedding` set.

        This method must be called after instantiation and before `aroute()`
        to prepare the router for use.
        """
        if not self._routes:
            raise VishustraRouterException("Cannot initialize SemanticRouter with no routes.")

        embeddings_to_compute_texts: List[str] = []
        embeddings_to_compute_indices: List[int] = []
        # Temporary list to hold embeddings as they are prepared/computed
        prepared_embeddings_list: List[Optional[np.ndarray]] = [None] * len(self._routes)

        for i, route in enumerate(self._routes):
            if route.precomputed_embedding is not None:
                # Ensure precomputed embeddings are numpy arrays of float32
                prepared_embeddings_list[i] = np.array(route.precomputed_embedding, dtype=np.float32)
            elif route.example_phrases:
                # Use the first example phrase as the primary embedding for the route's intent
                embeddings_to_compute_texts.append(route.example_phrases[0])
                embeddings_to_compute_indices.append(i)
            else:
                # This case should be caught by RouterRoute's __post_init__ but added for robustness
                raise VishustraRouterException(
                    f"Route '{route.name}' has neither precomputed_embedding nor example_phrases."
                )

        if embeddings_to_compute_texts:
            # Compute embeddings for all texts that need them in one batch call
            computed_embeddings_raw = await self._embedding_model.aembed(embeddings_to_compute_texts)
            computed_embeddings_np = [np.array(e, dtype=np.float32) for e in computed_embeddings_raw]

            # Place the newly computed embeddings into their correct positions
            for j, original_index in enumerate(embeddings_to_compute_indices):
                prepared_embeddings_list[original_index] = computed_embeddings_np[j]

        # Final check to ensure all routes now have an embedding
        if any(e is None for e in prepared_embeddings_list):
            raise VishustraRouterException("Failed to compute or load embeddings for all routes during initialization.")

        # Stack all route embeddings into a single 2D NumPy array for efficient matrix operations
        self._route_embeddings = np.stack([e for e in prepared_embeddings_list if e is not None])
        if self._route_embeddings.ndim != 2:
            raise VishustraRouterException(
                f"Route embeddings stack resulted in incorrect dimensions: {self._route_embeddings.shape}"
            )
        # Normalize route embeddings for cosine similarity calculation
        self._route_embeddings = self._route_embeddings / np.linalg.norm(self._route_embeddings, axis=1, keepdims=True)

    async def aroute(self, query: str, similarity_threshold: float = 0.7) -> Optional[str]:
        """
        Asynchronously routes an incoming query to the most semantically similar route.

        The router will return the name of the route whose embedding has the highest
        cosine similarity with the query's embedding, provided that similarity
        exceeds the specified `similarity_threshold`.

        Args:
            query: The input query string to route.
            similarity_threshold: The minimum cosine similarity required for a route
                                  to be considered a match. If no route meets or
                                  exceeds this threshold, `None` is returned.
                                  Defaults to 0.7.

        Returns:
            The `name` of the best-matching route, or `None` if no route
            exceeds the `similarity_threshold`.
        """
        if self._route_embeddings is None:
            raise VishustraRouterException(
                "SemanticRouter has not been initialized. Call `ainitialize()` first."
            )
        if not isinstance(query, str):
            raise TypeError("Query must be a string.")
        if not isinstance(similarity_threshold, (int, float)) or not (0.0 <= similarity_threshold <= 1.0):
            raise ValueError("Similarity threshold must be a float between 0.0 and 1.0.")

        # Embed the incoming query
        query_embedding_list = await self._embedding_model.aembed_query(query)
        query_embedding = np.array(query_embedding_list, dtype=np.float32)

        # Ensure query embedding is a 2D array (1, embedding_dim) for matrix multiplication
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        elif query_embedding.ndim != 2 or query_embedding.shape[0] != 1:
            raise VishustraRouterException(
                f"Query embedding has incorrect dimensions: {query_embedding.shape}. Expected (1, N)."
            )

        # Normalize the query embedding for cosine similarity
        query_embedding_norm = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

        # Calculate cosine similarity: dot product of normalized vectors
        # self._route_embeddings are already normalized during ainitialize()
        similarities = np.dot(query_embedding_norm, self._route_embeddings.T).flatten()

        if not similarities.size:
            return None # No routes to compare against

        max_similarity_index = np.argmax(similarities)
        max_similarity = similarities[max_similarity_index]

        if max_similarity >= similarity_threshold:
            return self._route_names[max_similarity_index]
        return None

# --- Example Usage (for documentation/testing purposes) ---

# In a real setup, these would be in separate files and imported.
# class MockEmbeddingModel(BaseEmbeddingModel):
#     """A simple mock embedding model for demonstration."""
#     async def aembed(self, texts: List[str]) -> List[List[float]]:
#         # Simulate different embeddings for different intents
#         embeddings = []
#         for text in texts:
#             if "order" in text.lower() or "purchase" in text.lower():
#                 embeddings.append([0.9, 0.1, 0.1, 0.1])
#             elif "support" in text.lower() or "help" in text.lower():
#                 embeddings.append([0.1, 0.9, 0.1, 0.1])
#             elif "pricing" in text.lower() or "cost" in text.lower():
#                 embeddings.append([0.1, 0.1, 0.9, 0.1])
#             else:
#                 embeddings.append([0.5, 0.5, 0.5, 0.5])
#         # Normalize for cosine similarity simulation
#         return [[x / sum(v for v in emb) for x in emb] for emb in embeddings] if embeddings else []

# async def main():
#     # 1. Instantiate the embedding model
#     embedding_model = MockEmbeddingModel()

#     # 2. Define your routes
#     routes = [
#         RouterRoute(
#             name="customer_service",
#             description="Handles general customer support inquiries, complaints, or assistance requests.",
#             example_phrases=["I need help with my account", "My product is broken", "I have a complaint"]
#         ),
#         RouterRoute(
#             name="order_management",
#             description="Manages inquiries related to new orders, existing orders, tracking, or cancellations.",
#             example_phrases=["Where is my order?", "I want to change my order", "How do I place a new purchase?"]
#         ),
#         RouterRoute(
#             name="billing_inquiries",
#             description="Deals with questions about invoices, payments, refunds, or subscription costs.",
#             example_phrases=["What is my bill?", "I have a question about a charge", "Can I get a refund?"]
#         )
#     ]

#     # 3. Instantiate the router
#     router = SemanticRouter(embedding_model=embedding_model, routes=routes)

#     # 4. Initialize the router (computes/loads route embeddings)
#     print("Initializing SemanticRouter...")
#     await router.ainitialize()
#     print("SemanticRouter initialized.")

#     # 5. Route some queries
#     queries = [
#         "I want to track my recent purchase.",
#         "How much does your premium plan cost?",
#         "My login is not working, can you help?",
#         "What is the status of my invoice?",
#         "Tell me a joke." # Unrelated query
#     ]

#     print("\nRouting queries:")
#     for query in queries:
#         routed_to = await router.aroute(query, similarity_threshold=0.7)
#         print(f"Query: '{query}' -> Routed to: {routed_to if routed_to else 'No match'}")

# if __name__ == "__main__":
#     # This block is commented out to ensure the output is pure code as requested,
#     # but it demonstrates how the component would be used.
#     # asyncio.run(main())
#     pass