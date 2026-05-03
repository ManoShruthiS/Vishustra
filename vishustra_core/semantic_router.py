import abc
import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Generic

from pydantic import BaseModel, Field, PrivateAttr

logger = logging.getLogger(__name__)

# --- Abstract Base Classes for Pluggable Components ---

Vector = List[float]
T = TypeVar("T")  # Generic type for metadata in VectorStore

class EmbeddingModel(abc.ABC):
    """
    Abstract base class for embedding models.
    Provides a standardized interface for text embedding.
    """

    @abc.abstractmethod
    async def embed_query(self, text: str) -> Vector:
        """
        Embeds a single text query into a vector representation asynchronously.

        Args:
            text: The input text to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def embed_documents(self, texts: List[str]) -> List[Vector]:
        """
        Embeds a list of documents into their vector representations asynchronously.

        Args:
            texts: A list of input texts (documents) to embed.

        Returns:
            A list of lists of floats, where each inner list is an embedding vector
            for the corresponding document.
        """
        raise NotImplementedError

class VectorStore(abc.ABC, Generic[T]):
    """
    Abstract base class for vector store implementations.
    Manages storage and retrieval of vectors with associated metadata.
    """

    @abc.abstractmethod
    async def add_vectors(
        self,
        ids: List[str],
        vectors: List[Vector],
        metadatas: Optional[List[T]] = None,
    ) -> None:
        """
        Adds vectors along with their unique identifiers and optional metadata
        to the vector store asynchronously.

        Args:
            ids: A list of unique identifiers for each vector.
            vectors: A list of embedding vectors.
            metadatas: An optional list of metadata objects, corresponding to each vector.
                       The type T is defined by the specific vector store implementation.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def similarity_search(
        self, query_vector: Vector, k: int = 1
    ) -> List[Tuple[str, float, T]]:
        """
        Performs a similarity search in the vector store using a query vector.

        Args:
            query_vector: The embedding vector of the query.
            k: The number of top similar results to return.

        Returns:
            A list of tuples, where each tuple contains:
            - The identifier (str) of the similar vector.
            - The similarity score (float).
            - The associated metadata (T) of the similar vector.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def delete_vectors(self, ids: List[str]) -> None:
        """
        Deletes vectors from the store by their identifiers asynchronously.

        Args:
            ids: A list of identifiers of vectors to delete.
        """
        raise NotImplementedError

# --- Core Vishustra Models ---

class RoutingTarget(BaseModel):
    """
    Represents a specific target or chain that the SemanticRouter can route to.
    Each target has a unique ID, a descriptive name, a detailed description
    (for semantic matching), and arbitrary metadata for downstream processing.
    """
    id: str = Field(..., description="Unique identifier for this routing target.")
    name: str = Field(..., description="Human-readable name of the routing target.")
    description: str = Field(
        ...,
        description="Detailed natural language description of what this target does "
                    "or what kind of queries it handles. Used for semantic matching."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary key-value pairs associated with this target, "
                    "e.g., chain ID, specific LLM configuration, required tools, etc."
    )

    class Config:
        frozen = True # Targets should be immutable after creation for consistent hashing/storage

class SemanticRouterConfig(BaseModel):
    """
    Configuration for the SemanticRouter.
    """
    similarity_threshold: float = Field(
        0.75,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity score required for a successful route. "
                    "Scores below this threshold will result in no route found."
    )
    top_k_candidates: int = Field(
        3,
        ge=1,
        description="Number of top similar routing targets to retrieve from the vector store "
                    "before applying the similarity threshold. "
                    "Higher values increase robustness but may slightly increase latency."
    )
    max_query_retries: int = Field(
        0,
        ge=0,
        description="Maximum number of times to retry an embedding or vector store "
                    "operation in case of transient failures."
    )

# --- Semantic Router Implementation ---

class SemanticRouter:
    """
    The Vishustra SemanticRouter dynamically directs incoming queries to the most
    semantically relevant processing chain or module based on predefined routing targets.

    It leverages an EmbeddingModel to convert queries and target descriptions into
    vector representations, and a VectorStore to perform efficient similarity searches.
    """

    _target_cache: Dict[str, RoutingTarget] = PrivateAttr(default_factory=dict) # In-memory cache for targets

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore[RoutingTarget],
        config: Optional[SemanticRouterConfig] = None,
    ):
        """
        Initializes the SemanticRouter.

        Args:
            embedding_model: An instance of an EmbeddingModel to generate vector embeddings.
            vector_store: An instance of a VectorStore to store and query routing target embeddings.
                          The VectorStore should be typed to handle `RoutingTarget` as its metadata.
            config: Optional configuration for the router. If None, default settings are used.
        """
        if not isinstance(embedding_model, EmbeddingModel):
            raise TypeError("embedding_model must be an instance of EmbeddingModel.")
        if not isinstance(vector_store, VectorStore):
            raise TypeError("vector_store must be an instance of VectorStore.")

        self._embedding_model = embedding_model
        self._vector_store = vector_store
        self._config = config or SemanticRouterConfig()
        logger.info(
            f"SemanticRouter initialized with threshold={self._config.similarity_threshold}, "
            f"top_k={self._config.top_k_candidates}."
        )

    async def _retry_operation(self, func, *args, **kwargs):
        """Internal helper for retrying transient operations."""
        for attempt in range(self._config.max_query_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.warning(
                    f"Operation {func.__name__} failed on attempt {attempt+1}/{self._config.max_query_retries+1}: {e}"
                )
                if attempt < self._config.max_query_retries:
                    await asyncio.sleep(2**attempt)  # Exponential backoff
                else:
                    raise

    async def register_target(self, target: RoutingTarget) -> None:
        """
        Registers a new routing target with the router.
        This involves embedding its description and storing it in the vector store.

        Args:
            target: The RoutingTarget object to register.

        Raises:
            ValueError: If a target with the same ID is already registered.
            Exception: For underlying embedding or vector store failures.
        """
        if target.id in self._target_cache:
            raise ValueError(f"RoutingTarget with ID '{target.id}' is already registered.")

        logger.info(f"Registering new routing target: {target.name} (ID: {target.id})")
        try:
            target_vector = await self._retry_operation(
                self._embedding_model.embed_query, target.description
            )
            await self._retry_operation(
                self._vector_store.add_vectors,
                ids=[target.id],
                vectors=[target_vector],
                metadatas=[target], # Store the full Pydantic model as metadata
            )
            self._target_cache[target.id] = target # Cache the target object
            logger.debug(f"Successfully registered target: {target.name}")
        except Exception as e:
            logger.error(f"Failed to register target '{target.id}': {e}", exc_info=True)
            raise

    async def unregister_target(self, target_id: str) -> None:
        """
        Unregisters a routing target by its ID.
        This removes its embedding from the vector store and from the internal cache.

        Args:
            target_id: The unique ID of the target to unregister.

        Raises:
            KeyError: If no target with the given ID is found.
            Exception: For underlying vector store failures.
        """
        if target_id not in self._target_cache:
            raise KeyError(f"No routing target found with ID '{target_id}'.")

        logger.info(f"Unregistering routing target with ID: {target_id}")
        try:
            await self._retry_operation(self._vector_store.delete_vectors, ids=[target_id])
            del self._target_cache[target_id]
            logger.debug(f"Successfully unregistered target: {target_id}")
        except Exception as e:
            logger.error(f"Failed to unregister target '{target_id}': {e}", exc_info=True)
            raise

    async def route_query(self, query: str) -> Optional[RoutingTarget]:
        """
        Routes an incoming query to the most semantically similar registered target.

        The router embeds the query, performs a similarity search, and returns
        the best match that meets the configured similarity threshold.

        Args:
            query: The user query or intent description to route.

        Returns:
            The most relevant RoutingTarget if a match above the threshold is found,
            otherwise None.

        Raises:
            Exception: For underlying embedding or vector store failures.
        """
        if not self._target_cache:
            logger.warning("No routing targets registered. Cannot route query.")
            return None

        logger.debug(f"Attempting to route query: '{query[:100]}...'")
        try:
            query_vector = await self._retry_operation(
                self._embedding_model.embed_query, query
            )
            search_results = await self._retry_operation(
                self._vector_store.similarity_search,
                query_vector,
                k=self._config.top_k_candidates,
            )

            if not search_results:
                logger.debug("No similarity search results found.")
                return None

            best_match: Optional[RoutingTarget] = None
            highest_score: float = 0.0

            for target_id, score, metadata in search_results:
                if score >= self._config.similarity_threshold:
                    if score > highest_score:
                        # The metadata retrieved from the vector store *is* our RoutingTarget
                        # since we stored the full Pydantic model.
                        # We must ensure the vector store implementation correctly handles
                        # serialization/deserialization of this object.
                        # For robust retrieval, we can also look up by ID in _target_cache
                        # if the vector store only stores a dict and not the Pydantic instance.
                        retrieved_target = metadata
                        if isinstance(retrieved_target, dict):
                             # If vector store returned a dict, rehydrate it.
                             # This assumes the dict keys match Pydantic model fields.
                             retrieved_target = self._target_cache.get(target_id)
                             if retrieved_target is None:
                                 logger.warning(
                                    f"Target ID {target_id} found in vector store but not in cache. "
                                    f"Skipping this match. Potential cache desync."
                                )
                                 continue
                        elif not isinstance(retrieved_target, RoutingTarget):
                             logger.error(f"Vector store returned unexpected metadata type: {type(retrieved_target)}. "
                                          f"Expected RoutingTarget or dict. Skipping this match.")
                             continue

                        best_match = retrieved_target
                        highest_score = score
                        logger.debug(
                            f"Candidate match: {best_match.name} (ID: {best_match.id}) "
                            f"with score {score:.4f} (threshold: {self._config.similarity_threshold:.4f})"
                        )
                else:
                    logger.debug(
                        f"Candidate below threshold: {target_id} with score {score:.4f} "
                        f"(threshold: {self._config.similarity_threshold:.4f})"
                    )

            if best_match:
                logger.info(
                    f"Query routed to target: {best_match.name} (ID: {best_match.id}) "
                    f"with confidence {highest_score:.4f}"
                )
                return best_match
            else:
                logger.info(
                    f"No routing target found above similarity threshold "
                    f"({self._config.similarity_threshold:.4f}) for query: '{query[:50]}...'"
                )
                return None

        except Exception as e:
            logger.error(f"Error during query routing for '{query[:50]}...': {e}", exc_info=True)
            raise

    async def get_registered_targets(self) -> List[RoutingTarget]:
        """
        Returns a list of all currently registered routing targets.

        Returns:
            A list of RoutingTarget objects.
        """
        return list(self._target_cache.values())

    async def get_target_by_id(self, target_id: str) -> Optional[RoutingTarget]:
        """
        Retrieves a registered target by its unique ID.

        Args:
            target_id: The unique ID of the target to retrieve.

        Returns:
            The RoutingTarget object if found, otherwise None.
        """
        return self._target_cache.get(target_id)

    async def refresh_target_embeddings(self) -> None:
        """
        Re-embeds and updates all registered target descriptions in the vector store.
        Useful if the underlying embedding model changes or if vector store data
        needs to be refreshed. This is an expensive operation.
        """
        if not self._target_cache:
            logger.info("No targets to refresh embeddings for.")
            return

        logger.info("Starting refresh of all routing target embeddings...")
        updated_ids: List[str] = []
        updated_vectors: List[Vector] = []
        updated_metadatas: List[RoutingTarget] = []

        for target_id, target in self._target_cache.items():
            try:
                new_vector = await self._retry_operation(
                    self._embedding_model.embed_query, target.description
                )
                updated_ids.append(target_id)
                updated_vectors.append(new_vector)
                updated_metadatas.append(target)
            except Exception as e:
                logger.error(
                    f"Failed to re-embed target '{target_id}' ({target.name}): {e}. "
                    "This target might be stale in the vector store.", exc_info=True
                )
        
        if updated_ids:
            try:
                # Delete existing vectors and add new ones.
                # A robust vector store might have an 'update' method.
                # For simplicity here, we assume delete+add works.
                await self._retry_operation(self._vector_store.delete_vectors, ids=updated_ids)
                await self._retry_operation(
                    self._vector_store.add_vectors,
                    ids=updated_ids,
                    vectors=updated_vectors,
                    metadatas=updated_metadatas,
                )
                logger.info(f"Successfully refreshed embeddings for {len(updated_ids)} targets.")
            except Exception as e:
                logger.error(f"Failed to update embeddings in vector store during refresh: {e}", exc_info=True)
                raise
        else:
            logger.info("No targets were successfully re-embedded for refresh.")

# --- Vishustra Internal Utility/Example Implementations (Not for external use, for completeness) ---

class InMemoryEmbeddingModel(EmbeddingModel):
    """
    A simplistic in-memory embedding model for testing purposes.
    Generates dummy, but distinct, embeddings.
    """
    async def embed_query(self, text: str) -> Vector:
        # Simple hash-based dummy embedding for uniqueness
        seed = sum(ord(c) for c in text)
        return [float(seed % 1000) / 1000.0, float((seed // 1000) % 1000) / 1000.0, float((seed // 1000000) % 1000) / 1000.0]

    async def embed_documents(self, texts: List[str]) -> List[Vector]:
        return [await self.embed_query(text) for text in texts]

class InMemoryVectorStore(VectorStore[RoutingTarget]):
    """
    A simplistic in-memory vector store for testing purposes.
    Calculates cosine similarity.
    """
    _vectors: Dict[str, Tuple[Vector, RoutingTarget]] = PrivateAttr(default_factory=dict)

    async def add_vectors(
        self,
        ids: List[str],
        vectors: List[Vector],
        metadatas: Optional[List[RoutingTarget]] = None,
    ) -> None:
        if metadatas is None:
            raise ValueError("Metadata (RoutingTarget) must be provided for InMemoryVectorStore.")
        if len(ids) != len(vectors) or len(ids) != len(metadatas):
            raise ValueError("Lengths of ids, vectors, and metadatas must match.")
        
        for i, _id in enumerate(ids):
            self._vectors[_id] = (vectors[i], metadatas[i])
        logger.debug(f"Added {len(ids)} vectors to in-memory store.")

    async def similarity_search(
        self, query_vector: Vector, k: int = 1
    ) -> List[Tuple[str, float, RoutingTarget]]:
        results: List[Tuple[str, float, RoutingTarget]] = []
        
        if not self._vectors:
            return []

        # Simple cosine similarity calculation
        def cosine_similarity(vec1: Vector, vec2: Vector) -> float:
            dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
            magnitude1 = sum(v**2 for v in vec1)**0.5
            magnitude2 = sum(v**2 for v in vec2)**0.5
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            return dot_product / (magnitude1 * magnitude2)

        for _id, (stored_vector, metadata) in self._vectors.items():
            score = cosine_similarity(query_vector, stored_vector)
            results.append((_id, score, metadata))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    async def delete_vectors(self, ids: List[str]) -> None:
        for _id in ids:
            self._vectors.pop(_id, None)
        logger.debug(f"Deleted {len(ids)} vectors from in-memory store.")

# Example Usage (typically in a separate test or main file, kept here for context)
async def _example_usage():
    logger.setLevel(logging.DEBUG) # Enable debug logging for example
    logging.basicConfig()

    # 1. Instantiate components
    embed_model = InMemoryEmbeddingModel()
    vector_store = InMemoryVectorStore()
    router_config = SemanticRouterConfig(similarity_threshold=0.8, top_k_candidates=2)
    router = SemanticRouter(embedding_model=embed_model, vector_store=vector_store, config=router_config)

    # 2. Define routing targets
    target_qa = RoutingTarget(
        id="TGT_QA_1",
        name="Knowledge Base QA",
        description="Handles questions about Vishustra framework features, documentation, and general technical queries. Uses a RAG chain over internal docs.",
        metadata={"chain_id": "vishustra_rag_chain", "llm_model": "gpt-4-turbo"}
    )
    target_code_gen = RoutingTarget(
        id="TGT_CODE_GEN_2",
        name="Code Generation Assistant",
        description="Generates Python code snippets, refactors existing code, or provides examples for Vishustra components. Focuses on coding tasks.",
        metadata={"chain_id": "code_writer_agent", "llm_model": "claude-3-opus", "tool_access": ["python_interpreter"]}
    )
    target_support = RoutingTarget(
        id="TGT_SUPPORT_3",
        name="Customer Support Agent",
        description="Assists with customer support requests, billing inquiries, account management, and general helpdesk questions. Escalates to human support if needed.",
        metadata={"chain_id": "support_handoff_flow", "llm_model": "gpt-3.5-turbo"}
    )
    target_smalltalk = RoutingTarget(
        id="TGT_SMALLTALK_4",
        name="Small Talk & Greetings",
        description="Engages in casual conversation, greets users, and responds to polite social interactions. Aims for friendly, non-task-specific chat.",
        metadata={"chain_id": "conversational_ai"}
    )

    # 3. Register targets
    await router.register_target(target_qa)
    await router.register_target(target_code_gen)
    await router.register_target(target_support)
    await router.register_target(target_smalltalk)

    print("\n--- Routing Queries ---")

    # 4. Test routing with various queries
    queries = [
        "How do I use the memory buffer in Vishustra?", # Should go to QA
        "Write a Python function to create a new agent tool wrapper.", # Should go to Code Gen
        "I need help with my account, can you check my subscription details?", # Should go to Support
        "Hello, how are you today?", # Should go to Small Talk
        "What is the best way to optimize LLM calls?", # Should go to QA
        "Show me an example of dynamic tool registration.", # Should go to Code Gen
        "My bill is incorrect, who can I talk to?", # Should go to Support
        "Tell me a joke.", # Should go to Small Talk
        "What is the capital of France?", # Should go to QA (general knowledge)
        "I want to send an email." # Might not match well, expect None or a fallback
    ]

    for i, q in enumerate(queries):
        print(f"\nQuery {i+1}: '{q}'")
        routed_target = await router.route_query(q)
        if routed_target:
            print(f"  -> Routed to: {routed_target.name} (ID: {routed_target.id})")
            print(f"     Metadata: {routed_target.metadata}")
        else:
            print("  -> No suitable route found.")
    
    print("\n--- Unregistering Target ---")
    await router.unregister_target("TGT_SMALLTALK_4")
    print("\nQuery after unregistering Small Talk:")
    routed_target = await router.route_query("How's the weather?")
    if routed_target:
        print(f"  -> Routed to: {routed_target.name} (ID: {routed_target.id})")
    else:
        print("  -> No suitable route found.") # Should be None now

    print("\n--- Refreshing Embeddings ---")
    # Simulate a scenario where descriptions might change or model updates
    target_qa_updated = RoutingTarget(
        id="TGT_QA_1",
        name="Knowledge Base QA v2",
        description="Handles complex questions about Vishustra architecture, scalability, security, and advanced usage patterns. Integrates with multiple internal knowledge sources.",
        metadata={"chain_id": "vishustra_rag_chain_v2", "llm_model": "gpt-4-1106-preview"}
    )
    # Re-registering with same ID will raise error unless unregister first
    # For refresh, we'd typically update the existing object or replace it and then refresh
    # For this example, let's just trigger refresh without modifying original targets to show it works
    await router.refresh_target_embeddings()
    print("\n--- After Refresh, Query again ---")
    routed_target = await router.route_query("What are the scaling limitations of Vishustra?")
    if routed_target:
        print(f"  -> Routed to: {routed_target.name} (ID: {routed_target.id})")
    else:
        print("  -> No suitable route found.")

if __name__ == "__main__":
    # Ensure this block is only run when the file is executed directly
    # and not imported as part of the framework.
    asyncio.run(_example_usage())
