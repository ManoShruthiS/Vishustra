"""
Vishustra Semantic Router Module

This module provides a robust, intent-based semantic router designed to intelligently
direct incoming user queries or internal prompts to the most appropriate downstream
components within the Vishustra framework. It leverages advanced embedding models
and vector stores to perform semantic similarity matching against a registry of
pre-defined routes, each representing a specific intent or target component.

Features:
- Asynchronous operations for non-blocking I/O.
- Pluggable embedding models and vector store implementations.
- Pydantic models for strict data validation and clear route definitions.
- Configurable similarity threshold for precise matching.
- Scalable design, offloading vector storage and search to an external vector store.
- Comprehensive logging for observability.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4
import json # Used for robust handling of target_config in metadata

from pydantic import BaseModel, Field, PrivateAttr, ValidationError

logger = logging.getLogger(__name__)

# --- Vishustra Core Interfaces (Abstract Base Classes for Dependencies) ---
# These are defined here for self-containation and clarity, but would typically be
# imported from vishustra.core.abc or vishustra.components.interfaces in a full framework.

class AsyncEmbeddingModel(ABC):
    """
    Abstract Base Class for asynchronous embedding models within Vishustra.
    Defines the contract for models capable of generating vector embeddings.
    """

    @abstractmethod
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Asynchronously embeds a list of documents (strings) into a list of
        vector embeddings.

        Args:
            texts: A list of strings to embed.

        Returns:
            A list of lists of floats, where each inner list is an embedding vector.
        """
        raise NotImplementedError

    @abstractmethod
    async def embed_query(self, text: str) -> List[float]:
        """
        Asynchronously embeds a single query string into a single vector embedding.

        Args:
            text: The query string to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def dimension(self) -> int:
        """
        Returns the dimension of the embedding vectors produced by this model.
        """
        raise NotImplementedError

class VectorDocument(BaseModel):
    """
    Vishustra standard model for a document stored in a vector store.
    """
    id: str = Field(..., description="Unique identifier for the document.")
    text: Optional[str] = Field(None, description="Original text content associated with the embedding.")
    embedding: List[float] = Field(..., description="The vector embedding of the document.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata associated with the document.")

class AsyncVectorStore(ABC):
    """
    Abstract Base Class for asynchronous vector store implementations within Vishustra.
    Defines the contract for storing and searching vector embeddings.
    """

    @abstractmethod
    async def add_documents(self, documents: List[VectorDocument]) -> List[str]:
        """
        Asynchronously adds a list of VectorDocument objects to the store.

        Args:
            documents: A list of VectorDocument objects to add.

        Returns:
            A list of IDs for the added documents.
        """
        raise NotImplementedError

    @abstractmethod
    async def similarity_search_by_vector(
        self,
        embedding: List[float],
        top_k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[VectorDocument, float]]:
        """
        Asynchronously performs a similarity search using a query embedding.

        Args:
            embedding: The query embedding vector.
            top_k: The number of top similar documents to retrieve.
            filter: Optional dictionary to filter documents based on metadata.

        Returns:
            A list of tuples, each containing a VectorDocument and its similarity score.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_documents(self, ids: List[str]) -> None:
        """
        Asynchronously deletes documents by their IDs from the store.

        Args:
            ids: A list of document IDs to delete.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_document_by_id(self, doc_id: str) -> Optional[VectorDocument]:
        """
        Asynchronously retrieves a single document by its ID.

        Args:
            doc_id: The ID of the document to retrieve.

        Returns:
            The VectorDocument if found, otherwise None.
        """
        raise NotImplementedError

    # Future enhancement: A production-grade VectorStore would likely offer
    # methods for filtering and deleting based on metadata directly,
    # which would improve the `remove_route` and `get_route_definition` implementations.
    # @abstractmethod
    # async def get_documents_by_metadata(self, metadata_filter: Dict[str, Any]) -> List[VectorDocument]:
    #     """Retrieves documents matching a given metadata filter."""
    #     raise NotImplementedError
    #
    # @abstractmethod
    # async def delete_documents_by_metadata(self, metadata_filter: Dict[str, Any]) -> int:
    #     """Deletes documents matching a given metadata filter and returns the count of deleted documents."""
    #     raise NotImplementedError

# --- Semantic Router Pydantic Models ---

class RouteTarget(BaseModel):
    """
    Defines the destination or action for a matched route.
    This can be a reference to another chain, an agent, a tool, or a specific function.
    """
    target_id: str = Field(..., description="Unique identifier for the target component (e.g., 'summarize_chain', 'tool_web_search').")
    target_type: str = Field(..., description="Type of the target component (e.g., 'chain', 'agent', 'tool', 'function').")
    config: Dict[str, Any] = Field(default_factory=dict, description="Configuration parameters to pass to the target component.")

class RouteDefinition(BaseModel):
    """
    Defines a specific route for the semantic router.
    Each route is characterized by patterns (example queries/intent descriptions)
    and a target component it should direct to.
    """
    route_id: UUID = Field(default_factory=uuid4, description="Unique identifier for this route definition.")
    name: str = Field(..., description="Human-readable name for the route (e.g., 'Summarization Intent', 'Database Query').")
    description: str = Field(..., description="Detailed description of what this route handles.")
    patterns: List[str] = Field(..., min_items=1, description="Example queries or phrases that represent this route's intent.")
    target: RouteTarget = Field(..., description="The component or action this route directs to.")

class RouteMatch(BaseModel):
    """
    Represents a successful match found by the semantic router.
    """
    route_id: UUID = Field(..., description="The ID of the matched route definition.")
    name: str = Field(..., description="Name of the matched route.")
    description: str = Field(..., description="Description of the matched route.")
    target: RouteTarget = Field(..., description="The target component defined by the matched route.")
    similarity: float = Field(..., description="The similarity score between the query and the matched route pattern.")

# --- Semantic Router Implementation ---

class SemanticRouter:
    """
    The Vishustra SemanticRouter orchestrates intelligent routing of user queries
    or internal prompts based on semantic similarity to predefined route definitions.

    It utilizes an embedding model to convert text into vector representations and
    an asynchronous vector store for efficient storage and retrieval of route patterns.
    """

    _embedding_model: AsyncEmbeddingModel = PrivateAttr()
    _vector_store: AsyncVectorStore = PrivateAttr()
    _similarity_threshold: float = PrivateAttr()

    def __init__(
        self,
        embedding_model: AsyncEmbeddingModel,
        vector_store: AsyncVectorStore,
        similarity_threshold: float = 0.75,
    ):
        """
        Initializes the SemanticRouter.

        Args:
            embedding_model: An instance of AsyncEmbeddingModel to generate embeddings.
            vector_store: An instance of AsyncVectorStore to store and search route patterns.
            similarity_threshold: The minimum cosine similarity score required for a
                                  route to be considered a match (0.0 to 1.0).
        """
        if not isinstance(embedding_model, AsyncEmbeddingModel):
            raise TypeError("embedding_model must be an instance of AsyncEmbeddingModel.")
        if not isinstance(vector_store, AsyncVectorStore):
            raise TypeError("vector_store must be an instance of AsyncVectorStore.")
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0.")

        self._embedding_model = embedding_model
        self._vector_store = vector_store
        self._similarity_threshold = similarity_threshold
        logger.info(f"SemanticRouter initialized with embedding_model: {type(embedding_model).__name__}, "
                    f"vector_store: {type(vector_store).__name__}, "
                    f"similarity_threshold: {similarity_threshold}")

    async def add_route(self, route_definition: RouteDefinition) -> None:
        """
        Asynchronously adds a new route definition to the router.
        Each pattern in the route definition is embedded and stored individually
        in the vector store, linked back to the parent route_id via metadata.

        Args:
            route_definition: The RouteDefinition object to add.

        Raises:
            ValueError: If the route definition contains no patterns.
        """
        if not route_definition.patterns:
            logger.warning(f"Route '{route_definition.name}' (ID: {route_definition.route_id}) has no patterns. Skipping.")
            raise ValueError("RouteDefinition must contain at least one pattern.")

        pattern_embeddings = await self._embedding_model.embed_documents(route_definition.patterns)
        documents_to_add: List[VectorDocument] = []

        for i, pattern in enumerate(route_definition.patterns):
            # Each pattern gets a unique document ID, linked by route_id in metadata
            doc_id = f"{route_definition.route_id}-{i}"
            
            # Store full route metadata, including serialized target_config for robust retrieval
            metadata = {
                "route_id": str(route_definition.route_id),
                "route_name": route_definition.name,
                "route_description": route_definition.description,
                "target_id": route_definition.target.target_id,
                "target_type": route_definition.target.target_type,
                "target_config_json": json.dumps(route_definition.target.config), # Store as JSON string
                "original_pattern": pattern # Store original pattern for context/debugging
            }
            documents_to_add.append(
                VectorDocument(
                    id=doc_id,
                    text=pattern, # The pattern text itself
                    embedding=pattern_embeddings[i],
                    metadata=metadata
                )
            )

        await self._vector_store.add_documents(documents_to_add)
        logger.info(f"Added route '{route_definition.name}' (ID: {route_definition.route_id}) with {len(documents_to_add)} patterns.")

    async def remove_route(self, route_id: UUID) -> None:
        """
        Asynchronously removes a route and all its associated pattern documents
        from the vector store.

        NOTE: This implementation relies on the `AsyncVectorStore` having a
        mechanism to retrieve document IDs based on metadata or to delete
        documents directly by metadata. With the current `AsyncVectorStore`
        interface (defined within this file), this operation is not directly
        supported and would require fetching all documents and then deleting by ID.
        A more robust VectorStore would offer `delete_documents_by_metadata`.
        """
        logger.warning(f"Removal of route '{route_id}' is limited by current AsyncVectorStore interface. "
                       "For full efficiency, AsyncVectorStore would need `get_documents_by_metadata` "
                       "or `delete_documents_by_metadata` to retrieve/delete all patterns for a route_id.")
        
        # Placeholder for a more robust implementation:
        # In a production system with `get_documents_by_metadata` in AsyncVectorStore:
        # related_docs = await self._vector_store.get_documents_by_metadata({"route_id": str(route_id)})
        # doc_ids_to_delete = [doc.id for doc in related_docs]
        # if doc_ids_to_delete:
        #     await self._vector_store.delete_documents(doc_ids_to_delete)
        #     logger.info(f"Removed route '{route_id}' and {len(doc_ids_to_delete)} associated patterns.")
        # else:
        #     logger.info(f"No patterns found for route '{route_id}'.")

        # For the current interface, we will issue a warning and do nothing specific,
        # indicating this needs a future refinement of the VectorStore interface.
        logger.info(f"Attempted to remove route '{route_id}'. No documents deleted via current interface.")


    async def route(self, query: str, top_k: int = 1) -> Optional[RouteMatch]:
        """
        Asynchronously routes an incoming query to the most semantically similar
        predefined route.

        Args:
            query: The user query or prompt string.
            top_k: The number of top similar routes to consider before applying threshold.

        Returns:
            An Optional RouteMatch object if a suitable route is found within the
            similarity threshold, otherwise None.
        """
        if not query:
            logger.warning("Received empty query for routing. Returning None.")
            return None

        query_embedding = await self._embedding_model.embed_query(query)
        
        # Perform similarity search. The metadata for the original route definition
        # is stored directly within the VectorDocument's metadata.
        search_results = await self._vector_store.similarity_search_by_vector(
            embedding=query_embedding,
            top_k=top_k
        )

        if not search_results:
            logger.debug(f"No semantic matches found for query: '{query[:50]}...'")
            return None

        best_match: Optional[RouteMatch] = None
        for doc, similarity in search_results:
            if similarity >= self._similarity_threshold:
                try:
                    # Attempt to deserialize target_config from JSON string
                    target_config = {}
                    config_json_str = doc.metadata.get("target_config_json")
                    if isinstance(config_json_str, str):
                        try:
                            target_config = json.loads(config_json_str)
                        except json.JSONDecodeError:
                            logger.error(f"Failed to decode target_config_json for route pattern {doc.id}. Defaulting to empty dict.")
                    elif isinstance(config_json_str, dict): # Handle cases where it might be a dict already
                        target_config = config_json_str
                    
                    matched_route = RouteMatch(
                        route_id=UUID(doc.metadata["route_id"]),
                        name=doc.metadata["route_name"],
                        description=doc.metadata["route_description"],
                        target=RouteTarget(
                            target_id=doc.metadata["target_id"],
                            target_type=doc.metadata["target_type"],
                            config=target_config
                        ),
                        similarity=similarity
                    )
                    
                    if best_match is None or similarity > best_match.similarity:
                         best_match = matched_route

                except (KeyError, ValueError, ValidationError, TypeError) as e:
                    logger.error(f"Failed to reconstruct RouteMatch from document metadata (ID: {doc.id}): {e}. Metadata: {doc.metadata}")
                    continue
            else:
                logger.debug(f"Skipping match (ID: {doc.id}) with similarity {similarity:.2f} below threshold {self._similarity_threshold:.2f}.")

        if best_match:
            logger.info(f"Query '{query[:50]}...' routed to '{best_match.name}' (ID: {best_match.route_id}) with similarity {best_match.similarity:.2f}.")
        else:
            logger.debug(f"No suitable route found within threshold {self._similarity_threshold:.2f} for query: '{query[:50]}...'")

        return best_match

    async def get_route_definition(self, route_id: UUID) -> Optional[RouteDefinition]:
        """
        Retrieves a full RouteDefinition by its ID.
        
        NOTE: With the current `AsyncVectorStore` interface, reconstructing the
        complete list of `patterns` for a `RouteDefinition` is not straightforward
        as patterns are stored as individual documents. This method will attempt
        to reconstruct the `RouteDefinition` from a single associated pattern document
        (specifically, the one with index '0' in its ID), and therefore the
        `patterns` list in the returned `RouteDefinition` might be incomplete,
        containing only the pattern from the retrieved document.
        A more capable `AsyncVectorStore` (`get_documents_by_metadata`) or a
        separate metadata store for `RouteDefinition` objects would be needed
        for full reconstruction.
        """
        # We'll try to retrieve the first pattern document associated with this route_id.
        # This assumes that if a route was added, at least one pattern document exists.
        doc_id_guess = f"{route_id}-0"
        doc = await self._vector_store.get_document_by_id(doc_id_guess)

        if not doc or doc.metadata.get("route_id") != str(route_id):
            logger.debug(f"Could not find primary pattern document '{doc_id_guess}' for route {route_id}.")
            return None

        try:
            target_config = {}
            config_json_str = doc.metadata.get("target_config_json")
            if isinstance(config_json_str, str):
                try:
                    target_config = json.loads(config_json_str)
                except json.JSONDecodeError:
                    logger.error(f"Failed to decode target_config_json for route {route_id} from doc {doc_id_guess}. Defaulting to empty dict.")
            elif isinstance(config_json_str, dict):
                target_config = config_json_str

            logger.warning(f"get_route_definition for {route_id} can only reconstruct metadata from one pattern document "
                           "with current AsyncVectorStore interface. Full list of patterns may be incomplete.")
            
            # Reconstruct with the pattern from the retrieved document as a single pattern in the list
            return RouteDefinition(
                route_id=route_id,
                name=doc.metadata["route_name"],
                description=doc.metadata["route_description"],
                patterns=[doc.text] if doc.text else [], # Only one pattern can be reliably reconstructed this way
                target=RouteTarget(
                    target_id=doc.metadata["target_id"],
                    target_type=doc.metadata["target_type"],
                    config=target_config
                )
            )

        except (KeyError, ValueError, ValidationError, TypeError) as e:
            logger.error(f"Failed to reconstruct RouteDefinition for ID {route_id} from document ID {doc_id_guess}: {e}.")
            return None

# --- Example Usage (Conceptual - would typically be in a separate test/example file) ---
if __name__ == "__main__":
    import asyncio

    # Set up basic logging for the example
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Uncomment next line for more verbose debugging output from router and dummy components
    # logging.getLogger(__name__).setLevel(logging.DEBUG)

    # Dummy implementations for demonstration purposes.
    # In a real Vishustra setup, these would be concrete implementations
    # connecting to actual models (e.g., OpenAIEmbeddings) and vector databases (e.g., Qdrant, Chroma).
    class DummyEmbeddingModel(AsyncEmbeddingModel):
        def __init__(self, dimension: int = 768):
            self._dimension = dimension
            logger.info(f"Initialized DummyEmbeddingModel with dimension {dimension}")

        async def embed_documents(self, texts: List[str]) -> List[List[float]]:
            logger.debug(f"Dummy embedding {len(texts)} documents.")
            # Generate somewhat distinct but random-like embeddings based on text hash
            return [[float(hash(t + str(i)) % 1000) / 1000.0 for i in range(self._dimension)] for t in texts]

        async def embed_query(self, text: str) -> List[float]:
            logger.debug(f"Dummy embedding query: '{text[:20]}...'")
            return [float(hash(text + "query") % 1000) / 1000.0 for _ in range(self._dimension)]

        @property
        def dimension(self) -> int:
            return self._dimension

    class InMemoryVectorStore(AsyncVectorStore):
        def __init__(self):
            self._store: Dict[str, VectorDocument] = {}
            logger.info("Initialized InMemoryVectorStore.")

        async def add_documents(self, documents: List[VectorDocument]) -> List[str]:
            ids = []
            for doc in documents:
                if doc.id in self._store:
                    logger.warning(f"Document with ID {doc.id} already exists. Overwriting.")
                self._store[doc.id] = doc
                ids.append(doc.id)
            logger.debug(f"Added {len(documents)} documents to in-memory store.")
            return ids

        async def similarity_search_by_vector(
            self,
            embedding: List[float],
            top_k: int = 4,
            filter: Optional[Dict[str, Any]] = None,
        ) -> List[Tuple[VectorDocument, float]]:
            
            results: List[Tuple[VectorDocument, float]] = []
            
            # Simple cosine similarity calculation
            def cosine_similarity(vec1, vec2):
                if not vec1 or not vec2: return 0.0
                dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
                magnitude1 = sum(v1*v1 for v1 in vec1)**0.5
                magnitude2 = sum(v2*v2 for v2 in vec2)**0.5
                if magnitude1 == 0 or magnitude2 == 0:
                    return 0.0
                return dot_product / (magnitude1 * magnitude2)

            for doc_id, doc in self._store.items():
                # Apply metadata filter if provided
                if filter:
                    if not all(doc.metadata.get(k) == v for k, v in filter.items()):
                        continue
                
                sim = cosine_similarity(embedding, doc.embedding)
                results.append((doc, sim))
            
            results.sort(key=lambda x: x[1], reverse=True)
            logger.debug(f"Performed in-memory similarity search. Found {len(results)} matches.")
            return results[:top_k]

        async def delete_documents(self, ids: List[str]) -> None:
            deleted_count = 0
            for doc_id in ids:
                if doc_id in self._store:
                    del self._store[doc_id]
                    deleted_count += 1
            logger.debug(f"Deleted {deleted_count} documents from in-memory store.")

        async def get_document_by_id(self, doc_id: str) -> Optional[VectorDocument]:
            logger.debug(f"Retrieving document by ID: {doc_id}")
            return self._store.get(doc_id)

    async def run_example():
        embed_model = DummyEmbeddingModel()
        vector_store = InMemoryVectorStore()
        router = SemanticRouter(embedding_model=embed_model, vector_store=vector_store, similarity_threshold=0.7)

        # Define several example routes for the framework
        route_summarize = RouteDefinition(
            name="Summarization Chain",
            description="Route for summarizing long documents or conversations.",
            patterns=[
                "summarize this document",
                "can you give me a summary?",
                "tl;dr of this text",
                "condense this information"
            ],
            target=RouteTarget(target_id="summarize_chain_v1", target_type="chain", config={"model_name": "gpt-4-turbo", "length": "short"})
        )

        route_qa = RouteDefinition(
            name="Knowledge Base QA",
            description="Route for answering questions from a specific internal knowledge base.",
            patterns=[
                "what is Vishustra?",
                "how does the system work?",
                "questions about framework",
                "explain the architecture",
                "where can I find documentation?"
            ],
            target=RouteTarget(target_id="kb_qa_agent", target_type="agent", config={"kb_name": "vishustra_docs_v2"})
        )

        route_tool_search = RouteDefinition(
            name="Web Search Tool",
            description="Route for queries requiring external web search capabilities.",
            patterns=[
                "what's the weather in Paris?",
                "latest news on AI developments",
                "search for recent research in quantum computing",
                "find information about"
            ],
            target=RouteTarget(target_id="web_search_tool", target_type="tool", config={"provider": "bing"})
        )
        
        route_tool_calendar = RouteDefinition(
            name="Calendar Tool",
            description="Route for queries related to calendar management, scheduling, or reminders.",
            patterns=[
                "schedule a meeting for tomorrow at 10 AM",
                "add event to calendar: team sync on Friday",
                "what's on my schedule for next Tuesday?",
                "remind me about the project deadline"
            ],
            target=RouteTarget(target_id="calendar_tool", target_type="tool", config={"user_context": "current_user"})
        )

        logger.info("\n--- Adding routes to the router ---")
        await router.add_route(route_summarize)
        await router.add_route(route_qa)
        await router.add_route(route_tool_search)
        await router.add_route(route_tool_calendar)
        logger.info("All routes added.")

        # Test various queries against the router
        test_queries = [
            "Please summarize this long email for me.",
            "What is the core functionality of Vishustra framework?",
            "Search the web for how to train a large language model.",
            "What's my schedule look like next Tuesday?",
            "Tell me about something completely unrelated, like the history of jazz.", # Should ideally not match strongly
            "Can you give me a summary of the provided text?",
            "how to use agent in vishustra?",
            "find facts about mars exploration", # Ambiguous, might hit web search
            "add a reminder for my dentist appointment",
        ]

        for query in test_queries:
            print(f"\n--- Routing Query: '{query}' ---")
            match = await router.route(query)
            if match:
                print(f"  -> Matched Route: '{match.name}' (Similarity: {match.similarity:.2f})")
                print(f"     Target: {match.target.target_type} '{match.target.target_id}' with config: {match.target.config}")
            else:
                print("  -> No suitable route found above similarity threshold.")
        
        # Test `get_route_definition` (with noted limitation)
        print(f"\n--- Testing `get_route_definition` for Knowledge Base QA route (ID: {route_qa.route_id}) ---")
        retrieved_route_qa = await router.get_route_definition(route_qa.route_id)
        if retrieved_route_qa:
            print(f"  Retrieved Route Name: {retrieved_route_qa.name}")
            print(f"  Retrieved Description: {retrieved_route_qa.description}")
            print(f"  Retrieved Patterns (partial): {retrieved_route_qa.patterns}")
            print(f"  Retrieved Target: {retrieved_route_qa.target.target_id} ({retrieved_route_qa.target.target_type}) with config: {retrieved_route_qa.target.config}")
        else:
            print(f"  Could not retrieve RouteDefinition for ID {route_qa.route_id}.")

        # Test removing a route (note: current `remove_route` is a placeholder due to `AsyncVectorStore` interface)
        print(f"\n--- Attempting to remove route: '{route_summarize.name}' (ID: {route_summarize.route_id}) ---")
        await router.remove_route(route_summarize.route_id) 
        
        print(f"\n--- Re-routing query after attempted removal: '{test_queries[0]}' ---")
        match_after_removal = await router.route(test_queries[0])
        if match_after_removal:
            print(f"  -> Matched Route: '{match_after_removal.name}' (Similarity: {match_after_removal.similarity:.2f}) (Note: Deletion might not be effective with dummy vector store's current interface for metadata-based removal.)")
        else:
            print("  -> No suitable route found. (Expected if deletion was fully effective)")

    asyncio.run(run_example())