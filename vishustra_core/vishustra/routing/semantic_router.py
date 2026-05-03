import abc
import asyncio
import json
import logging
from enum import Enum
from typing import Dict, List, Literal, Optional, Protocol, Tuple, Type, TypeVar

from pydantic import BaseModel, Field, ValidationError, parse_obj_as

# --- Vishustra Core Protocols and Models (Assumed to be in respective modules) ---
# For demonstration purposes, these foundational types are included here.
# In a real Vishustra setup, they would be imported from e.g., vishustra.llms, vishustra.embeddings, etc.

class BaseLLM(Protocol):
    """Abstract base class (Protocol) for Language Model clients in Vishustra."""
    async def invoke(self, prompt: str, **kwargs) -> str:
        """
        Invokes the LLM with the given prompt and returns its response.

        Args:
            prompt: The input prompt string for the LLM.
            **kwargs: Additional parameters specific to the LLM (e.g., `temperature`, `model`, `response_format`).

        Returns:
            The raw string response from the LLM.
        """
        ...

class BaseEmbedder(Protocol):
    """Abstract base class (Protocol) for Embedding Model clients in Vishustra."""
    async def embed_query(self, text: str) -> List[float]:
        """
        Generates an embedding for a single text query.

        Args:
            text: The text string to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        ...
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a list of document texts.

        Args:
            texts: A list of text strings to embed.

        Returns:
            A list of embedding vectors, where each vector is a list of floats.
        """
        ...

class VectorStoreDocument(BaseModel):
    """
    Represents a document stored in a vector store, typically associated with an embedding.
    This model facilitates consistent data transfer between Vishustra components and vector stores.
    """
    id: str = Field(description="Unique identifier for the document.")
    content: str = Field(description="The textual content of the document.")
    embedding: Optional[List[float]] = Field(None, description="The vector embedding of the content.")
    metadata: Dict = Field(default_factory=dict, description="Additional key-value pairs associated with the document.")

class BaseVectorStore(Protocol):
    """
    Abstract base class (Protocol) for Vector Store clients in Vishustra.
    Defines the essential interface for interacting with various vector database backends.
    """
    async def add_documents(self, documents: List[VectorStoreDocument], **kwargs):
        """
        Adds documents to the vector store. Embeddings should ideally be pre-computed
        and included in the `VectorStoreDocument` objects, or the store might handle it.

        Args:
            documents: A list of `VectorStoreDocument` instances to add.
            **kwargs: Additional parameters for the vector store's add operation.
        """
        ...
    async def search(self, query_embedding: List[float], k: int = 4, **kwargs) -> List[VectorStoreDocument]:
        """
        Searches the vector store for documents similar to the provided query embedding.
        Results should ideally include a 'similarity_score' in their metadata when returned.

        Args:
            query_embedding: The embedding vector of the query.
            k: The number of top similar documents to retrieve.
            **kwargs: Additional parameters for the vector store's search operation.

        Returns:
            A list of `VectorStoreDocument` instances that are most similar to the query.
        """
        ...
    async def delete(self, ids: List[str]):
        """
        Deletes documents from the vector store by their unique IDs.

        Args:
            ids: A list of document IDs to delete.
        """
        ...
    async def clear(self):
        """
        Clears all documents from the vector store. Use with caution.
        """
        ...

# --- End Vishustra Core Protocols and Models ---

logger = logging.getLogger(__name__)

# --- Configuration Models ---

class RouteDefinition(BaseModel):
    """
    Defines a specific route that the SemanticPromptRouter can direct prompts to.
    Each route represents a distinct downstream capability or workflow in Vishustra.
    """
    id: str = Field(description="Unique identifier for the route (e.g., 'customer_support', 'data_query').")
    description: str = Field(description="A concise description of what this route handles and its purpose.")
    example_prompts: List[str] = Field(
        default_factory=list,
        description="A list of example user prompts that should ideally be directed to this route. "
                    "These examples are used for vector similarity matching and as few-shot examples for LLM classification."
    )
    metadata: Dict = Field(
        default_factory=dict,
        description="Optional additional key-value pair metadata associated with the route. "
                    "This metadata will be included in the `RoutingDecision` if this route is chosen, "
                    "allowing downstream components to dynamically adapt."
    )

class RoutingStrategy(str, Enum):
    """Defines the primary strategy used by the semantic router to make routing decisions."""
    LLM_CLASSIFICATION = "llm_classification"
    VECTOR_SIMILARITY = "vector_similarity"
    HYBRID = "hybrid" # Combines LLM classification and vector similarity, prioritizing more confident outcomes.

class SemanticRouterConfig(BaseModel):
    """
    Configuration for the `SemanticPromptRouter`. This model dictates the behavior
    and parameters used for routing prompts.
    """
    routes: List[RouteDefinition] = Field(
        default_factory=list,
        description="A list of `RouteDefinition` objects to be loaded and considered by the router upon initialization."
    )
    default_route_id: Optional[str] = Field(
        None,
        description="The ID of a predefined route to fall back to if no suitable route is found "
                    "or if all identified routes have a confidence score below the `confidence_threshold`."
    )
    strategy: RoutingStrategy = Field(
        RoutingStrategy.HYBRID,
        description="The primary strategy to use for prompt routing. Options are "
                    "`LLM_CLASSIFICATION`, `VECTOR_SIMILARITY`, or `HYBRID`."
    )
    confidence_threshold: float = Field(
        0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score (between 0.0 and 1.0) for a route to be considered valid. "
                    "Decisions with confidence below this threshold will be treated as 'no match' "
                    "and may trigger a fallback to the `default_route_id`."
    )
    top_k_vector_search: int = Field(
        3,
        ge=1,
        description="Number of top similar documents (example prompts) to retrieve from the vector store "
                    "during `VECTOR_SIMILARITY` or `HYBRID` routing to inform the decision."
    )
    llm_temperature: float = Field(
        0.2,
        ge=0.0,
        le=1.0,
        description="Temperature setting for the LLM when performing classification. "
                    "Lower values (closer to 0.0) make the LLM's output more deterministic and focused, "
                    "while higher values encourage more diverse responses."
    )
    llm_model_name: Optional[str] = Field(
        None,
        description="Specific LLM model name to use for classification. If `None`, the default model "
                    "of the provided `llm_client` instance will be used."
    )
    enable_dynamic_vector_store_updates: bool = Field(
        True,
        description="If `True`, `add_route` and `remove_route` methods will automatically update "
                    "the underlying vector store with example prompts. Set to `False` if the vector store "
                    "is managed externally or is read-only for route examples."
    )

# --- Routing Outcomes ---

class RoutingDecision(BaseModel):
    """
    Represents the detailed outcome of a routing operation by the `SemanticPromptRouter`.
    This object encapsulates the chosen route and all relevant information about the decision.
    """
    route_id: str = Field(description="The identifier of the chosen route. Will be 'none' if no suitable route was found.")
    confidence: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="A confidence score (0.0 to 1.0) indicating the certainty of the decision. "
                    "0.0 typically indicates no confidence or no valid match."
    )
    explanation: Optional[str] = Field(
        None,
        description="A brief explanation for why this route was chosen. This could be from LLM reasoning, "
                    "vector similarity matching details, or fallback logic."
    )
    matched_document_ids: List[str] = Field(
        default_factory=list,
        description="A list of IDs of documents from the vector store (e.g., example prompts) that "
                    "contributed significantly to the routing decision, primarily in vector-based strategies."
    )
    metadata: Dict = Field(
        default_factory=dict,
        description="Any additional metadata associated with the chosen route definition. "
                    "This is directly propagated from the `RouteDefinition`."
    )
    fallback_used: bool = Field(
        False,
        description="`True` if the `default_route_id` was used because no confident match was found "
                    "through the primary routing strategy."
    )

# --- Router Base Class ---

R = TypeVar('R', bound=RoutingDecision)

class BasePromptRouter(abc.ABC):
    """
    Abstract base class for all prompt routers in Vishustra.

    Defines the fundamental interface for routing incoming prompts to appropriate
    downstream components (e.g., agents, chains, specific tool sets) based on various strategies.
    """

    @abc.abstractmethod
    async def route(self, prompt: str, context: Optional[Dict] = None) -> R:
        """
        Routes an incoming prompt to an appropriate handler based on its content, context, or configured rules.

        Args:
            prompt: The user's input prompt string to be routed.
            context: Optional dictionary of additional context for routing (e.g., user ID, session state, internal metadata).

        Returns:
            A `RoutingDecision` object containing the chosen route ID and related information.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def add_route(self, route_definition: RouteDefinition):
        """
        Adds or updates a new route definition to the router's consideration set.

        Args:
            route_definition: The `RouteDefinition` object to add or update.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def remove_route(self, route_id: str):
        """
        Removes an existing route definition from the router's consideration set.

        Args:
            route_id: The unique identifier of the route to remove.
        """
        raise NotImplementedError

# --- Semantic Prompt Router Implementation ---

class SemanticPromptRouter(BasePromptRouter):
    """
    An advanced semantic prompt router for 'Vishustra' that leverages Language Models (LLMs)
    and vector similarity search to dynamically direct incoming user prompts to the most
    appropriate downstream workflow or agent.

    This router supports flexible configuration of routing strategies (LLM-based classification,
    vector similarity matching, or a hybrid approach) and allows for dynamic management of routes.
    It intelligently determines the most suitable component based on the semantic meaning
    and intent of the user's input, enhancing modularity and adaptability in LLM applications.

    Dependencies:
    - `llm_client`: An instance of an LLM client implementing `BaseLLM` for semantic classification.
    - `embedder_client`: An instance of an embedding client implementing `BaseEmbedder` for generating
      vector representations of prompts and route examples.
    - `vector_store_client`: An instance of a vector store client implementing `BaseVectorStore`
      for efficient similarity search against example prompts, crucial for vector-based routing.
    """

    def __init__(
        self,
        config: SemanticRouterConfig,
        llm_client: BaseLLM,
        embedder_client: BaseEmbedder,
        vector_store_client: BaseVectorStore
    ):
        """
        Initializes the SemanticPromptRouter.

        Args:
            config: The configuration object for the router, defining routes, strategy, confidence thresholds, etc.
            llm_client: An instance of an LLM client (e.g., for OpenAI, Anthropic, or local LLMs).
            embedder_client: An instance of an embedding client (e.g., for OpenAI Embeddings, HuggingFace embeddings).
            vector_store_client: An instance of a vector store client (e.g., Qdrant, Pinecone, ChromaDB).
        """
        if not isinstance(config, SemanticRouterConfig):
            raise TypeError("`config` must be an instance of `SemanticRouterConfig`.")
        # Runtime type checks for protocols are less strict, but good practice for robustness.
        # `isinstance` checks against `Protocol` classes might not always work as expected for all subclasses/implementations
        # directly at runtime unless they explicitly inherit from `Protocol`.
        # For simplicity, we assume adherence to the protocol interface here.

        self._config = config
        self._llm = llm_client
        self._embedder = embedder_client
        self._vector_store = vector_store_client
        self._routes: Dict[str, RouteDefinition] = {r.id: r for r in config.routes}
        self._initialized_vector_store = asyncio.Event() # Used to signal when the vector store is ready

        # Kick off asynchronous initialization of the vector store with initial routes.
        # This allows the router to be instantiated quickly while setup proceeds in the background.
        asyncio.create_task(self._initialize_vector_store())

        logger.info(
            f"SemanticPromptRouter initialized with strategy: {self._config.strategy.value}. "
            f"Total initial routes loaded: {len(self._routes)}"
        )
        if self._config.default_route_id:
            logger.info(f"Default fallback route set to: '{self._config.default_route_id}'")

    async def _initialize_vector_store(self):
        """
        Initializes the vector store by embedding all example prompts from the
        predefined routes and adding them to the store. This process runs once
        asynchronously during router startup. If `enable_dynamic_vector_store_updates`
        is False, this step may be skipped or handle minimal setup.
        """
        if not self._config.enable_dynamic_vector_store_updates and \
           (self._config.strategy == RoutingStrategy.VECTOR_SIMILARITY or self._config.strategy == RoutingStrategy.HYBRID):
            logger.warning(
                "Dynamic vector store updates are disabled, but vector-based strategies are in use. "
                "Ensure vector store for route examples is pre-populated correctly externally if routes change. "
                "Skipping internal vector store initialization."
            )
            self._initialized_vector_store.set() # Signal completion even if nothing was done
            return

        logger.info("Initializing vector store for semantic router with initial routes...")
        documents_to_add: List[VectorStoreDocument] = []
        for route_id, route_def in self._routes.items():
            if not route_def.example_prompts and self._config.strategy in [
                RoutingStrategy.VECTOR_SIMILARITY, RoutingStrategy.HYBRID
            ]:
                logger.debug(
                    f"Route '{route_id}' has no example prompts defined. "
                    "It will not be discoverable by vector similarity search methods."
                )
            for i, example_prompt in enumerate(route_def.example_prompts):
                doc_id = f"{route_id}__example__{i}" # Generate a unique ID for each example prompt
                documents_to_add.append(
                    VectorStoreDocument(
                        id=doc_id,
                        content=example_prompt,
                        metadata={
                            "route_id": route_id,
                            "route_description": route_def.description,
                            "original_metadata": route_def.metadata # Store original metadata for potential retrieval
                        }
                    )
                )

        if documents_to_add:
            try:
                # Embed documents in batches to optimize API calls or memory usage.
                # The embedder might handle internal batching, but explicit batching here
                # adds an extra layer of control for potentially very large lists.
                batch_size = 100
                for i in range(0, len(documents_to_add), batch_size):
                    batch_docs = documents_to_add[i:i + batch_size]
                    contents_batch = [doc.content for doc in batch_docs]
                    embeddings_batch = await self._embedder.embed_documents(contents_batch)
                    for j, doc in enumerate(batch_docs):
                        doc.embedding = embeddings_batch[j]
                    await self._vector_store.add_documents(batch_docs)
                logger.info(f"Added {len(documents_to_add)} initial route examples to vector store.")
            except Exception as e:
                logger.error(f"Failed to initialize vector store with initial routes: {e}", exc_info=True)
                # If vector store initialization fails, vector-based strategies will be severely impacted.
                # The router might still function for LLM_CLASSIFICATION if the LLM client is independent.
        else:
            logger.warning(
                "No example prompts found across all initial routes. "
                "Vector similarity routing will be ineffective unless routes are added dynamically with examples."
            )
        self._initialized_vector_store.set() # Signal that initialization is complete
        logger.info("Vector store initialization complete.")

    async def add_route(self, route_definition: RouteDefinition):
        """
        Adds a new route definition to the router or updates an existing one.
        If `enable_dynamic_vector_store_updates` is True, it will also update the vector store
        by removing old examples (if updating) and adding new ones.

        Args:
            route_definition: The `RouteDefinition` object to add or update.
        """
        await self._initialized_vector_store.wait() # Ensure vector store is ready before modifying it

        if route_definition.id in self._routes:
            logger.info(f"Route '{route_definition.id}' already exists. Updating its definition.")
            # Remove old examples from vector store if they exist for this route
            await self._remove_route_from_vector_store(route_definition.id)

        self._routes[route_definition.id] = route_definition # Update or add the route in memory

        if self._config.enable_dynamic_vector_store_updates and (
            self._config.strategy == RoutingStrategy.VECTOR_SIMILARITY or self._config.strategy == RoutingStrategy.HYBRID
        ):
            if route_definition.example_prompts:
                documents_to_add: List[VectorStoreDocument] = []
                for i, example_prompt in enumerate(route_definition.example_prompts):
                    doc_id = f"{route_definition.id}__example__{i}"
                    documents_to_add.append(
                        VectorStoreDocument(
                            id=doc_id,
                            content=example_prompt,
                            metadata={
                                "route_id": route_definition.id,
                                "route_description": route_definition.description,
                                "original_metadata": route_definition.metadata
                            }
                        )
                    )
                if documents_to_add:
                    try:
                        contents = [doc.content for doc in documents_to_add]
                        embeddings = await self._embedder.embed_documents(contents)
                        for i, doc in enumerate(documents_to_add):
                            doc.embedding = embeddings[i]
                        await self._vector_store.add_documents(documents_to_add)
                        logger.info(f"Added {len(documents_to_add)} examples for route '{route_definition.id}' to vector store.")
                    except Exception as e:
                        logger.error(
                            f"Failed to add examples for route '{route_definition.id}' to vector store: {e}", exc_info=True
                        )
            else:
                logger.debug(f"Route '{route_definition.id}' has no example prompts. No vector store update needed.")
        else:
            logger.debug(
                f"Dynamic vector store updates are disabled or current strategy does not require it for route '{route_definition.id}'. "
                "Vector store was not updated."
            )
        logger.info(f"Route '{route_definition.id}' added/updated successfully.")

    async def _remove_route_from_vector_store(self, route_id: str):
        """
        Helper method to remove documents associated with a specific route from the vector store.
        This is typically called when a route is updated or removed, to maintain data integrity.
        """
        if not self._config.enable_dynamic_vector_store_updates:
            logger.debug(f"Dynamic vector store updates disabled. Not removing route '{route_id}' examples from vector store.")
            return

        # Assuming example IDs follow the pattern {route_id}__example__{index}
        # A more robust solution might query the vector store directly by metadata ('route_id') if supported,
        # or maintain an internal mapping of route_id to associated document IDs.
        route_def = self._routes.get(route_id)
        if not route_def or not route_def.example_prompts:
            logger.debug(f"No example prompts found for route '{route_id}', skipping vector store deletion.")
            return

        example_doc_ids = [f"{route_id}__example__{i}" for i in range(len(route_def.example_prompts))]
        if example_doc_ids:
            try:
                await self._vector_store.delete(example_doc_ids)
                logger.info(f"Removed associated examples for route '{route_id}' from vector store.")
            except Exception as e:
                logger.error(
                    f"Failed to delete vector store examples for route '{route_id}': {e}", exc_info=True
                )

    async def remove_route(self, route_id: str):
        """
        Removes a route definition from the router's in-memory configuration.
        If `enable_dynamic_vector_store_updates` is True, it will also remove
        all associated example prompts from the configured vector store.

        Args:
            route_id: The unique identifier of the route to remove.
        """
        await self._initialized_vector_store.wait() # Ensure vector store is ready

        if route_id not in self._routes:
            logger.warning(f"Attempted to remove non-existent route '{route_id}'. No action taken.")
            return

        await self._remove_route_from_vector_store(route_id) # Remove from vector store first
        del self._routes[route_id] # Then remove from in-memory dictionary
        logger.info(f"Route '{route_id}' removed successfully.")

    async def _route_with_llm_classification(self, prompt: str) -> RoutingDecision:
        """
        Routes the prompt using an LLM to classify it against predefined routes.
        The LLM is carefully prompted to output a structured JSON response for reliable parsing.
        """
        if not self._routes:
            return RoutingDecision(route_id="none", confidence=0.0, explanation="No routes defined for LLM classification.")

        # Dynamically construct a clear, structured prompt for the LLM based on current routes
        route_descriptions = []
        for r_id, r_def in self._routes.items():
            examples_str = (
                f"Example prompts: {'; '.join(r_def.example_prompts[:3])}"
                if r_def.example_prompts
                else "No examples provided."
            )
            route_descriptions.append(
                f"- Route ID: '{r_id}'\n"
                f"  Description: {r_def.description}\n"
                f"  {examples_str}"
            )

        system_prompt = (
            "You are an intelligent routing system for a large language model orchestration framework. "
            "Your task is to analyze the user's prompt and accurately determine which of the available routes "
            "it should be directed to. Carefully consider the user's intent, keywords, and overall context.\n\n"
            "Available routes are described below. Each route has a unique ID, a clear description, "
            "and optionally example prompts that it typically handles:\n"
            f"{'\\n\\n'.join(route_descriptions)}\n\n"
            "Respond ONLY with a JSON object. The JSON object must contain three keys:\n"
            "- 'route_id': (string) The ID of the most appropriate route. If no route is suitable, return 'none'.\n"
            "- 'confidence': (float) Your confidence in this decision, a value between 0.0 (no confidence) and 1.0 (very confident).\n"
            "- 'explanation': (string) A concise, one-sentence explanation for your choice.\n\n"
            "Example JSON for a support route: "
            "{'route_id': 'customer_support_flow', 'confidence': 0.95, 'explanation': 'The user is clearly asking for help with a product issue.'}\n"
            "Example JSON for no match: "
            "{'route_id': 'none', 'confidence': 0.0, 'explanation': 'The prompt does not match any known route criteria.'}\n\n"
            "Now, analyze the following user prompt and provide your routing decision."
        )

        llm_response_str = "" # Initialize for error reporting
        try:
            llm_response_str = await self._llm.invoke(
                prompt=f"{system_prompt}\nUser Prompt: {prompt}",
                temperature=self._config.llm_temperature,
                model=self._config.llm_model_name, # Pass model name if specified in config
                response_format={"type": "json_object"} # Instruct LLM to output JSON
            )
            llm_response_json = json.loads(llm_response_str)

            # Validate LLM output against Pydantic model for robustness and type safety
            decision_data = {
                "route_id": llm_response_json.get("route_id"),
                "confidence": llm_response_json.get("confidence", 0.0),
                "explanation": llm_response_json.get("explanation"),
            }
            decision = parse_obj_as(RoutingDecision, decision_data)

            # Post-processing: Additional checks for suggested route validity and confidence threshold
            if decision.route_id != "none" and decision.route_id not in self._routes:
                logger.warning(
                    f"LLM suggested an unknown route ID '{decision.route_id}' for prompt '{prompt[:50]}...'. "
                    "Treating this as no match due to invalid route ID."
                )
                decision.route_id = "none"
                decision.confidence = 0.0 # Force confidence to 0 for invalid route
                decision.explanation = f"LLM suggested an invalid route ID: '{decision.route_id}'."

            # If the LLM's confidence is below the configured threshold, treat as 'no match'
            if decision.confidence < self._config.confidence_threshold:
                logger.debug(
                    f"LLM decision for '{prompt[:50]}...' (route: {decision.route_id}, conf: {decision.confidence:.2f}) "
                    f"is below the configured confidence threshold ({self._config.confidence_threshold:.2f}). "
                    "Treating as no valid match."
                )
                decision.route_id = "none"
                decision.confidence = 0.0
                decision.explanation = (
                    decision.explanation + " (Confidence below threshold.)"
                    if decision.explanation else "Confidence below threshold."
                )

            logger.info(
                f"LLM classification for '{prompt[:50]}...' resulted in: "
                f"Route ID: {decision.route_id}, Confidence: {decision.confidence:.2f}"
            )
            return decision

        except (json.JSONDecodeError, ValidationError, KeyError) as e:
            logger.error(
                f"Failed to parse or validate LLM response for routing '{prompt[:50]}...': {e}. "
                f"Raw LLM response: '{llm_response_str}'", exc_info=True
            )
            return RoutingDecision(route_id="none", confidence=0.0, explanation=f"LLM response parsing/validation error: {e}")
        except Exception as e:
            logger.error(
                f"Unhandled error during LLM classification for routing '{prompt[:50]}...': {e}", exc_info=True
            )
            return RoutingDecision(route_id="none", confidence=0.0, explanation=f"LLM invocation error: {e}")

    async def _route_with_vector_similarity(self, prompt: str) -> RoutingDecision:
        """
        Routes the prompt by finding the most semantically similar example prompts in the vector store.
        The route associated with the most similar examples is chosen.
        """
        await self._initialized_vector_store.wait() # Ensure vector store has completed initial population

        if not self._routes:
            return RoutingDecision(route_id="none", confidence=0.0, explanation="No routes defined for vector similarity.")

        # Check if the vector store is expected to be populated for this strategy
        if not (self._config.enable_dynamic_vector_store_updates or any(r.example_prompts for r in self._routes.values())):
            logger.warning(
                "Vector store is not dynamically updated and no initial example prompts were provided. "
                "Vector similarity routing will likely be ineffective. Returning no match."
            )
            return RoutingDecision(route_id="none", confidence=0.0, explanation="Vector store unpopulated/not dynamic for vector strategy.")

        try:
            query_embedding = await self._embedder.embed_query(prompt)
            results = await self._vector_store.search(
                query_embedding=query_embedding,
                k=self._config.top_k_vector_search
            )

            if not results:
                logger.debug(f"Vector search for '{prompt[:50]}...' returned no results.")
                return RoutingDecision(route_id="none", confidence=0.0, explanation="No similar examples found in vector store.")

            # Aggregate results by route_id and determine the best matching route based on highest similarity
            route_confidences: Dict[str, List[float]] = {}
            matched_doc_ids_per_route: Dict[str, List[str]] = {}

            for doc in results:
                route_id = doc.metadata.get("route_id")
                # Assume vector store returns a similarity score in document metadata (e.g., 'similarity_score')
                # If not, a default (e.g., `1.0 - distance`) would be needed depending on the vector store client.
                similarity = doc.metadata.get("similarity_score", 0.0)
                if route_id and route_id in self._routes:
                    route_confidences.setdefault(route_id, []).append(similarity)
                    matched_doc_ids_per_route.setdefault(route_id, []).append(doc.id)
                else:
                    logger.debug(f"Vector search result with unknown or invalid route_id '{route_id}' ignored (Doc ID: {doc.id}).")

            if not route_confidences:
                return RoutingDecision(route_id="none", confidence=0.0, explanation="No valid routes associated with vector search results.")

            # Choose the route with the highest confidence (e.g., the max similarity among its examples)
            max_overall_confidence = 0.0
            chosen_route_id = "none"
            chosen_explanation = ""
            chosen_matched_ids = []
            chosen_metadata = {}

            for route_id, confidences in route_confidences.items():
                max_route_confidence = max(confidences) # Take the highest similarity score for this route
                if max_route_confidence > max_overall_confidence:
                    max_overall_confidence = max_route_confidence
                    chosen_route_id = route_id
                    # Provide a simple explanation based on the vector match
                    chosen_explanation = (
                        f"Matched example prompt for route '{route_id}' with similarity {max_route_confidence:.2f}."
                    )
                    chosen_matched_ids = matched_doc_ids_per_route.get(route_id, [])
                    chosen_metadata = self._routes[route_id].metadata # Propagate original metadata

            # If the best match's confidence is below the threshold, treat as no match
            if chosen_route_id == "none" or max_overall_confidence < self._config.confidence_threshold:
                logger.debug(
                    f"Vector similarity for '{prompt[:50]}...' (route: {chosen_route_id}, conf: {max_overall_confidence:.2f}) "
                    f"is below the configured confidence threshold ({self._config.confidence_threshold:.2f}). "
                    "Treating as no valid match."
                )
                return RoutingDecision(
                    route_id="none",
                    confidence=0.0,
                    explanation=f"Similarity score below threshold ({self._config.confidence_threshold:.2f})."
                )

            logger.info(
                f"Vector similarity for '{prompt[:50]}...' resulted in: "
                f"Route ID: {chosen_route_id}, Confidence: {max_overall_confidence:.2f}"
            )
            return RoutingDecision(
                route_id=chosen_route_id,
                confidence=max_overall_confidence,
                explanation=chosen_explanation,
                matched_document_ids=chosen_matched_ids,
                metadata=chosen_metadata
            )

        except Exception as e:
            logger.error(
                f"Unhandled error during vector similarity routing for '{prompt[:50]}...': {e}", exc_info=True
            )
            return RoutingDecision(route_id="none", confidence=0.0, explanation=f"Vector store search error: {e}")

    async def route(self, prompt: str, context: Optional[Dict] = None) -> RoutingDecision:
        """
        The main public interface for the router. Routes an incoming prompt to an appropriate handler
        based on the configured strategy (LLM, Vector, or Hybrid).

        Args:
            prompt: The user's input prompt string.
            context: Optional dictionary of additional context for routing (e.g., user ID, session state).

        Returns:
            A `RoutingDecision` object containing the chosen route ID and related information.
        """
        await self._initialized_vector_store.wait() # Ensure vector store is fully ready before routing

        if not self._routes:
            logger.warning("No routes defined for SemanticPromptRouter. Checking for default fallback.")
            return self._handle_fallback(
                prompt=prompt,
                initial_decision=RoutingDecision(route_id="none", confidence=0.0, explanation="No routes defined for routing.")
            )

        decision: RoutingDecision = RoutingDecision(route_id="none", confidence=0.0)
        strategy_explanation = ""

        if self._config.strategy == RoutingStrategy.LLM_CLASSIFICATION:
            decision = await self._route_with_llm_classification(prompt)
            strategy_explanation = "LLM_CLASSIFICATION strategy"
        elif self._config.strategy == RoutingStrategy.VECTOR_SIMILARITY:
            decision = await self._route_with_vector_similarity(prompt)
            strategy_explanation = "VECTOR_SIMILARITY strategy"
        elif self._config.strategy == RoutingStrategy.HYBRID:
            # Execute both LLM and vector similarity routing concurrently for efficiency
            llm_task = self._route_with_llm_classification(prompt)
            vector_task = self._route_with_vector_similarity(prompt)
            llm_decision, vector_decision = await asyncio.gather(llm_task, vector_task)

            # Hybrid logic: Prioritize LLM's decision if it's confident and relevant,
            # otherwise defer to vector similarity if it's confident.
            # If both are confident, LLM is often preferred for nuanced understanding.
            # If neither is sufficiently confident, the one with slightly higher confidence might be chosen,
            # or it falls through to the general fallback mechanism.
            llm_confident = llm_decision.route_id != "none" and llm_decision.confidence >= self._config.confidence_threshold
            vector_confident = vector_decision.route_id != "none" and vector_decision.confidence >= self._config.confidence_threshold

            if llm_confident and vector_confident:
                # If both are confident, choose the one with higher confidence.
                decision = llm_decision if llm_decision.confidence >= vector_decision.confidence else vector_decision
                strategy_explanation = "HYBRID strategy: Both confident, chose the one with higher confidence."
            elif llm_confident:
                decision = llm_decision
                strategy_explanation = "HYBRID strategy: LLM was confident, vector was not."
            elif vector_confident:
                decision = vector_decision
                strategy_explanation = "HYBRID strategy: Vector similarity was confident, LLM was not."
            else:
                # Neither met individual confidence thresholds or suggested 'none'.
                # Pick the higher confidence one, even if below threshold, before attempting fallback.
                if llm_decision.confidence > vector_decision.confidence:
                    decision = llm_decision
                else:
                    decision = vector_decision
                strategy_explanation = "HYBRID strategy: Neither confident, picked the higher overall confidence but likely still needs fallback."
            logger.debug(
                f"Hybrid strategy raw results for '{prompt[:50]}...': "
                f"LLM(route={llm_decision.route_id}, conf={llm_decision.confidence:.2f}, exp='{llm_decision.explanation or ''}'), "
                f"Vector(route={vector_decision.route_id}, conf={vector_decision.confidence:.2f}, exp='{vector_decision.explanation or ''}')"
            )

        # After primary routing, apply fallback logic if no confident route was found
        if decision.route_id == "none" or decision.confidence < self._config.confidence_threshold:
            logger.info(
                f"Primary routing ({strategy_explanation}) for '{prompt[:50]}...' "
                f"did not yield a confident route (current decision: {decision.route_id}, conf: {decision.confidence:.2f}). "
                "Attempting fallback."
            )
            return self._handle_fallback(prompt, decision)
        else:
            logger.info(
                f"Routing for '{prompt[:50]}...' successfully decided on "
                f"Route ID: '{decision.route_id}', Confidence: {decision.confidence:.2f} ({strategy_explanation})"
            )

        # Attach original metadata from the chosen route definition if not already present or fully merged
        if decision.route_id in self._routes:
            chosen_route_def = self._routes[decision.route_id]
            # Merge original metadata, prioritizing metadata already in the decision (e.g., from vector search doc metadata)
            combined_metadata = {**chosen_route_def.metadata, **decision.metadata}
            decision.metadata = combined_metadata

        return decision

    def _handle_fallback(self, prompt: str, initial_decision: RoutingDecision) -> RoutingDecision:
        """
        Internal helper method to apply fallback logic when no confident route is found.
        If `default_route_id` is configured and valid, it will be used.
        """
        if self._config.default_route_id and self._config.default_route_id in self._routes:
            default_route_def = self._routes[self._config.default_route_id]
            logger.info(
                f"Falling back to default route '{self._config.default_route_id}' "
                f"for prompt '{prompt[:50]}...' (original routing explanation: {initial_decision.explanation or 'N/A'})."
            )
            return RoutingDecision(
                route_id=self._config.default_route_id,
                confidence=initial_decision.confidence, # Preserve the original (low) confidence
                explanation=(
                    initial_decision.explanation + " (Falling back to default route due to low confidence/no match)."
                    if initial_decision.explanation else f"No suitable route found; falling back to default route '{self._config.default_route_id}'."
                ),
                fallback_used=True,
                metadata=default_route_def.metadata
            )
        else:
            logger.warning(
                f"No confident route found for '{prompt[:50]}...' and no default fallback route configured or found. "
                "Returning 'none' route decision."
            )
            # Ensure the initial decision is explicitly set to 'none' if it wasn't already.
            initial_decision.route_id = "none"
            initial_decision.confidence = 0.0
            initial_decision.explanation = initial_decision.explanation or "No suitable route found and no default fallback route available."
            initial_decision.fallback_used = False # Explicitly not using fallback if none configured
            return initial_decision