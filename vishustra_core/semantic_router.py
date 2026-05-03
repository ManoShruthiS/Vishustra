import abc
import asyncio
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Awaitable

import numpy as np
from pydantic import BaseModel, Field

# Setup logging for the module to be easily integrated into Vishustra's logging system
logger = logging.getLogger(__name__)


# --- Protocols for External Dependencies ---

class EmbeddingProvider(Protocol):
    """
    Protocol for an embedding provider.
    Vishustra uses this to abstract different embedding model integrations.
    """
    async def get_embedding(self, text: str) -> List[float]:
        """Generates an embedding for a single text string."""
        ...

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a list of text strings."""
        ...

class LLMProvider(Protocol):
    """
    Protocol for an LLM provider.
    Vishustra uses this to abstract different LLM integrations for decision making.
    """
    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generates a response from the LLM based on the given prompt."""
        ...

# --- Pydantic Models for Router Outputs ---

class RouterResult(BaseModel):
    """
    Represents the outcome of a routing operation, including the chosen destination
    and associated confidence.
    """
    routed_destination_name: Optional[str] = Field(
        None, description="The name of the destination chosen by the router."
    )
    confidence_score: Optional[float] = Field(
        None, description="The confidence score associated with the chosen destination."
    )
    all_scores: Dict[str, float] = Field(
        {}, description="A dictionary of all destinations and their scores from the winning strategy."
    )
    strategy_name: Optional[str] = Field(
        None, description="The name of the routing strategy that made the decision."
    )
    reasoning: Optional[str] = Field(
        None, description="Optional reasoning provided for the routing decision, especially for fallback."
    )


# --- Abstract Base Classes for Modularity ---

class RouterDestination(abc.ABC):
    """
    Abstract Base Class for any component that the DynamicSemanticRouter can route to.
    This could be an LLM chain, an agent tool, a database query, a specific service, etc.
    """
    name: str
    description: str

    def __init__(self, name: str, description: str, **kwargs: Any):
        """
        Initializes a RouterDestination.

        Args:
            name: A unique identifier for this destination.
            description: A concise, human-readable description of what this
                         destination does. This is crucial for routing decisions.
            **kwargs: Additional arbitrary keyword arguments.
        """
        if not name or not isinstance(name, str):
            raise ValueError("Destination 'name' must be a non-empty string.")
        if not description or not isinstance(description, str):
            raise ValueError("Destination 'description' must be a non-empty string.")

        self.name = name
        self.description = description
        # Store other kwargs if needed, e.g., for serialization hints or additional metadata
        self._meta = kwargs

    @abc.abstractmethod
    async def invoke(self, inputs: Dict[str, Any]) -> Any:
        """
        Invokes the underlying component with the given inputs.
        This method defines how the destination actually executes its functionality.

        Args:
            inputs: A dictionary of inputs required by the destination.

        Returns:
            The result of invoking the destination.
        """
        raise NotImplementedError

    def __str__(self) -> str:
        return f"Destination('{self.name}', '{self.description[:50]}...')"

    def __repr__(self) -> str:
        return self.__str__()

    def to_dict(self) -> Dict[str, Any]:
        """Returns a dictionary representation of the destination."""
        return {"name": self.name, "description": self.description, **self._meta}


class RoutingStrategy(abc.ABC):
    """
    Abstract Base Class for different routing strategies.
    Each strategy provides a method to score potential destinations based on a query.
    """
    name: str

    def __init__(self, name: str, **kwargs: Any):
        """
        Initializes a RoutingStrategy.

        Args:
            name: A unique identifier for this routing strategy.
            **kwargs: Additional arbitrary keyword arguments.
        """
        if not name or not isinstance(name, str):
            raise ValueError("Strategy 'name' must be a non-empty string.")
        self.name = name
        self._meta = kwargs

    @abc.abstractmethod
    async def route(
        self,
        query: str,
        destinations: List[RouterDestination],
        embedding_provider: EmbeddingProvider,
        llm_provider: Optional[LLMProvider] = None,
        **kwargs: Any,
    ) -> List[Tuple[RouterDestination, float]]:
        """
        Routes a query to a list of destinations, returning a list of (destination, score) tuples.
        A higher score indicates a more suitable destination. The list should be sorted
        in descending order by score.

        Args:
            query: The user query string.
            destinations: A list of available RouterDestination objects.
            embedding_provider: An instance conforming to the EmbeddingProvider protocol.
                                May or may not be used by specific strategies.
            llm_provider: An instance conforming to the LLMProvider protocol.
                          May or may not be used by specific strategies.
            **kwargs: Additional arguments to pass to the routing logic.

        Returns:
            A list of tuples, where each tuple contains a RouterDestination and its
            associated confidence score, sorted by score in descending order.
        """
        raise NotImplementedError

    def __str__(self) -> str:
        return f"Strategy('{self.name}')"

    def __repr__(self) -> str:
        return self.__str__()


# --- Concrete Router Destination Implementations (Examples) ---

class ChainDestination(RouterDestination):
    """
    A concrete RouterDestination representing an LLM chain within Vishustra.
    """
    def __init__(
        self,
        name: str,
        description: str,
        chain_callable: Callable[[Dict[str, Any]], Awaitable[Any]],
    ):
        """
        Args:
            name: Unique name for the chain destination.
            description: Description of what the chain does.
            chain_callable: An awaitable callable that represents the LLM chain's execution.
                            It should accept a dictionary of inputs and return any result.
        """
        super().__init__(name, description)
        if not callable(chain_callable):
            raise TypeError("chain_callable must be a callable.")
        self._chain_callable = chain_callable
        if not asyncio.iscoroutinefunction(self._chain_callable):
            logger.warning(
                f"ChainDestination '{name}' was initialized with a non-async callable. "
                "It will be awaited, but direct async callables are preferred for consistency."
            )

    async def invoke(self, inputs: Dict[str, Any]) -> Any:
        """
        Invokes the underlying LLM chain with the given inputs.
        """
        logger.debug(f"Invoking ChainDestination '{self.name}' with inputs: {inputs}")
        try:
            return await self._chain_callable(inputs)
        except Exception as e:
            logger.error(f"Error invoking ChainDestination '{self.name}': {e}", exc_info=True)
            raise


class ToolDestination(RouterDestination):
    """
    A concrete RouterDestination representing an agent tool within Vishustra.
    """
    def __init__(
        self,
        name: str,
        description: str,
        tool_callable: Callable[[Dict[str, Any]], Awaitable[Any]],
    ):
        """
        Args:
            name: Unique name for the tool destination.
            description: Description of what the tool does.
            tool_callable: An awaitable callable that represents the agent tool's execution.
                           It should accept a dictionary of inputs and return any result.
        """
        super().__init__(name, description)
        if not callable(tool_callable):
            raise TypeError("tool_callable must be a callable.")
        self._tool_callable = tool_callable
        if not asyncio.iscoroutinefunction(self._tool_callable):
            logger.warning(
                f"ToolDestination '{name}' was initialized with a non-async callable. "
                "It will be awaited, but direct async callables are preferred for consistency."
            )

    async def invoke(self, inputs: Dict[str, Any]) -> Any:
        """
        Invokes the underlying agent tool with the given inputs.
        """
        logger.debug(f"Invoking ToolDestination '{self.name}' with inputs: {inputs}")
        try:
            return await self._tool_callable(inputs)
        except Exception as e:
            logger.error(f"Error invoking ToolDestination '{self.name}': {e}", exc_info=True)
            raise


# --- Concrete Routing Strategy Implementations ---

def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Computes the cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


class SemanticSimilarityStrategy(RoutingStrategy):
    """
    A routing strategy that uses semantic similarity between the query
    and destination descriptions to determine the best route.
    """
    def __init__(self, name: str = "semantic_similarity", threshold: float = 0.7):
        """
        Args:
            name: Name of the strategy.
            threshold: Minimum cosine similarity score required for a destination
                       to be considered a candidate.
        """
        super().__init__(name)
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("Threshold must be between 0.0 and 1.0")
        self.threshold = threshold

    async def route(
        self,
        query: str,
        destinations: List[RouterDestination],
        embedding_provider: EmbeddingProvider,
        llm_provider: Optional[LLMProvider] = None,  # Not used by this strategy
        **kwargs: Any,
    ) -> List[Tuple[RouterDestination, float]]:
        """
        Calculates cosine similarity between the query embedding and each destination's
        description embedding. Filters results by a confidence threshold.
        """
        if not destinations:
            return []

        if not embedding_provider:
            logger.error("EmbeddingProvider is required for SemanticSimilarityStrategy. Cannot route.")
            return []

        try:
            query_embedding_np = np.asarray(await embedding_provider.get_embedding(query))
        except Exception as e:
            logger.error(f"Failed to get embedding for query using '{self.name}' strategy: {e}")
            return []

        destination_descriptions = [d.description for d in destinations]
        try:
            dest_embeddings_nps = [
                np.asarray(emb) for emb in await embedding_provider.get_embeddings(destination_descriptions)
            ]
        except Exception as e:
            logger.error(f"Failed to get embeddings for destinations using '{self.name}' strategy: {e}")
            return []

        scores: List[Tuple[RouterDestination, float]] = []
        for i, dest in enumerate(destinations):
            similarity = _cosine_similarity(query_embedding_np, dest_embeddings_nps[i])
            if similarity >= self.threshold:
                scores.append((dest, similarity))
            else:
                logger.debug(
                    f"[{self.name}] Destination '{dest.name}' similarity '{similarity:.2f}' "
                    f"below threshold '{self.threshold:.2f}'."
                )

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


class LLMDecisionStrategy(RoutingStrategy):
    """
    A routing strategy that uses an LLM to decide the most suitable destination
    based on the query and destination descriptions.
    """
    DEFAULT_PROMPT_TEMPLATE = """
    You are an intelligent routing agent for a sophisticated AI framework.
    Your task is to analyze a user query and determine the SINGLE most suitable
    destination from a provided list of options. Prioritize accuracy and direct relevance.

    User Query: "{query}"

    Available Destinations (review their names and descriptions carefully):
    {destinations_info}

    Based on the User Query and the available destinations, please provide your
    decision in JSON format. The JSON must contain:
    - "destination_name": The EXACT name of the most suitable destination.
                          If no destination is suitable, use 'null'.
    - "confidence_score": A floating-point number between 0.0 (no confidence) and 1.0 (very confident)
                          indicating your certainty in the chosen destination.
    - "reasoning": A brief, clear explanation for your choice.

    Example of expected JSON format:
    {{
        "destination_name": "example_destination_name",
        "confidence_score": 0.95,
        "reasoning": "This destination best matches the intent of the query because it directly handles [specific keyword/concept]."
    }}

    Always output valid JSON.
    """

    def __init__(self, name: str = "llm_decision", prompt_template: Optional[str] = None):
        """
        Args:
            name: Name of the strategy.
            prompt_template: Custom prompt template for the LLM. If None, uses a default.
        """
        super().__init__(name)
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE

    async def route(
        self,
        query: str,
        destinations: List[RouterDestination],
        embedding_provider: EmbeddingProvider,  # Not used by this strategy
        llm_provider: Optional[LLMProvider] = None,
        **kwargs: Any,
    ) -> List[Tuple[RouterDestination, float]]:
        """
        Uses an LLM to analyze the query and destination descriptions to
        determine the most suitable route.
        """
        if not destinations:
            return []

        if not llm_provider:
            logger.error("LLMProvider is required for LLMDecisionStrategy. Cannot route.")
            return []

        destinations_info = "\n".join(
            [f"- Name: {d.name}\n  Description: {d.description}" for d in destinations]
        )

        prompt = self.prompt_template.format(
            query=query, destinations_info=destinations_info
        )

        try:
            llm_response_str = await llm_provider.generate(prompt)
            # Robust parsing: find the first and last curly brace to extract JSON
            start_idx = llm_response_str.find('{')
            end_idx = llm_response_str.rfind('}') + 1

            if start_idx == -1 or end_idx == 0 or start_idx >= end_idx:
                logger.error(f"[{self.name}] LLM response did not contain valid JSON: {llm_response_str}")
                return []

            llm_response = json.loads(llm_response_str[start_idx:end_idx])

            chosen_name = llm_response.get("destination_name")
            confidence = float(llm_response.get("confidence_score", 0.0))
            reasoning = llm_response.get("reasoning", "")

            if chosen_name:
                chosen_dest = next((d for d in destinations if d.name == chosen_name), None)
                if chosen_dest:
                    logger.info(
                        f"[{self.name}] LLM chose '{chosen_name}' with confidence {confidence:.2f}. "
                        f"Reasoning: {reasoning}"
                    )
                    # For RouterResult.reasoning, we would need to pass this up.
                    # Currently, the tuple only holds dest and score. This is a design trade-off
                    # for simplicity. If detailed reasoning per strategy is needed in result,
                    # the return type would need to be a custom object.
                    return [(chosen_dest, confidence)]
                else:
                    logger.warning(
                        f"[{self.name}] LLM chose unknown destination '{chosen_name}'. "
                        f"Query: '{query}'. LLM Response: {llm_response_str}"
                    )
            else:
                logger.info(f"[{self.name}] LLM Strategy indicated no suitable destination for query: '{query}'.")
            return []

        except json.JSONDecodeError as e:
            logger.error(f"[{self.name}] Failed to parse LLM response JSON: {e}. Response: {llm_response_str}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"[{self.name}] Error during LLM decision routing: {e}", exc_info=True)
            return []


# --- Main Dynamic Semantic Router ---

class DynamicSemanticRouter:
    """
    The core dynamic semantic router for Vishustra.
    It orchestrates multiple routing strategies to intelligently dispatch
    user queries to the most appropriate backend components (chains, tools, etc.).

    This router supports pluggable destinations and routing strategies, enabling
    highly flexible and intelligent request handling based on semantic understanding
    and configurable decision logic.
    """
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        llm_provider: Optional[LLMProvider] = None,
        default_strategy_name: Optional[str] = None,
        fallback_destination: Optional[RouterDestination] = None,
        min_confidence_for_route: float = 0.5,
    ):
        """
        Initializes the DynamicSemanticRouter.

        Args:
            embedding_provider: An instance conforming to the EmbeddingProvider protocol.
                                Required for semantic strategies like `SemanticSimilarityStrategy`.
            llm_provider: An optional instance conforming to the LLMProvider protocol.
                          Required for LLM-based strategies like `LLMDecisionStrategy`.
            default_strategy_name: The name of a specific strategy to prioritize if multiple
                                   strategies yield results. If this strategy provides a valid
                                   route above `min_confidence_for_route`, its result will be
                                   chosen over potentially higher-scoring results from other strategies.
            fallback_destination: An optional RouterDestination to use if no suitable
                                  route is found by any strategy above `min_confidence_for_route`.
            min_confidence_for_route: The minimum confidence score required for a destination
                                      to be considered valid by the router. Routes below this
                                      threshold will be ignored, potentially leading to a fallback.
        """
        if not isinstance(embedding_provider, EmbeddingProvider):
            raise TypeError("embedding_provider must conform to the EmbeddingProvider protocol.")
        if llm_provider is not None and not isinstance(llm_provider, LLMProvider):
             raise TypeError("llm_provider must conform to the LLMProvider protocol or be None.")
        if not (0.0 <= min_confidence_for_route <= 1.0):
            raise ValueError("min_confidence_for_route must be between 0.0 and 1.0.")

        self._destinations: Dict[str, RouterDestination] = {}
        self._strategies: Dict[str, RoutingStrategy] = {}
        self._embedding_provider = embedding_provider
        self._llm_provider = llm_provider
        self._default_strategy_name = default_strategy_name
        self._fallback_destination = fallback_destination
        self._min_confidence_for_route = min_confidence_for_route

        # Add default strategies, which can be overridden or removed later
        self.add_strategy(SemanticSimilarityStrategy())
        if self._llm_provider:
            self.add_strategy(LLMDecisionStrategy())
        else:
            logger.warning(
                "LLMProvider not configured. LLM-based routing strategies "
                "(e.g., LLMDecisionStrategy) will not be active."
            )

        if self._default_strategy_name and self._default_strategy_name not in self._strategies:
            logger.warning(
                f"Default strategy '{self._default_strategy_name}' was specified but not added. "
                "It will be ignored."
            )
            self._default_strategy_name = None

    def add_destination(self, destination: RouterDestination) -> None:
        """Adds a new destination that the router can dispatch to."""
        if not isinstance(destination, RouterDestination):
            raise TypeError("Provided object is not an instance of RouterDestination.")
        if destination.name in self._destinations:
            logger.warning(f"Destination with name '{destination.name}' already exists. Overwriting.")
        self._destinations[destination.name] = destination
        logger.info(f"Added destination: {destination.name}")

    def add_strategy(self, strategy: RoutingStrategy) -> None:
        """Adds a new routing strategy to be used by the router."""
        if not isinstance(strategy, RoutingStrategy):
            raise TypeError("Provided object is not an instance of RoutingStrategy.")
        if strategy.name in self._strategies:
            logger.warning(f"Strategy with name '{strategy.name}' already exists. Overwriting.")
        self._strategies[strategy.name] = strategy
        logger.info(f"Added routing strategy: {strategy.name}")

    def remove_destination(self, name: str) -> None:
        """Removes a destination by its name."""
        if name in self._destinations:
            del self._destinations[name]
            logger.info(f"Removed destination: {name}")
        else:
            logger.warning(f"Destination '{name}' not found, cannot remove.")

    def remove_strategy(self, name: str) -> None:
        """Removes a routing strategy by its name."""
        if name in self._strategies:
            del self._strategies[name]
            logger.info(f"Removed strategy: {name}")
            if self._default_strategy_name == name:
                self._default_strategy_name = None
                logger.warning(f"Removed default strategy '{name}'. Default strategy unset.")
        else:
            logger.warning(f"Strategy '{name}' not found, cannot remove.")

    def get_destination(self, name: str) -> Optional[RouterDestination]:
        """Retrieves a destination object by its name."""
        return self._destinations.get(name)

    async def route(self, query: str, **kwargs: Any) -> RouterResult:
        """
        Routes the given query to the most appropriate destination using
        the configured strategies.

        Args:
            query: The user query string to be routed.
            **kwargs: Additional arguments to pass to the routing strategies.

        Returns:
            A RouterResult object containing the chosen destination's name,
            confidence score, and scores from the winning strategy.
        """
        if not self._destinations:
            logger.warning("No destinations registered with the router.")
            return self._handle_fallback(
                strategy_name="no_destinations",
                reasoning="No destinations registered with the router."
            )

        active_strategies: Dict[str, RoutingStrategy] = {}
        for strategy_name, strategy in self._strategies.items():
            # Filter out strategies that cannot run due to missing dependencies
            if isinstance(strategy, LLMDecisionStrategy) and not self._llm_provider:
                logger.debug(f"Skipping LLMDecisionStrategy '{strategy_name}' as no LLM provider is configured.")
                continue
            active_strategies[strategy_name] = strategy

        if not active_strategies:
            logger.warning("No active routing strategies after filtering.")
            return self._handle_fallback(
                strategy_name="no_active_strategies",
                reasoning="No active routing strategies after filtering by dependencies."
            )

        all_strategy_results: Dict[str, List[Tuple[RouterDestination, float]]] = {}
        tasks = []
        for strategy_name, strategy in active_strategies.items():
            tasks.append(
                self._run_strategy_and_store_results(
                    strategy=strategy,
                    query=query,
                    embedding_provider=self._embedding_provider,
                    llm_provider=self._llm_provider,
                    results_dict=all_strategy_results,
                    **kwargs,
                )
            )
        await asyncio.gather(*tasks)

        best_overall_score = -1.0
        best_overall_destination: Optional[RouterDestination] = None
        winning_strategy_name: Optional[str] = None
        all_destination_scores: Dict[str, float] = {}

        # 1. Prioritize default strategy if specified and it returned valid results
        if self._default_strategy_name and self._default_strategy_name in all_strategy_results:
            default_results = all_strategy_results[self._default_strategy_name]
            if default_results:
                top_default_dest, top_default_score = default_results[0]
                if top_default_score >= self._min_confidence_for_route:
                    best_overall_destination = top_default_dest
                    best_overall_score = top_default_score
                    winning_strategy_name = self._default_strategy_name
                    all_destination_scores = {dest.name: score for dest, score in default_results}
                    logger.debug(
                        f"Default strategy '{winning_strategy_name}' selected '{top_default_dest.name}' "
                        f"with score {top_default_score:.2f}."
                    )

        # 2. If no default strategy or it didn't yield a high-confidence result,
        #    then iterate through all results to find the highest score across all strategies.
        if not winning_strategy_name:
            for strategy_name, results in all_strategy_results.items():
                if results:
                    current_best_dest, current_best_score = results[0]
                    if current_best_score > best_overall_score:
                        best_overall_score = current_best_score
                        best_overall_destination = current_best_dest
                        winning_strategy_name = strategy_name
                        all_destination_scores = {dest.name: score for dest, score in results}
            if best_overall_destination:
                logger.debug(
                    f"Highest score from non-default strategies: '{best_overall_destination.name}' "
                    f"with score {best_overall_score:.2f} by '{winning_strategy_name}'."
                )


        final_routed_destination: Optional[RouterDestination] = None
        final_confidence: Optional[float] = None
        final_reasoning: Optional[str] = None

        if best_overall_destination and best_overall_score >= self._min_confidence_for_route:
            final_routed_destination = best_overall_destination
            final_confidence = best_overall_score
            # Reasoning from strategies (e.g., LLM) would be captured here if the strategy
            # return type included it. For now, it's a simplification.
        else:
            logger.info(
                f"No strategy yielded a route above min confidence {self._min_confidence_for_route:.2f}. "
                f"Highest score was {best_overall_score:.2f} for '{best_overall_destination.name if best_overall_destination else 'N/A'}'."
            )

        # 3. Fallback mechanism
        if not final_routed_destination:
            return self._handle_fallback(
                strategy_name="no_high_confidence_route",
                reasoning="No high-confidence route found by any strategy, falling back."
            )

        # Construct the RouterResult
        result = RouterResult(
            routed_destination_name=final_routed_destination.name,
            confidence_score=final_confidence,
            all_scores=all_destination_scores,
            strategy_name=winning_strategy_name,
            reasoning=final_reasoning,
        )

        logger.info(
            f"Router successfully selected destination '{result.routed_destination_name}' "
            f"with confidence {result.confidence_score:.2f} "
            f"using strategy '{result.strategy_name}' for query: '{query[:50]}...'."
        )
        return result

    async def _run_strategy_and_store_results(
        self,
        strategy: RoutingStrategy,
        query: str,
        embedding_provider: EmbeddingProvider,
        llm_provider: Optional[LLMProvider],
        results_dict: Dict[str, List[Tuple[RouterDestination, float]]],
        **kwargs: Any,
    ) -> None:
        """Helper to run a strategy and safely store its results, handling exceptions."""
        try:
            logger.debug(f"Running strategy '{strategy.name}' for query: '{query[:50]}...'")
            strategy_results = await strategy.route(
                query=query,
                destinations=list(self._destinations.values()),
                embedding_provider=embedding_provider,
                llm_provider=llm_provider,
                **kwargs,
            )
            results_dict[strategy.name] = strategy_results
            logger.debug(f"Strategy '{strategy.name}' returned {len(strategy_results)} results.")
        except Exception as e:
            logger.error(f"Error running strategy '{strategy.name}': {e}", exc_info=True)
            results_dict[strategy.name] = []  # Ensure it's always in the dict, even if empty

    def _handle_fallback(self, strategy_name: str, reasoning: str) -> RouterResult:
        """Handles the fallback mechanism and returns a RouterResult."""
        if self._fallback_destination:
            logger.warning(
                f"Falling back to destination '{self._fallback_destination.name}'. "
                f"Reason: {reasoning}"
            )
            return RouterResult(
                routed_destination_name=self._fallback_destination.name,
                confidence_score=0.0,  # Explicitly 0.0 for fallback routes
                strategy_name=strategy_name,
                reasoning=reasoning,
            )
        else:
            logger.warning(f"No suitable route found and no fallback destination configured. Reason: {reasoning}")
            return RouterResult(
                routed_destination_name=None,
                confidence_score=None,
                strategy_name=strategy_name,
                reasoning=reasoning,
            )