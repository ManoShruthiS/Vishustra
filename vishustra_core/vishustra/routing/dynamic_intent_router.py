from __future__ import annotations
import abc
import asyncio
import logging
from typing import List, Optional, Type, Dict, Any, AsyncIterator

from pydantic import BaseModel, Field, PrivateAttr

# --- Start Mocking External Vishustra Dependencies ---
# In a real Vishustra environment, these would be imported from core modules.
# We include them here for self-containment and clarity of the router's dependencies.

class LLMResponse(BaseModel):
    """Represents a standardized response from an LLM."""
    text: str = Field(..., description="The main text output from the LLM.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata from the LLM provider, e.g., token usage, model ID.")

class BaseLLMProvider(abc.ABC):
    """
    Abstract base class defining the interface for all Large Language Model providers
    within Vishustra.
    """
    @abc.abstractmethod
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Asynchronously generates a single text response from the LLM based on the given prompt.

        Args:
            prompt: The input prompt string for the LLM.
            **kwargs: Additional parameters specific to the LLM provider (e.g., temperature, max_tokens).

        Returns:
            An LLMResponse object containing the generated text and any associated metadata.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def stream_generate(self, prompt: str, **kwargs) -> AsyncIterator[LLMResponse]:
        """
        Asynchronously streams text generation from the LLM based on the given prompt.

        Args:
            prompt: The input prompt string for the LLM.
            **kwargs: Additional parameters specific to the LLM provider (e.g., temperature, max_tokens).

        Yields:
            LLMResponse objects as parts of the text are generated.
        """
        raise NotImplementedError

class VishustraError(Exception):
    """Base exception class for all custom errors in Vishustra."""
    pass

class RouteNotFound(VishustraError):
    """
    Exception raised when the `DynamicIntentRouter` cannot determine a suitable route
    for a query, and no default route is available.
    """
    pass

class LLMClassificationError(VishustraError):
    """
    Exception raised specifically when an LLM-based routing strategy fails to
    classify a query into a valid route after multiple attempts.
    """
    pass

class PromptTemplate(BaseModel):
    """
    A simple prompt templating class for constructing LLM prompts dynamically.
    Uses Python's f-string like formatting via `.format()`.
    """
    template: str = Field(..., description="The template string with placeholders (e.g., '{variable_name}').")

    def format(self, **kwargs) -> str:
        """
        Formats the template string with the provided keyword arguments.

        Args:
            **kwargs: Key-value pairs to substitute into the template.

        Returns:
            The formatted string.
        """
        return self.template.format(**kwargs)

# --- End Mocking External Vishustra Dependencies ---

_LOGGER = logging.getLogger(__name__)

class Route(BaseModel):
    """
    Represents a single routing destination within Vishustra's orchestration framework.

    Each route has a unique name, a detailed natural language description for semantic
    matching, and can optionally include keywords for simpler, direct matching.
    It can also carry arbitrary metadata relevant to the downstream component.
    """
    name: str = Field(..., description="A unique identifier for the route (e.g., 'CheckOrderStatus', 'TechnicalSupport').")
    description: str = Field(..., description="A detailed natural language description of what this route handles. This is crucial for LLM-based classification and clarity.")
    keywords: List[str] = Field(default_factory=list, description="Optional keywords associated with the route for direct string matching or enhanced relevance scoring.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata associated with this route, e.g., a reference to a handler function, component ID, or specific configuration.")

    def __hash__(self):
        """Allows Route objects to be used in sets or as dictionary keys based on their name."""
        return hash(self.name)

    def __eq__(self, other):
        """Defines equality for Route objects based on their name."""
        if not isinstance(other, Route):
            return NotImplemented
        return self.name == other.name

class RoutingStrategy(abc.ABC, BaseModel):
    """
    Abstract base class for all routing strategies in Vishustra.

    Routing strategies encapsulate the logic for how an input query is mapped
    to a specific `Route` from a collection of available routes.
    """
    strategy_name: str = "base_strategy"
    _config: Dict[str, Any] = PrivateAttr(default_factory=dict) # Internal config for strategy-specific settings

    @abc.abstractmethod
    async def determine_route(self, query: str, available_routes: List[Route]) -> Optional[Route]:
        """
        Asynchronously determines the most suitable route for a given query
        from a list of available routes.

        Args:
            query: The input query string to be routed.
            available_routes: A list of `Route` objects to choose from.

        Returns:
            The chosen `Route` object if a match is found, otherwise `None`.
        """
        raise NotImplementedError

class LLMClassificationStrategy(RoutingStrategy):
    """
    An advanced routing strategy that leverages a Large Language Model (LLM)
    to semantically classify an input query against the natural language
    descriptions of available routes.

    This strategy constructs a prompt asking the LLM to identify the best route
    or indicate no match, based on the provided route descriptions. It includes
    retry logic for robust operation against LLM API flakes or invalid responses.
    """
    strategy_name: str = "llm_classification"
    llm: BaseLLMProvider = Field(..., description="The LLM provider instance to use for classification.")
    classification_prompt_template: PromptTemplate = Field(
        default=PromptTemplate(
            template=(
                "You are an expert routing system for a large language model orchestration framework. "
                "Your task is to analyze a user's query and determine which of the following available "
                "routes is most relevant. Respond ONLY with the exact 'name' of the chosen route. "
                "If no route is suitable, or the query is ambiguous, respond ONLY with '{no_match_id}'.\n\n"
                "Available Routes:\n"
                "{route_options}\n\n"
                "User Query: {query}\n\n"
                "Chosen Route:"
            )
        ),
        description="The prompt template used to instruct the LLM for classification. It must contain `{route_options}`, `{query}`, and `{no_match_id}` placeholders."
    )
    no_match_identifier: str = Field("NONE", description="The specific string the LLM should return if it determines no route matches the query. This must match the `{no_match_id}` in the prompt template.")
    max_retries: int = Field(3, ge=0, description="Maximum number of retries if the LLM response is invalid or an error occurs during classification.")
    retry_delay_sec: float = Field(1.0, ge=0.0, description="Delay in seconds between retries to prevent overwhelming the LLM API.")

    # Private attributes for internal caching of routes, not part of serialized model
    _route_map: Dict[str, Route] = PrivateAttr(default_factory=dict)

    def __init__(self, **data: Any):
        super().__init__(**data)
        # Ensure the prompt template can be formatted with no_match_identifier
        self.classification_prompt_template.template = self.classification_prompt_template.template.format(
            no_match_id=self.no_match_identifier,
            route_options="{route_options}", # Re-add placeholders for later dynamic formatting
            query="{query}"
        )

    @staticmethod
    def _format_route_options(routes: List[Route]) -> str:
        """
        Helper method to format the descriptions of available routes into a concise
        string suitable for inclusion in an LLM prompt.
        """
        options_str = ""
        for i, route in enumerate(routes):
            options_str += f"{i+1}. Name: '{route.name}'\n   Description: '{route.description}'\n"
        return options_str.strip()

    async def determine_route(self, query: str, available_routes: List[Route]) -> Optional[Route]:
        """
        Determines the most relevant route for the given query using LLM-based classification.

        The method constructs a prompt, sends it to the configured LLM, and parses the
        LLM's response to identify the chosen route. It includes retry logic for robustness.

        Args:
            query: The user's input query string.
            available_routes: A list of `Route` objects to choose from.

        Returns:
            The most relevant `Route` object if successfully classified, otherwise `None`.

        Raises:
            LLMClassificationError: If the LLM consistently fails to provide a valid route name
                                    or an unrecoverable error occurs after `max_retries`.
        """
        if not available_routes:
            _LOGGER.warning("LLMClassificationStrategy received no available routes to classify against. Returning None.")
            return None

        self._route_map = {route.name: route for route in available_routes}
        route_options_str = self._format_route_options(available_routes)

        for attempt in range(self.max_retries + 1):
            try:
                # Format the prompt dynamically with current route options and query
                prompt = self.classification_prompt_template.format(
                    route_options=route_options_str,
                    query=query
                )
                _LOGGER.debug(f"Attempt {attempt+1}: Sending classification prompt to LLM for query '{query}':\n{prompt}")

                # Use a low temperature for classification tasks to encourage deterministic output
                llm_response = await self.llm.generate(prompt=prompt, temperature=0.0)
                predicted_route_name = llm_response.text.strip().strip("'\"") # Clean potential quotes from LLM output

                _LOGGER.debug(f"LLM predicted route name for query '{query}': '{predicted_route_name}' (LLM response: '{llm_response.text}')")

                if predicted_route_name == self.no_match_identifier:
                    _LOGGER.info(f"LLM indicated no suitable route for query: '{query}'.")
                    return None
                elif predicted_route_name in self._route_map:
                    _LOGGER.info(f"LLM successfully classified query '{query}' to route '{predicted_route_name}'.")
                    return self._route_map[predicted_route_name]
                else:
                    _LOGGER.warning(
                        f"LLM returned an invalid route name '{predicted_route_name}' for query '{query}'. "
                        f"Expected one of: {list(self._route_map.keys())} or '{self.no_match_identifier}'."
                    )
            except Exception as e:
                _LOGGER.error(f"Error during LLM classification attempt {attempt+1} for query '{query}': {e}", exc_info=True)

            if attempt < self.max_retries:
                _LOGGER.info(f"Retrying LLM classification for query '{query}' in {self.retry_delay_sec} seconds...")
                await asyncio.sleep(self.retry_delay_sec)
            else:
                _LOGGER.error(f"Max retries ({self.max_retries}) reached for LLM classification for query '{query}'. "
                              "Raising LLMClassificationError.")
                raise LLMClassificationError(
                    f"Failed to classify query '{query}' after {self.max_retries} retries using LLM strategy."
                )
        return None # Should be unreachable due to the raise, but for type safety.


class KeywordMatchingStrategy(RoutingStrategy):
    """
    A simpler routing strategy that attempts to match keywords in the input query
    against the `keywords` attribute defined for each available route.

    This strategy prioritizes routes with a higher count of matching keywords and
    can be configured for case sensitivity and a minimum match threshold. It serves
    as a fast, deterministic fallback or primary strategy for well-defined routes.
    """
    strategy_name: str = "keyword_matching"
    case_sensitive: bool = Field(False, description="Whether keyword matching should be case-sensitive.")
    min_match_count: int = Field(1, ge=1, description="Minimum number of keyword matches required to consider a route as a valid candidate.")

    async def determine_route(self, query: str, available_routes: List[Route]) -> Optional[Route]:
        """
        Determines the route based on keyword matching between the query and route keywords.

        Args:
            query: The user's input query string.
            available_routes: A list of `Route` objects to choose from.

        Returns:
            The `Route` object with the most keyword matches that meet the `min_match_count`,
            or `None` if no route meets the criteria.
        """
        if not available_routes:
            _LOGGER.warning("KeywordMatchingStrategy received no available routes to match against. Returning None.")
            return None

        # Normalize query tokens based on case_sensitive setting
        query_tokens = set(query.split())
        if not self.case_sensitive:
            query_tokens = {token.lower() for token in query_tokens}

        best_match_route: Optional[Route] = None
        max_matches = 0

        for route in available_routes:
            # Normalize route keywords based on case_sensitive setting
            route_keywords = set(route.keywords)
            if not self.case_sensitive:
                route_keywords = {k.lower() for k in route_keywords}

            common_keywords = query_tokens.intersection(route_keywords)
            match_count = len(common_keywords)

            if match_count >= self.min_match_count and match_count > max_matches:
                max_matches = match_count
                best_match_route = route
                _LOGGER.debug(f"Route '{route.name}' matched with {match_count} keywords: {common_keywords}")

        if best_match_route:
            _LOGGER.info(f"KeywordMatchingStrategy routed query '{query}' to '{best_match_route.name}' with {max_matches} matches.")
        else:
            _LOGGER.info(f"KeywordMatchingStrategy found no suitable route for query '{query}' (min_match_count={self.min_match_count}).")

        return best_match_route


class DynamicIntentRouter(BaseModel):
    """
    An elite dynamic intent router for Vishustra, designed to intelligently steer
    user queries to the most appropriate workflow, agent, or component based on their intent.

    This router is highly modular, supporting multiple pluggable routing strategies
    (e.g., LLM-powered classification, keyword matching) and robust fallback mechanisms.
    """
    routes: List[Route] = Field(..., description="A list of all possible routes the router can direct to.")
    strategy: RoutingStrategy = Field(..., description="The primary strategy to use for determining the route (e.g., LLMClassificationStrategy, KeywordMatchingStrategy).")
    default_route_name: Optional[str] = Field(None, description="The name of a fallback route to use if the primary strategy fails to determine a route. This route must exist in `routes`.")

    _route_map: Dict[str, Route] = PrivateAttr()

    def __init__(self, **data: Any):
        """
        Initializes the DynamicIntentRouter, building an internal map of routes
        and validating the existence of the default route if specified.
        """
        super().__init__(**data)
        self._route_map = {route.name: route for route in self.routes}
        if self.default_route_name and self.default_route_name not in self._route_map:
            raise ValueError(f"Configured default route '{self.default_route_name}' not found in the provided list of routes.")
        _LOGGER.info(f"DynamicIntentRouter initialized with {len(self.routes)} routes and strategy: {self.strategy.strategy_name}")
        if self.default_route_name:
            _LOGGER.info(f"Default route configured: '{self.default_route_name}'")

    async def route(self, query: str) -> Route:
        """
        Determines the best route for a given query using the configured primary strategy.

        If the primary strategy fails to find a route (returns `None` or raises an exception),
        and a `default_route_name` is provided, it will attempt to fall back to the default route.

        Args:
            query: The input query string from the user or another component that needs routing.

        Returns:
            The determined `Route` object, representing the intended destination.

        Raises:
            RouteNotFound: If no suitable route can be determined by the primary strategy
                           and no default route is configured or found.
            VishustraError: Propagates any underlying errors from the routing strategy
                            if not handled by fallback logic.
        """
        _LOGGER.info(f"Attempting to route query: '{query}' using strategy '{self.strategy.strategy_name}'.")

        chosen_route: Optional[Route] = None
        try:
            chosen_route = await self.strategy.determine_route(query, self.routes)
        except VishustraError as e:
            _LOGGER.warning(f"Primary routing strategy '{self.strategy.strategy_name}' failed for query '{query}': {e}")
            # Do not re-raise immediately; attempt fallback
        except Exception as e:
            _LOGGER.error(f"Unexpected error during primary routing strategy '{self.strategy.strategy_name}' for query '{query}': {e}", exc_info=True)
            # Attempt fallback

        if chosen_route:
            _LOGGER.info(f"Query '{query}' successfully routed to '{chosen_route.name}' via primary strategy.")
            return chosen_route
        elif self.default_route_name:
            _LOGGER.warning(f"Primary strategy failed for query '{query}'. Falling back to default route '{self.default_route_name}'.")
            default_route = self._route_map.get(self.default_route_name)
            if default_route:
                _LOGGER.info(f"Query '{query}' routed to default route '{default_route.name}'.")
                return default_route
            else:
                # This case should ideally be caught during initialization, but provides robust runtime check.
                _LOGGER.critical(f"Configured default route '{self.default_route_name}' was not found in the route list during runtime. This indicates a configuration error.")
                raise RouteNotFound(f"No route determined for query '{query}' and default route '{self.default_route_name}' is invalid/missing.")
        else:
            _LOGGER.error(f"No route determined for query '{query}' by primary strategy, and no default route is configured. Raising RouteNotFound.")
            raise RouteNotFound(f"No suitable route determined for query '{query}' and no default route specified.")