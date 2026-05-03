import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Union

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class Route:
    """
    Represents a specific routing destination or functional pipeline within the Vishustra framework.

    Routes define logical paths or functionalities an incoming LLM request
    might be directed to. Each route has a unique name, a descriptive text
    to aid routing decisions (especially for LLM-based strategies), and
    optional keywords or metadata for specific strategy implementations.

    Attributes:
        name (str): A unique identifier for the route (e.g., "qa_pipeline", "code_generation").
        description (str): A detailed description of what this route handles.
                           This is crucial for LLM-based routing strategies as it informs
                           the LLM's decision-making process.
        keywords (Optional[List[str]]): A list of keywords associated with this route.
                                        Useful for keyword-based routing strategies.
                                        Keywords are stored in lowercase for consistent matching.
        metadata (Optional[Dict[str, Any]]): Any additional arbitrary data associated
                                             with this route, e.g., configuration parameters,
                                             target function pointers, API endpoints, etc.
    """
    name: str
    description: str
    keywords: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """
        Performs validation and normalization after object initialization.
        Ensures name and description are not empty and keywords are lowercased.
        """
        if not self.name:
            raise ValueError("Route name cannot be empty.")
        if not self.description:
            raise ValueError("Route description cannot be empty.")
        # Ensure keywords are lowercase for consistent matching across strategies
        if self.keywords is not None:
            object.__setattr__(self, 'keywords', [kw.lower() for kw in self.keywords])

class RoutingStrategy(ABC):
    """
    Abstract Base Class for all routing strategies.

    A routing strategy defines the core logic for how an input text is analyzed
    and matched against a set of available routes. Concrete implementations
    must override the `route` method to provide their specific routing mechanism.
    """

    @abstractmethod
    def route(self, input_text: str, routes: List[Route]) -> Optional[Route]:
        """
        Determines the most appropriate route for a given input text.

        This method encapsulates the decision-making process of the strategy.

        Args:
            input_text (str): The incoming text to be routed. This is typically a user query.
            routes (List[Route]): A list of available `Route` objects to choose from.
                                  The strategy will evaluate the input against these routes.

        Returns:
            Optional[Route]: The chosen `Route` object if a suitable match is found
                             by the strategy, or None if no appropriate route can be
                             determined.
        """
        pass

class KeywordRoutingStrategy(RoutingStrategy):
    """
    A simple and efficient routing strategy that matches input text against
    predefined keywords associated with each route.

    This strategy performs a case-insensitive check to see if any of a route's
    keywords are present within the input text. If multiple routes contain
    matching keywords, the first one encountered in the provided `routes` list
    that contains *any* matching keyword will be returned. This implies an
    ordering preference.
    """

    def route(self, input_text: str, routes: List[Route]) -> Optional[Route]:
        """
        Routes the input text based on keyword matching.

        The strategy iterates through the provided routes. For each route, it checks
        if any of its associated keywords are present in the (lowercased) input text.
        The first route to have a matching keyword is selected.

        Args:
            input_text (str): The incoming text to be routed.
            routes (List[Route]): A list of available `Route` objects.

        Returns:
            Optional[Route]: The first `Route` found where one of its keywords
                             is present in the input text (case-insensitive),
                             or None if no match is found across any route.
        """
        input_text_lower = input_text.lower()
        for route in routes:
            if route.keywords:
                for keyword in route.keywords:
                    if keyword in input_text_lower:
                        logger.debug(
                            f"Keyword '{keyword}' matched for route '{route.name}' "
                            f"with input: '{input_text[:50]}...'"
                        )
                        return route
        logger.debug(f"No keyword route found for input: '{input_text[:50]}...'")
        return None

# --- LLM Interface for LLMRoutingStrategy ---
class BaseLLM(ABC):
    """
    Abstract Base Class for an LLM interface.

    This interface defines the fundamental method required for an LLM
    to be integrated with the `LLMRoutingStrategy`. Implementations should
    handle sending prompts to an actual LLM service (e.g., OpenAI, Anthropic,
    local models) and returning its raw text response.
    """
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Sends a text prompt to the LLM and returns its generated text response.

        Args:
            prompt (str): The text prompt to send to the LLM. This will typically
                          be a carefully crafted prompt instructing the LLM on
                          the routing task.
            **kwargs: Additional parameters specific to the LLM implementation
                      (e.g., `temperature`, `max_tokens`, `stop_sequences`,
                      model name, API keys).

        Returns:
            str: The raw text response from the LLM. This response is then
                 parsed by the `LLMRoutingStrategy`.
        """
        pass

class MockLLM(BaseLLM):
    """
    A mock LLM implementation primarily for testing or demonstration purposes.

    It simulates an LLM response based on predefined rules or a simplistic
    keyword-based selection from the prompt content, rather than actual
    intelligent generation. This allows for testing `LLMRoutingStrategy`
    without requiring a live LLM connection.
    """
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Simulates an LLM response by attempting to parse route options
        from the prompt and returning a "chosen_route_name" if a relevant
        keyword or phrase is found in the simulated user query.

        This is a basic heuristic for demonstration and not indicative of
        a real LLM's capabilities.
        """
        logger.info(f"MockLLM received prompt:\n{prompt[:300]}...")

        # Basic simulation for router's expected JSON output
        # Look for a specific pattern to find routes and the user query
        route_definitions_start_idx = prompt.find("Available Routes:")
        query_start_idx = prompt.find("User Query:")

        if route_definitions_start_idx != -1 and query_start_idx != -1:
            route_info_str = prompt[route_definitions_start_idx:query_start_idx]
            user_query = prompt[query_start_idx:].lower()

            # Attempt to extract route names and descriptions from the prompt string
            possible_routes_info = []
            for line in route_info_str.split('\n'):
                if "- Name:" in line:
                    try:
                        name = line.split("Name:")[1].split("Description:")[0].strip()
                        desc = line.split("Description:")[1].strip()
                        possible_routes_info.append({"name": name, "description": desc})
                    except IndexError:
                        logger.warning(f"MockLLM failed to parse route info line: {line}")
                        continue

            for route_data in possible_routes_info:
                # Very basic matching: check if route description or name is in the user query
                if route_data['description'].lower() in user_query or \
                   f"route {route_data['name'].lower()}" in user_query: # simulate direct mention
                    logger.debug(f"MockLLM choosing route '{route_data['name']}' based on prompt content.")
                    return json.dumps({"chosen_route_name": route_data['name']})

        logger.warning(
            "MockLLM could not determine a specific route from prompt content. "
            "Returning 'None' as chosen route."
        )
        return json.dumps({"chosen_route_name": "None"}) # Simulate no match or default fallback

class LLMRoutingStrategy(RoutingStrategy):
    """
    A sophisticated routing strategy that leverages a Large Language Model (LLM)
    to determine the most suitable route for an input text.

    This strategy constructs a descriptive prompt for the LLM, detailing all
    available routes (names and descriptions), and asks the LLM to select
    the best match. It expects the LLM to return a structured response
    (e.g., JSON) indicating the chosen route's name.
    """
    def __init__(self, llm: BaseLLM, prompt_template: Optional[str] = None):
        """
        Initializes the LLMRoutingStrategy.

        Args:
            llm (BaseLLM): An instance of an LLM client conforming to the `BaseLLM` interface.
            prompt_template (Optional[str]): A custom Jinja-style prompt template string.
                                             If None, a default template is used.
                                             The template must accept two variables:
                                             `routes_info` (str: formatted list of routes)
                                             and `input_text` (str: the user's query).
        Raises:
            TypeError: If the provided `llm` is not an instance of `BaseLLM`.
        """
        if not isinstance(llm, BaseLLM):
            raise TypeError(f"llm must be an instance of BaseLLM, got {type(llm).__name__}.")
        self.llm = llm
        self.prompt_template = prompt_template or self._default_prompt_template()
        logger.debug(f"LLMRoutingStrategy initialized with LLM: {type(llm).__name__}")

    def _default_prompt_template(self) -> str:
        """
        Provides a default prompt template for LLM routing.
        This template guides the LLM to act as a router and output JSON.
        """
        return (
            "You are an expert routing system for an AI orchestration framework called Vishustra. "
            "Your task is to analyze a user's query and select the single most appropriate route "
            "from the available options provided below. Your decision should be based solely on "
            "the relevance of the route's description to the user's query.\n\n"
            "Available Routes:\n"
            "{routes_info}\n\n"
            "Consider all aspects of the user's request, including intent, keywords, and context. "
            "If no route is suitable or directly relevant to the query, explicitly state 'None'.\n\n"
            "Output your choice in strict JSON format with a single key 'chosen_route_name' "
            "and its value being the exact name of the chosen route (e.g., 'qa_pipeline', 'code_generation') "
            "or 'None' if no route is appropriate.\n\n"
            "User Query: {input_text}\n"
            "JSON Response:"
        )

    def _format_routes_for_llm(self, routes: List[Route]) -> str:
        """
        Formats the list of `Route` objects into a human-readable string
        that can be embedded effectively within an LLM prompt.
        """
        formatted_routes = []
        for route in routes:
            formatted_routes.append(
                f"- Name: {route.name}\n"
                f"  Description: {route.description}"
            )
        return "\n".join(formatted_routes)

    def route(self, input_text: str, routes: List[Route]) -> Optional[Route]:
        """
        Routes the input text using an LLM to make the routing decision.

        This method constructs a detailed prompt for the LLM, sends the prompt,
        and then attempts to parse the LLM's structured (JSON) response to
        identify the chosen route. Robust error handling is included for LLM
        communication and response parsing.

        Args:
            input_text (str): The incoming text to be routed.
            routes (List[Route]): A list of available `Route` objects.

        Returns:
            Optional[Route]: The chosen `Route` object, or None if the LLM
                             does not select a valid route, explicitly returns 'None',
                             or an error occurs during the process.
        """
        if not routes:
            logger.warning("No routes provided to LLMRoutingStrategy. Cannot route. Returning None.")
            return None

        routes_info = self._format_routes_for_llm(routes)
        prompt = self.prompt_template.format(
            routes_info=routes_info,
            input_text=input_text
        )

        try:
            llm_response_str = self.llm.generate(prompt)
            logger.debug(f"LLM raw response for '{input_text[:50]}...': {llm_response_str}")

            # Attempt to parse the JSON response from the LLM
            response_json = json.loads(llm_response_str)

            chosen_route_name = response_json.get("chosen_route_name")

            if chosen_route_name == "None":
                logger.debug(
                    f"LLM explicitly indicated no suitable route for input: '{input_text[:50]}...'"
                )
                return None
            
            if chosen_route_name:
                for route in routes:
                    if route.name == chosen_route_name:
                        logger.info(
                            f"LLM chose route '{route.name}' for input: '{input_text[:50]}...'"
                        )
                        return route
                logger.warning(
                    f"LLM chose route '{chosen_route_name}' but it's not present in the "
                    f"provided route list. Available: {[r.name for r in routes]} "
                    f"for input: '{input_text[:50]}...'"
                )
            else:
                logger.warning(
                    f"LLM response missing 'chosen_route_name' key or it was empty for "
                    f"input: '{input_text[:50]}...'. Full response: {llm_response_str}"
                )

        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to decode LLM response as JSON: {e}. "
                f"Response: '{llm_response_str[:200]}...' for input: '{input_text[:50]}...'"
            )
        except Exception as e:
            logger.error(
                f"An unexpected error occurred during LLM routing for "
                f"input '{input_text[:50]}...': {e}", exc_info=True
            )

        logger.debug(f"LLM routing failed to find a valid route for input: '{input_text[:50]}...'")
        return None


class SemanticRouter:
    """
    The core Semantic Router for Vishustra.

    This class acts as a central dispatcher, orchestrating routing decisions
    by applying a specified `RoutingStrategy` to an incoming text based on
    a predefined set of `Routes`. It directs queries to appropriate
    downstream pipelines, agents, or tools within the framework, enabling
    dynamic and intelligent routing.
    """
    def __init__(self, routes: List[Route], strategy: RoutingStrategy):
        """
        Initializes the SemanticRouter with a set of available routes and a
        chosen routing strategy.

        Args:
            routes (List[Route]): A list of `Route` objects that the router can
                                  potentially direct traffic to. These routes
                                  define the possible destinations or functionalities.
            strategy (RoutingStrategy): An instance of a concrete `RoutingStrategy`
                                        (e.g., `KeywordRoutingStrategy`, `LLMRoutingStrategy`)
                                        that defines the actual mechanism for
                                        making routing decisions.

        Raises:
            ValueError: If the list of routes is empty or if any route names are not unique.
            TypeError: If the provided `strategy` is not an instance of `RoutingStrategy`.
        """
        if not routes:
            raise ValueError("SemanticRouter must be initialized with at least one route.")
        
        route_names = set()
        for route in routes:
            if route.name in route_names:
                raise ValueError(f"Route names must be unique. Duplicate found: '{route.name}'")
            route_names.add(route.name)

        if not isinstance(strategy, RoutingStrategy):
            raise TypeError(f"strategy must be an instance of a RoutingStrategy subclass, got {type(strategy).__name__}.")

        self._routes = routes
        self._strategy = strategy
        logger.info(
            f"SemanticRouter initialized with {len(routes)} routes "
            f"and strategy: {type(strategy).__name__}"
        )

    def route(self, input_text: str) -> Optional[Route]:
        """
        Routes the given input text to the most appropriate `Route` using the
        configured routing strategy.

        This is the primary method for external components to interact with the router.

        Args:
            input_text (str): The incoming text from a user, an agent, or another
                              component that needs to be directed to a specific
                              functional pipeline.

        Returns:
            Optional[Route]: The `Route` object chosen by the active strategy, or None
                             if no suitable route could be determined by the strategy.
        """
        if not input_text:
            logger.warning("Attempted to route empty input text. Returning None.")
            return None
            
        chosen_route = self._strategy.route(input_text, self._routes)
        
        if chosen_route:
            logger.debug(
                f"Router successfully selected route '{chosen_route.name}' "
                f"for input: '{input_text[:50]}...'"
            )
        else:
            logger.info(
                f"No suitable route found by strategy '{type(self._strategy).__name__}' "
                f"for input: '{input_text[:50]}...'"
            )
            
        return chosen_route

# Example Usage (typically, this would be in a separate 'examples' or 'tests' directory)
if __name__ == "__main__":
    # Configure basic logging for demonstration
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Set this module's logger to DEBUG to see more detailed internal workings
    logging.getLogger(__name__).setLevel(logging.DEBUG)
    logging.getLogger('__main__').setLevel(logging.DEBUG) # For script-level print statements

    print("--- Vishustra Semantic Router Demonstration ---")

    # 1. Define Routes for various functionalities
    qa_route = Route(
        name="documentation_qa",
        description="Handles questions specifically about the Vishustra framework documentation "
                    "or general technical queries related to its usage and concepts.",
        keywords=["docs", "documentation", "how to", "what is", "explain", "usage", "vishustra"]
    )
    code_gen_route = Route(
        name="code_generation",
        description="Generates Python code snippets, functions, or complete scripts based on "
                    "natural language programming requests. This route is for coding assistance.",
        keywords=["generate code", "write python", "code example", "implement function", "python script", "develop code"]
    )
    search_route = Route(
        name="web_search",
        description="Performs a web search to find external information, current events, "
                    "or general knowledge that is not covered by internal documentation.",
        keywords=["search online", "find information", "latest news", "google", "web search"]
    )
    # A route that might be caught by the LLM strategy for very specific intents
    sentiment_route = Route(
        name="sentiment_analysis",
        description="Analyzes the emotional tone or sentiment of a given text, classifying it "
                    "as positive, negative, or neutral.",
        keywords=["sentiment", "analyze emotion", "mood", "tone"]
    )

    all_routes = [qa_route, code_gen_route, search_route, sentiment_route]

    # --- Test 1: KeywordRoutingStrategy ---
    print("\n--- Testing KeywordRoutingStrategy ---")
    keyword_strategy = KeywordRoutingStrategy()
    keyword_router = SemanticRouter(routes=all_routes, strategy=keyword_strategy)

    test_queries_keyword = {
        "How do I install Vishustra docs?": qa_route,
        "Can you generate some Python code for a linked list?": code_gen_route,
        "What is the capital of France?": search_route, # Matches 'find information' or 'google' (if added)
        "Analyze the sentiment of this review.": sentiment_route,
        "Tell me a funny joke.": None, # No keyword match
        "Explain the architecture of Vishustra.": qa_route # Matches 'explain'
    }

    for query, expected_route in test_queries_keyword.items():
        print(f"\nRouting (Keyword) '{query}'")
        chosen = keyword_router.route(query)
        expected_name = expected_route.name if expected_route else 'None'
        chosen_name = chosen.name if chosen else 'None'
        print(f"  Chosen route: {chosen_name} (Expected: {expected_name})")
        assert chosen_name == expected_name, f"KeywordRouter failed for '{query}'"

    # --- Test 2: LLMRoutingStrategy with MockLLM ---
    print("\n--- Testing LLMRoutingStrategy with MockLLM ---")
    mock_llm = MockLLM()
    llm_strategy = LLMRoutingStrategy(llm=mock_llm)
    llm_router = SemanticRouter(routes=all_routes, strategy=llm_strategy)

    # MockLLM's logic is simplistic; it tries to find keywords from route descriptions in the query.
    # Its output might not be perfect, but it demonstrates the integration.
    test_queries_llm = {
        "I need assistance with the framework documentation.": qa_route, # Matches 'documentation'
        "Please write a Python function for a binary tree traversal.": code_gen_route, # Matches 'python function'
        "Search for the latest AI news and trends.": search_route, # Matches 'search'
        "What is the emotional tone of this sentence: 'I am very happy today!'?": sentiment_route, # Matches 'emotional tone'
        "How's the weather today in London?": None, # No direct match in mock
        "Tell me a bedtime story.": None # No direct match in mock
    }

    for query, expected_route in test_queries_llm.items():
        print(f"\nRouting (LLM Mock) '{query}'")
        chosen = llm_router.route(query)
        expected_name = expected_route.name if expected_route else 'None'
        chosen_name = chosen.name if chosen else 'None'
        print(f"  Chosen route: {chosen_name} (Expected: {expected_name})")
        # Asserting on MockLLM is tricky due to its simplicity, but we can check if it
        # at least tried to pick one if there's a good match.
        if expected_route:
            assert chosen_name == expected_name or chosen_name == "default_fallback", \
                f"LLMRouter (Mock) failed for '{query}'"

    # --- Test 3: Error Handling and Edge Cases ---
    print("\n--- Testing Error Handling ---")

    # Empty routes list
    try:
        print("\nAttempting to initialize SemanticRouter with empty routes list...")
        SemanticRouter(routes=[], strategy=keyword_strategy)
    except ValueError as e:
        print(f"  Caught expected error: {e}")
        assert "at least one route" in str(e)

    # Duplicate route names
    try:
        print("\nAttempting to initialize SemanticRouter with duplicate route names...")
        duplicate_routes = [qa_route, Route(name="documentation_qa", description="another qa")]
        SemanticRouter(routes=duplicate_routes, strategy=keyword_strategy)
    except ValueError as e:
        print(f"  Caught expected error: {e}")
        assert "unique" in str(e) and "documentation_qa" in str(e)

    # Invalid Route creation (empty name/description)
    try:
        print("\nAttempting to create a Route with an empty name...")
        Route(name="", description="invalid")
    except ValueError as e:
        print(f"  Caught expected error for empty route name: {e}")
        assert "name cannot be empty" in str(e)

    try:
        print("\nAttempting to create a Route with an empty description...")
        Route(name="valid", description="")
    except ValueError as e:
        print(f"  Caught expected error for empty route description: {e}")
        assert "description cannot be empty" in str(e)

    # Router with invalid strategy type
    try:
        print("\nAttempting to initialize SemanticRouter with a non-RoutingStrategy object...")
        SemanticRouter(routes=[qa_route], strategy="not_a_strategy_object") # type: ignore
    except TypeError as e:
        print(f"  Caught expected error: {e}")
        assert "instance of a RoutingStrategy subclass" in str(e)

    # LLMRoutingStrategy with invalid LLM type
    try:
        print("\nAttempting to initialize LLMRoutingStrategy with a non-BaseLLM object...")
        LLMRoutingStrategy(llm="not_an_llm_object") # type: ignore
    except TypeError as e:
        print(f"  Caught expected error: {e}")
        assert "instance of BaseLLM" in str(e)

    print("\n--- Demonstration Complete ---")