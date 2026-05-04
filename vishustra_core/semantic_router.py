import json
import logging
from typing import List, Optional, Dict, Any, Type
import asyncio # Used only for example __main__ block

from pydantic import BaseModel, Field, ValidationError

# Assume these are standard imports within the Vishustra framework
# vishustra.llm_clients: Module for interacting with various LLM providers
# vishustra.core.errors: Custom error classes for the framework
# vishustra.core.prompts: Utilities for managing and formatting prompts

# Placeholder imports for self-contained example.
# In a real Vishustra setup, these would be actual framework components.
class BaseLLMClient:
    """Abstract base class for all LLM clients in Vishustra."""
    async def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError("LLM client must implement the 'generate' method.")

class RoutingError(Exception):
    """Custom exception for errors occurring during semantic routing."""
    pass

class InvalidRouteConfigurationError(Exception):
    """Custom exception for invalid configuration of routes."""
    pass

class PromptTemplate(BaseModel):
    """Simple prompt template model."""
    template: str
    
    def format(self, **kwargs) -> str:
        """Formats the template string with provided keyword arguments."""
        return self.template.format(**kwargs)

# End Placeholder imports

logger = logging.getLogger(__name__)

class Route(BaseModel):
    """
    Represents a potential processing route or destination within the Vishustra framework.
    Each route is defined by a unique name, a detailed description, and optional metadata.
    """
    name: str = Field(..., description="A unique identifier for the route.")
    description: str = Field(..., description="A detailed natural language description of what this route handles or represents.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary key-value pairs associated with the route, e.g., target service, required parameters.")

class RouteDecision(BaseModel):
    """
    The outcome of a semantic routing operation, indicating the chosen route
    and the reasoning behind the decision.
    """
    chosen_route_name: Optional[str] = Field(None, description="The name of the route chosen by the router. 'None' if no suitable route was found.")
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="An optional confidence score (0.0 to 1.0) for the routing decision.")
    reasoning: Optional[str] = Field(None, description="The LLM's explanation for choosing the particular route or for not finding a suitable one.")
    raw_llm_output: str = Field(..., description="The raw, unparsed output received directly from the LLM during the routing process.")

class SemanticRouter:
    """
    An advanced semantic router designed for Vishustra. This component leverages
    a Large Language Model (LLM) to intelligently select the most appropriate
    downstream processing route based on the semantic understanding of a user's
    input query.

    The router is highly configurable, supporting various LLM clients and allowing
    customization of the routing prompt. It ensures robustness through retry
    mechanisms for parsing LLM responses.
    """
    _DEFAULT_ROUTING_PROMPT_TEMPLATE = """
    You are an intelligent routing agent responsible for directing user queries to the most relevant processing path.
    Your decision should be based solely on the semantic content of the user's request and the descriptions of the available routes.

    Here are the available routes, each with a clear description of its purpose:
    {routes_description}

    ---
    User Query: "{user_query}"
    ---

    Please analyze the user's query and select the SINGLE BEST route.
    If no route is suitable, explicitly indicate 'None' as the chosen route.

    Respond STRICTLY with a JSON object. The JSON object MUST contain the following keys:
    1.  "chosen_route_name": (string) The exact 'name' of the selected route, or "None".
    2.  "reasoning": (string) A concise explanation of why this route was selected, or why no route was deemed suitable.
    3.  "confidence_score": (float) A value between 0.0 (very uncertain) and 1.0 (very certain) reflecting your confidence in the chosen route.

    Example for a specific route:
    {{
        "chosen_route_name": "data_management_route",
        "reasoning": "The query explicitly asks to update personal information, which is handled by data management.",
        "confidence_score": 0.98
    }}

    Example for no suitable route:
    {{
        "chosen_route_name": "None",
        "reasoning": "The query about a 'purple elephant' does not align with any defined processing routes.",
        "confidence_score": 0.15
    }}

    Begin your JSON response now:
    """

    def __init__(
        self,
        routes: List[Route],
        llm_client: BaseLLMClient,
        routing_prompt_template: Optional[PromptTemplate] = None,
        max_retries: int = 3,
        parse_retry_delay_seconds: float = 0.5
    ):
        """
        Initializes the SemanticRouter with a set of routes and an LLM client.

        Args:
            routes: A list of `Route` objects defining the available routing options.
            llm_client: An instance of `BaseLLMClient` to power the routing decisions.
            routing_prompt_template: An optional `PromptTemplate` to override the default
                                     prompt used for LLM routing. It must support
                                     `routes_description` and `user_query` variables.
            max_retries: The maximum number of times to retry parsing the LLM's response
                         if it fails to conform to the expected JSON format.
            parse_retry_delay_seconds: Delay in seconds between retries for parsing errors.

        Raises:
            InvalidRouteConfigurationError: If the provided `routes` list is empty or
                                            contains non-unique route names.
        """
        if not routes:
            raise InvalidRouteConfigurationError("SemanticRouter must be initialized with at least one route.")
        if len(set(r.name for r in routes)) != len(routes):
            duplicate_names = [name for name, count in Counter(r.name for r in routes).items() if count > 1]
            raise InvalidRouteConfigurationError(f"Route names must be unique. Duplicates found: {', '.join(duplicate_names)}")

        self._routes = {route.name: route for route in routes}
        self._llm_client = llm_client
        self._prompt_template = routing_prompt_template or PromptTemplate(template=self._DEFAULT_ROUTING_PROMPT_TEMPLATE)
        self._max_retries = max_retries
        self._parse_retry_delay = parse_retry_delay_seconds

        logger.info(f"SemanticRouter initialized with {len(routes)} routes: {[r.name for r in routes]}.")

    @property
    def routes(self) -> List[Route]:
        """Returns a list of the configured `Route` objects."""
        return list(self._routes.values())

    def _format_routes_description(self) -> str:
        """
        Generates a formatted string of all route names and their descriptions,
        suitable for inclusion in the LLM prompt.
        """
        description_parts = []
        for route in self._routes.values():
            description_parts.append(f"- Route Name: '{route.name}'\n  Description: '{route.description}'")
        return "\n\n".join(description_parts)

    async def route(self, user_query: str) -> RouteDecision:
        """
        Routes a given user query to the most semantically relevant processing path
        by consulting the configured LLM.

        The method constructs a prompt detailing the available routes and the user's
        query, sends it to the LLM, and parses the LLM's JSON response to determine
        the routing decision. It includes retry logic for robust parsing.

        Args:
            user_query: The input string from the user that needs to be routed.

        Returns:
            A `RouteDecision` object encapsulating the chosen route's name (or "None"),
            the LLM's reasoning, and the raw output received from the LLM.

        Raises:
            RoutingError: If the LLM response cannot be parsed into the expected format
                          after `max_retries`, or if any other unexpected error occurs
                          during the routing process.
        """
        routes_desc = self._format_routes_description()
        llm_response_content = "" # Initialize for logging in case of early error

        for attempt in range(self._max_retries):
            try:
                prompt_text = self._prompt_template.format(
                    routes_description=routes_desc,
                    user_query=user_query
                )
                
                logger.debug(f"Attempt {attempt + 1}/{self._max_retries}: Sending routing prompt to LLM for query: '{user_query[:50]}...'")
                llm_response_content = await self._llm_client.generate(prompt_text)
                
                # Robust parsing: Remove common LLM markdown wrappers if present
                cleaned_response = llm_response_content.strip()
                if cleaned_response.startswith("```json"):
                    cleaned_response = cleaned_response[len("```json"):]
                if cleaned_response.endswith("```"):
                    cleaned_response = cleaned_response[:-len("```")]
                cleaned_response = cleaned_response.strip()

                response_data = json.loads(cleaned_response)
                
                # Use Pydantic to validate the structure and types of the LLM's decision
                decision = RouteDecision(
                    chosen_route_name=response_data.get("chosen_route_name"),
                    confidence_score=response_data.get("confidence_score"),
                    reasoning=response_data.get("reasoning"),
                    raw_llm_output=llm_response_content
                )
                
                # Further validation: Ensure the chosen route name (if not 'None') actually exists
                if decision.chosen_route_name and decision.chosen_route_name != "None" \
                   and decision.chosen_route_name not in self._routes:
                    raise RoutingError(
                        f"LLM suggested non-existent route '{decision.chosen_route_name}'. "
                        f"Raw LLM output: {llm_response_content}"
                    )
                
                logger.info(f"Routing successful for query '{user_query[:50]}...'. Chosen route: '{decision.chosen_route_name}'. Confidence: {decision.confidence_score:.2f}")
                return decision

            except (json.JSONDecodeError, KeyError, ValidationError) as e:
                logger.warning(
                    f"Attempt {attempt + 1}/{self._max_retries}: Failed to parse LLM response for routing. Error: {type(e).__name__} - {e}. "
                    f"Raw LLM output (first 200 chars): '{llm_response_content[:200]}...'."
                )
                if attempt < self._max_retries - 1:
                    logger.debug(f"Retrying after {self._parse_retry_delay} seconds...")
                    await asyncio.sleep(self._parse_retry_delay) # Use asyncio.sleep for async context
                else:
                    raise RoutingError(
                        f"Failed to parse LLM response after {self._max_retries} retries. "
                        f"Last raw output: {llm_response_content}. Final error: {e}"
                    ) from e
            except Exception as e:
                logger.error(f"An unexpected error occurred during routing for query '{user_query[:50]}...': {e}", exc_info=True)
                raise RoutingError(f"Unexpected error during routing: {e}") from e

        # This line should logically be unreachable if max_retries is > 0 and exceptions are always raised.
        raise RoutingError(f"Routing failed for query '{user_query[:50]}...' due to an unhandled internal error.")


# Example Usage (for demonstration purposes, not part of the framework itself)
if __name__ == "__main__":
    import asyncio
    from collections import Counter
    import sys

    # Configure basic logging for the example
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.setLevel(logging.DEBUG) # Show debug messages for router internals

    # --- Mock Implementations for Demonstration ---
    class MockLLMClient(BaseLLMClient):
        """A mock LLM client for testing the SemanticRouter without a real LLM."""
        def __init__(self):
            self.call_count = 0

        async def generate(self, prompt: str, **kwargs) -> str:
            self.call_count += 1
            logger.debug(f"Mock LLM Call {self.call_count} received prompt (first 100 chars): {prompt[:100]}...")
            
            # Simulate different LLM responses based on keywords
            if "update my personal information" in prompt:
                return json.dumps({
                    "chosen_route_name": "user_data_management",
                    "reasoning": "The query is about managing user's personal data.",
                    "confidence_score": 0.98
                })
            elif "sentiment analysis" in prompt:
                return json.dumps({
                    "chosen_route_name": "text_analysis_service",
                    "reasoning": "The user explicitly requested sentiment analysis.",
                    "confidence_score": 0.95
                })
            elif "generate a report" in prompt:
                return json.dumps({
                    "chosen_route_name": "report_generation_engine",
                    "reasoning": "The query asks for report creation.",
                    "confidence_score": 0.90
                })
            elif "tell me a joke" in prompt or "general question" in prompt:
                return json.dumps({
                    "chosen_route_name": "conversational_ai",
                    "reasoning": "This is a general conversational query.",
                    "confidence_score": 0.85
                })
            elif "unrelated query xyz" in prompt:
                return json.dumps({
                    "chosen_route_name": "None",
                    "reasoning": "The query does not semantically align with any defined route.",
                    "confidence_score": 0.25
                })
            elif "faulty json test" in prompt:
                # Simulate an LLM occasionally returning malformed JSON for retry test
                if self.call_count % 2 == 1: # First retry attempt (or initial call if 1st attempt)
                    logger.debug("Mock LLM returning malformed JSON for retry test.")
                    return "```json\n{\"chosen_route_name\": \"conversational_ai\", \"reasoning\": \"Malform", "confidence_score\": 0.7}\n```" # Malformed
                else: # Subsequent attempts return valid JSON
                    logger.debug("Mock LLM returning valid JSON after malformed one.")
                    return json.dumps({
                        "chosen_route_name": "conversational_ai",
                        "reasoning": "Defaulting to general chat after initial parsing error.",
                        "confidence_score": 0.7
                    })
            elif "nonexistent route" in prompt:
                return json.dumps({
                    "chosen_route_name": "imaginary_route_123", # LLM hallucinates a route
                    "reasoning": "Thought this might be a new special route.",
                    "confidence_score": 0.6
                })
            else:
                return json.dumps({
                    "chosen_route_name": "conversational_ai",
                    "reasoning": "No specific route found, defaulting to general conversation.",
                    "confidence_score": 0.6
                })

    # --- Main example execution ---
    async def run_example():
        mock_llm_client = MockLLMClient()

        # Define some sample routes
        routes_config = [
            Route(name="user_data_management", description="Handles queries related to user profiles, account settings, privacy, and personal data updates."),
            Route(name="text_analysis_service", description="Provides natural language processing services like sentiment analysis, entity extraction, and summarization."),
            Route(name="report_generation_engine", description="Generates custom reports, dashboards, and data visualizations based on specific criteria or datasets."),
            Route(name="conversational_ai", description="Engages in general conversation, answers common questions, and handles casual chit-chat."),
        ]

        # Test router initialization with invalid configs
        try:
            SemanticRouter(routes=[], llm_client=mock_llm_client)
        except InvalidRouteConfigurationError as e:
            logger.error(f"Caught expected configuration error: {e}")

        duplicate_routes = [routes_config[0], routes_config[0]] # Simulate duplicate name
        try:
            SemanticRouter(routes=duplicate_routes, llm_client=mock_llm_client)
        except InvalidRouteConfigurationError as e:
            logger.error(f"Caught expected configuration error for duplicate routes: {e}")


        try:
            router = SemanticRouter(routes=routes_config, llm_client=mock_llm_client, max_retries=2, parse_retry_delay_seconds=0.1)

            test_queries = [
                "I need to change my email address and password. How do I do that?",
                "Can you perform sentiment analysis on this customer review: 'The service was atrocious, utterly disappointing.'",
                "Please generate a quarterly financial report for Q2 2024, focusing on revenue growth.",
                "Tell me a fun fact about space!",
                "What is the capital of France? This is a general question.",
                "This is a completely unrelated query xyz.",
                "I have a faulty json test query.", # Should trigger a retry
                "What if the LLM suggests a nonexistent route?", # Should trigger RoutingError
            ]

            for i, query in enumerate(test_queries):
                print(f"\n--- Routing Test {i+1}: Query = '{query}' ---")
                try:
                    decision = await router.route(query)
                    print(f"  Chosen Route: {decision.chosen_route_name or 'N/A'}")
                    print(f"  Confidence: {decision.confidence_score:.2f}" if decision.confidence_score is not None else "  Confidence: N/A")
                    print(f"  Reasoning: {decision.reasoning}")
                    # print(f"  Raw LLM Output (truncated): {decision.raw_llm_output[:150]}...")
                except RoutingError as e:
                    print(f"  ERROR: Failed to route query '{query}'. Reason: {e}")
                except Exception as e:
                    print(f"  UNEXPECTED ERROR during routing query '{query}': {e}")

        except Exception as e:
            logger.critical(f"An error occurred during router initialization or overall example execution: {e}")

    # Run the asynchronous example
    asyncio.run(run_example())
