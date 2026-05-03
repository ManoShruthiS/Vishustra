import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Generic, Awaitable

from pydantic import BaseModel, Field, ValidationError

# --- Configuration and Pydantic Models ---

logger = logging.getLogger(__name__)

class LLMResponse(BaseModel):
    """Represents a standardized response from an LLM call."""
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class LLMClient(ABC):
    """Abstract base class for LLM clients, enabling modular LLM integration."""

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Generates a response from the LLM.

        Args:
            prompt: The input prompt for the LLM.
            **kwargs: Additional parameters specific to the LLM provider (e.g., temperature, max_tokens).

        Returns:
            An LLMResponse object containing the generated text and metadata.
        """
        pass

    @abstractmethod
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """
        Generates a chat completion response from the LLM.

        Args:
            messages: A list of message dictionaries (e.g., [{"role": "user", "content": "..."}]).
            **kwargs: Additional parameters specific to the LLM provider.

        Returns:
            An LLMResponse object.
        """
        pass

class Route(BaseModel):
    """
    Defines a specific routing path within the Vishustra framework.

    Attributes:
        name: A unique identifier for the route.
        description: A concise description of what this route handles.
        destination: The target to route to (e.g., a function, a chain ID, a symbolic name).
        examples: Optional list of example user queries that should match this route.
    """
    name: str = Field(..., description="Unique name for the route.")
    description: str = Field(..., description="Description of what this route handles.")
    destination: Any = Field(..., description="The target destination (e.g., function, chain ID).")
    examples: List[str] = Field(default_factory=list, description="Example queries for this route.")

class RoutingResult(BaseModel):
    """
    Represents the outcome of a semantic routing operation.

    Attributes:
        selected_route_name: The name of the route that was selected.
        destination: The destination associated with the selected route.
        confidence_score: A score (0.0 to 1.0) indicating the LLM's confidence in the selection.
        original_query: The original query that was routed.
        classification_raw_output: The raw output from the LLM's classification attempt.
        reasoning: The LLM's explanation for its routing decision.
    """
    selected_route_name: str
    destination: Any
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    original_query: str
    classification_raw_output: str
    reasoning: Optional[str] = None

class SemanticRouterConfig(BaseModel):
    """
    Configuration for the SemanticRouter.
    """
    max_retries: int = Field(3, description="Maximum retries for LLM classification if parsing fails.")
    default_route_name: Optional[str] = Field(None, description="Name of the default route if no match is found.")
    routing_temperature: float = Field(0.1, ge=0.0, le=2.0, description="Temperature for the internal routing LLM.")
    routing_model_name: Optional[str] = Field(None, description="Specific model name to use for routing if LLM client supports it.")

# --- SemanticRouter Implementation ---

class SemanticRouter:
    """
    An advanced semantic router for Vishustra, capable of intelligently directing
    user queries to the most appropriate backend destination using an LLM classifier.

    This router excels in dynamic orchestration scenarios by abstracting away
    complex conditional logic into semantic descriptions.

    Example Usage (assuming an LLMClient implementation exists, e.g., OpenAIChatClient):
        import os
        # from vishustra.llm_clients.openai import OpenAIChatClient # Example LLMClient path

        # --- Mock LLMClient for demonstration purposes ---
        class MockOpenAIChatClient(LLMClient):
            async def generate(self, prompt: str, **kwargs) -> LLMResponse:
                # Mock implementation for generate, not used by this router
                raise NotImplementedError("Generate not implemented for mock chat client")

            async def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
                # Simulate LLM response based on query content
                user_query = messages[-1]["content"] if messages else ""
                
                if "weather" in user_query.lower() or "temperature" in user_query.lower():
                    response_text = '{"selected_route": "weather_query", "confidence_score": 0.95, "reasoning": "Query explicitly asks about weather conditions."}'
                elif "users" in user_query.lower() or "products" in user_query.lower():
                    response_text = '{"selected_route": "database_query", "confidence_score": 0.88, "reasoning": "Query seeks specific data from a database."}'
                else:
                    response_text = '{"selected_route": "general_chat", "confidence_score": 0.70, "reasoning": "General conversational query, no specific tool needed."}'
                
                # Simulate a parsing error occasionally for retry testing
                if "error" in user_query.lower():
                    response_text = '{"selected_route": "general_chat", "confidence_score": 0.70, "reasoning": "General conversational query, no specific tool needed."' # Malformed JSON

                await asyncio.sleep(0.1) # Simulate network latency
                return LLMResponse(text=response_text)
        # --- End Mock LLMClient ---
        
        # llm_client = OpenAIChatClient(api_key=os.getenv("OPENAI_API_KEY")) # Real client
        llm_client = MockOpenAIChatClient() # Using mock client for runnable example

        routes = [
            Route(
                name="weather_query",
                description="Routes queries asking about current weather or forecasts for a location.",
                destination="weather_tool_invoke_id",
                examples=["What's the weather like in London?", "Will it rain tomorrow in Paris?"]
            ),
            Route(
                name="database_query",
                description="Handles queries requiring information retrieval from a database.",
                destination="sql_agent_chain_id",
                examples=["How many users signed up last month?", "List all products in category 'electronics'."]
            ),
            Route(
                name="general_chat",
                description="For general conversation or questions not covered by other specific routes.",
                destination="general_llm_chat_chain_id"
            )
        ]

        router_config = SemanticRouterConfig(default_route_name="general_chat")
        router = SemanticRouter(llm_client=llm_client, routes=routes, config=router_config)

        async def run_example():
            result = await router.route("What's the temperature in New York right now?")
            print(f"Query: 'What's the temperature in New York right now?'")
            print(f"Routed to: {result.selected_route_name}, Destination: {result.destination}, Confidence: {result.confidence_score:.2f}")

            result = await router.route("Show me the sales figures for Q3 2023.")
            print(f"Query: 'Show me the sales figures for Q3 2023.'")
            print(f"Routed to: {result.selected_route_name}, Destination: {result.destination}, Confidence: {result.confidence_score:.2f}")

            result = await router.route("Tell me a fun fact about space.")
            print(f"Query: 'Tell me a fun fact about space.'")
            print(f"Routed to: {result.selected_route_name}, Destination: {result.destination}, Confidence: {result.confidence_score:.2f}")

            # Test with an input that might cause parsing error (mocked)
            try:
                result = await router.route("This query should trigger an error in parsing for retry test.")
                print(f"Query: 'This query should trigger an error in parsing for retry test.'")
                print(f"Routed to: {result.selected_route_name}, Destination: {result.destination}, Confidence: {result.confidence_score:.2f}")
            except Exception as e:
                print(f"Query: 'This query should trigger an error in parsing for retry test.' -> Encountered expected error: {e}")

        # asyncio.run(run_example())
    """
    
    def __init__(self, llm_client: LLMClient, routes: List[Route], config: Optional[SemanticRouterConfig] = None):
        """
        Initializes the SemanticRouter with an LLM client, a list of routes, and configuration.

        Args:
            llm_client: An instance of an LLMClient (e.g., OpenAI, Anthropic, local).
            routes: A list of Route objects defining the available routing paths.
            config: Optional configuration for the router.
        """
        if not isinstance(llm_client, LLMClient):
            raise TypeError("llm_client must be an instance of LLMClient.")
        if not routes:
            raise ValueError("At least one route must be provided.")

        self._llm_client = llm_client
        self._config = config or SemanticRouterConfig()

        self._routes_map: Dict[str, Route] = {route.name: route for route in routes}
        if len(self._routes_map) != len(routes):
            raise ValueError("Route names must be unique.")

        if self._config.default_route_name and self._config.default_route_name not in self._routes_map:
            raise ValueError(f"Default route '{self._config.default_route_name}' not found in provided routes.")

        logger.info(f"SemanticRouter initialized with {len(routes)} routes. Default: {self._config.default_route_name}")
        for route in routes:
            logger.debug(f"  - Route '{route.name}': '{route.description}'")

    def _construct_routing_prompt(self, query: str) -> List[Dict[str, str]]:
        """
        Constructs the prompt messages for the LLM to classify the user query.
        """
        route_descriptions_str = []
        for route_name, route in self._routes_map.items():
            examples_str = ""
            if route.examples:
                examples_str = "Example queries:\n" + "\n".join([f"- {ex}" for ex in route.examples])
            route_descriptions_str.append(
                f"<route_name>{route.name}</route_name>\n"
                f"<description>{route.description}</description>\n"
                f"{examples_str}\n"
            )
        
        # Use a system message to instruct the LLM on its role and output format
        system_message = (
            "You are an expert routing assistant for a large language model orchestration framework. "
            "Your task is to analyze a user's query and classify it into the most appropriate "
            "predefined route based on its semantic meaning. "
            "You MUST respond with a JSON object containing the 'selected_route', 'confidence_score' (0.0-1.0), "
            "and a brief 'reasoning' for your choice. "
            "If no route is a good fit, select the route that best represents a 'default' or 'general' handling, "
            "or indicate a lower confidence if no suitable default is available. "
            "Prioritize routes with examples that closely match the query. "
            "Ensure the 'selected_route' is one of the provided <route_name> values."
        )

        user_message_content = (
            "Here are the available routes:\n\n"
            + "\n---\n".join(route_descriptions_str) +
            "\n---\n\n"
            f"User Query: {query}\n\n"
            "Please select the best route and provide your reasoning. "
            "Ensure your response is a valid JSON object with 'selected_route', 'confidence_score', and 'reasoning' keys."
            "Example JSON: {\"selected_route\": \"weather_query\", \"confidence_score\": 0.9, \"reasoning\": \"Query explicitly asks about weather.\"} "
            "Make sure 'confidence_score' is a float between 0.0 and 1.0."
        )

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message_content}
        ]

    def _parse_llm_output(self, llm_output_text: str, query: str) -> RoutingResult:
        """
        Parses the raw LLM output into a structured RoutingResult.
        Handles JSON parsing errors and assigns default if necessary.
        """
        try:
            parsed_json = json.loads(llm_output_text)
            selected_route_name = parsed_json.get("selected_route")
            confidence_score = float(parsed_json.get("confidence_score", 0.0))
            reasoning = parsed_json.get("reasoning")

            if not isinstance(selected_route_name, str) or selected_route_name not in self._routes_map:
                logger.warning(
                    f"LLM selected unknown route '{selected_route_name}' or format error. "
                    f"Falling back to default or assigning highest confidence to a fallback. Output: {llm_output_text}"
                )
                if self._config.default_route_name:
                    selected_route_name = self._config.default_route_name
                    # Reduce confidence significantly for an LLM error-induced fallback
                    confidence_score = min(confidence_score, 0.2) 
                    reasoning = f"LLM selected unknown route or malformed response; fell back to default: {reasoning}"
                else:
                    raise ValueError(f"LLM output could not be parsed into a valid known route and no default route is configured: {llm_output_text}")

            selected_route = self._routes_map[selected_route_name]

            # Ensure confidence score is within valid range
            confidence_score = max(0.0, min(1.0, confidence_score))

            return RoutingResult(
                selected_route_name=selected_route.name,
                destination=selected_route.destination,
                confidence_score=confidence_score,
                original_query=query,
                classification_raw_output=llm_output_text,
                reasoning=reasoning
            )
        except (json.JSONDecodeError, ValueError, TypeError, KeyError) as e:
            logger.error(f"Failed to parse LLM routing output: {e}. Raw output: '{llm_output_text}'")
            # Fallback in case of parsing failure
            if self._config.default_route_name:
                default_route = self._routes_map[self._config.default_route_name]
                return RoutingResult(
                    selected_route_name=default_route.name,
                    destination=default_route.destination,
                    confidence_score=0.1,  # Very low confidence for parsing failure fallback
                    original_query=query,
                    classification_raw_output=llm_output_text,
                    reasoning=f"Failed to parse LLM output: {e}. Fell back to default route."
                )
            else:
                raise RuntimeError(
                    f"Critical parsing error in semantic router and no default route is configured. "
                    f"Raw LLM output: '{llm_output_text}'"
                ) from e

    async def route(self, query: str) -> RoutingResult:
        """
        Routes a user query to the most appropriate destination using the configured LLM.

        Args:
            query: The user's input query string.

        Returns:
            A RoutingResult object detailing the selected route and its destination.

        Raises:
            RuntimeError: If the LLM fails to provide a parseable response after retries,
                          and no default route is configured.
        """
        prompt_messages = self._construct_routing_prompt(query)
        
        llm_kwargs = {"temperature": self._config.routing_temperature}
        if self._config.routing_model_name:
            llm_kwargs["model"] = self._config.routing_model_name

        for attempt in range(self._config.max_retries):
            logger.debug(f"Attempt {attempt + 1}/{self._config.max_retries} to route query: '{query}'")
            try:
                llm_response = await self._llm_client.chat(messages=prompt_messages, **llm_kwargs)
                return self._parse_llm_output(llm_response.text, query)
            except (json.JSONDecodeError, ValidationError, RuntimeError) as e:
                logger.warning(f"Routing LLM response parsing failed (attempt {attempt + 1}): {e}. Retrying...")
                await asyncio.sleep(0.5 * (attempt + 1)) # Exponential backoff
            except Exception as e:
                logger.error(f"An unexpected error occurred during LLM routing: {e}")
                raise

        # If all retries fail, and a default route is configured, use it. Otherwise, raise.
        if self._config.default_route_name:
            default_route = self._routes_map[self._config.default_route_name]
            logger.error(
                f"All LLM routing attempts failed for query '{query}'. "
                f"Falling back to configured default route '{default_route.name}'."
            )
            return RoutingResult(
                selected_route_name=default_route.name,
                destination=default_route.destination,
                confidence_score=0.05, # Very low confidence for forced fallback
                original_query=query,
                classification_raw_output="LLM routing failed after retries.",
                reasoning="All LLM attempts failed, falling back to default."
            )
        else:
            raise RuntimeError(
                f"Failed to route query '{query}' after {self._config.max_retries} attempts, "
                "and no default route is configured. Manual intervention or re-evaluation needed."
            )