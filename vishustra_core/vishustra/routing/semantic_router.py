import abc
import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar, Generic

# Type variable for the target component, allowing for flexible callable types
T = TypeVar("T", bound=Callable[..., Any])

# --- Vishustra Core Component Assumptions (minimalistic for this file) ---
# These classes are assumed to exist elsewhere in the 'vishustra' framework,
# providing necessary abstractions for modularity and dependency injection.

class AbstractLLMClient(abc.ABC):
    """
    Abstract base class for LLM clients within Vishustra.
    Concrete implementations would connect to specific LLM providers (e.g., OpenAI, Anthropic, HuggingFace).
    This ensures the SemanticRouter is decoupled from any particular LLM vendor.
    """
    @abc.abstractmethod
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Sends a series of messages to the LLM and returns the completion string.

        Args:
            messages (List[Dict[str, str]]): A list of message dictionaries, each with 'role' and 'content'.
            model (Optional[str]): The specific model to use (e.g., "gpt-4-turbo"). If None, uses client's default.
            temperature (float): Sampling temperature for generation.
            max_tokens (Optional[int]): Maximum number of tokens to generate.
            response_format (Optional[Dict[str, str]]): Hints the LLM to respond in a specific format, e.g., JSON.
            **kwargs (Any): Additional provider-specific parameters.

        Returns:
            str: The raw text content of the LLM's response.
        """
        pass

    @abc.abstractmethod
    def get_default_model(self) -> str:
        """
        Returns the default model string configured for this LLM client.
        """
        pass

# --- Semantic Router Specific Components ---

@dataclass(frozen=True)
class Route(Generic[T]):
    """
    Represents a potential path or component the SemanticRouter can direct a query to.

    Attributes:
        name (str): A unique identifier for the route. This name will be used by the
                    router LLM to indicate its selection in its JSON response.
        description (str): A natural language description of what this route handles
                           or what its target component does. This description is
                           critical for the router LLM to make informed decisions.
        target (T): The callable component (e.g., a function, a method, an LLM chain,
                    an agent tool, another router) that should be invoked if this
                    route is chosen. The type `T` allows for arbitrary callable types.
    """
    name: str
    description: str
    target: T

@dataclass(frozen=True)
class RoutingResult(Generic[T]):
    """
    The outcome of a semantic routing operation.

    Attributes:
        chosen_route_name (str): The name of the route selected by the router LLM.
        target_component (T): The callable component associated with the chosen route.
        reasoning (str): The LLM's explanation for choosing this particular route.
        score (Optional[float]): An optional confidence score (0.0 to 1.0) provided by the LLM
                                 (if supported by the LLM's output format).
    """
    chosen_route_name: str
    target_component: T
    reasoning: str
    score: Optional[float] = None

class SemanticRouterError(Exception):
    """Base exception for SemanticRouter related errors."""
    pass

class InvalidRouteConfigurationError(SemanticRouterError):
    """Raised when a route is misconfigured (e.g., duplicate name)."""
    pass

class LLMResponseParsingError(SemanticRouterError):
    """Raised when the LLM's response cannot be parsed as expected JSON."""
    pass

class NoRouteChosenError(SemanticRouterError):
    """Raised when the LLM fails to choose an existing route, or if no fallback is available."""
    pass

class SemanticRouter(Generic[T]):
    """
    A powerful component for dynamically routing user queries or system prompts
    to appropriate downstream components based on semantic understanding.

    It leverages an LLM to analyze the intent of an input and select the best
    matching registered route. This allows for highly flexible and intelligent
    dispatching within an orchestration framework like Vishustra.

    The router expects the LLM to respond with a JSON object containing
    `"chosen_route"`, `"reasoning"`, and an optional `"score"`.

    Example Usage:
        import asyncio

        # Assume these are concrete components or functions defined elsewhere in Vishustra
        async def handle_rag_query(query: str):
            print(f"[RAG System] Processing: '{query}'")
            await asyncio.sleep(0.1) # Simulate async work
            return f"RAG response for '{query}'"

        async def handle_tool_query(query: str):
            print(f"[Tool Executor] Executing for: '{query}'")
            await asyncio.sleep(0.1)
            return f"Tool execution result for '{query}'"

        async def handle_greeting(query: str):
            print(f"[Greeting Handler] Responding to: '{query}'")
            await asyncio.sleep(0.05)
            return f"Hello! How can I help you with '{query}' today?"

        async def handle_fallback(query: str):
            print(f"[Fallback Handler] Defaulting for: '{query}'")
            await asyncio.sleep(0.05)
            return f"I'm not sure how to handle '{query}'. Could you rephrase?"

        # Assume `llm_client` is an instance of AbstractLLMClient (e.g., from vishustra.llm.openai_client)
        # For demonstration, we'd use a mock here.
        class MockLLMClient(AbstractLLMClient):
            def get_default_model(self) -> str:
                return "mock-routing-llm"
            async def chat_completion(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
                user_query = messages[-1]["content"].lower()
                if "capital" in user_query or "fact" in user_query:
                    return json.dumps({"chosen_route": "knowledge_base", "reasoning": "Factual query.", "score": 0.98})
                elif "book" in user_query or "restaurants" in user_query:
                    return json.dumps({"chosen_route": "tool_agent", "reasoning": "Requires tool use.", "score": 0.95})
                elif "hello" in user_query or "hi" in user_query:
                    return json.dumps({"chosen_route": "greeting_bot", "reasoning": "Simple greeting.", "score": 0.99})
                else:
                    return json.dumps({"chosen_route": "fallback", "reasoning": "No specific intent found.", "score": 0.6})

        async def main():
            mock_llm_client = MockLLMClient()
            router = SemanticRouter[Callable[[str], Any]](llm_client=mock_llm_client)

            router.add_route(Route(
                name="knowledge_base",
                description="Answers questions requiring factual information or data retrieval, typically using a RAG pipeline.",
                target=handle_rag_query
            ))
            router.add_route(Route(
                name="tool_agent",
                description="Executes a specific external tool or API call to perform an action or get dynamic real-time data.",
                target=handle_tool_query
            ))
            router.add_route(Route(
                name="greeting_bot",
                description="Handles general greetings and initial conversational pleasantries.",
                target=handle_greeting
            ))
            router.add_route(Route(
                name="fallback",
                description="A default route for queries that do not clearly match any other defined route.",
                target=handle_fallback
            ))

            queries_to_test = [
                "What is the capital of France?",
                "Find me Italian restaurants near Paris.",
                "Hi there!",
                "Tell me a story about space travel.", # This should hit the fallback route
            ]

            for query in queries_to_test:
                print(f"\nRouting query: '{query}'")
                try:
                    result = await router.route(query)
                    print(f"  Chosen: '{result.chosen_route_name}' (Score: {result.score:.2f})")
                    print(f"  Reasoning: {result.reasoning}")
                    # Invoke the chosen component with the original query
                    response = await result.target_component(query)
                    print(f"  Component Response: {response}")
                except SemanticRouterError as e:
                    print(f"  Routing Error: {e}")

        # if __name__ == "__main__":
        #    asyncio.run(main())
    """
    _DEFAULT_ROUTING_PROMPT_TEMPLATE = """
    You are an expert routing system for an AI orchestration framework called Vishustra.
    Your task is to analyze a user's query and determine the most appropriate
    downstream component (route) to handle it.

    Below is a list of available routes. Each route has a unique 'name' and a 'description'
    explaining what it is designed to do.

    --- Available Routes ---
    {routes_json}
    ------------------------

    Based on the user's query, select *ONE* of the routes that best matches the intent.
    Provide your choice and a brief reasoning in a JSON object.
    The JSON object MUST have the following structure:
    {{
        "chosen_route": "name_of_the_selected_route",
        "reasoning": "A brief explanation of why this route was chosen based on the query.",
        "score": 0.0-1.0 // An optional confidence score (1.0 for high confidence)
    }}
    If you cannot find a suitable route among the options, and a 'fallback' route is available,
    select 'fallback'.
    Ensure your response is valid JSON and contains *only* the JSON object.
    """

    def __init__(
        self,
        llm_client: AbstractLLMClient,
        routing_model: Optional[str] = None,
        prompt_template: Optional[str] = None,
        temperature: float = 0.0,
    ):
        """
        Initializes the SemanticRouter.

        Args:
            llm_client (AbstractLLMClient): An instance of an LLM client
                                            to power the routing decisions. This is
                                            a critical dependency.
            routing_model (Optional[str]): The specific LLM model to use for routing.
                                           If None, `llm_client.get_default_model()` is used.
                                           It's recommended to use a capable, fast model.
            prompt_template (Optional[str]): A custom prompt template string for the LLM.
                                             It MUST contain a `{routes_json}` placeholder
                                             where the route descriptions will be injected.
                                             The system message will be constructed from this.
            temperature (float): The sampling temperature for the routing LLM.
                                 Lower values (e.g., 0.0 or 0.1) make the decisions more
                                 deterministic and reliable, which is usually desired for routing.
        """
        self._llm_client = llm_client
        self._routing_model = routing_model if routing_model else llm_client.get_default_model()
        self._prompt_template = prompt_template if prompt_template else self._DEFAULT_ROUTING_PROMPT_TEMPLATE
        self._temperature = temperature
        self._routes: Dict[str, Route[T]] = {}

    def add_route(self, route: Route[T]) -> None:
        """
        Registers a new route with the SemanticRouter.

        Args:
            route (Route[T]): The route object containing its name, description, and target component.

        Raises:
            InvalidRouteConfigurationError: If a route with the same name already exists,
                                            preventing ambiguity.
        """
        if route.name in self._routes:
            raise InvalidRouteConfigurationError(f"Route with name '{route.name}' already exists.")
        self._routes[route.name] = route

    def _build_routing_prompt(self, query: str) -> List[Dict[str, str]]:
        """
        Constructs the messages list for the LLM based on all registered routes and the user query.

        Args:
            query (str): The user's input query.

        Returns:
            List[Dict[str, str]]: A list of messages formatted for the `AbstractLLMClient`.

        Raises:
            SemanticRouterError: If no routes have been registered, as routing would be impossible.
        """
        if not self._routes:
            raise SemanticRouterError("No routes have been registered with the SemanticRouter. Call `add_route()` first.")

        # Prepare route descriptions for the LLM
        route_descriptions = [
            {"name": r.name, "description": r.description} for r in self._routes.values()
        ]
        # Use compact JSON for the prompt to save token space unless indenting helps LLM.
        # Indent=None means most compact, Indent=2 for readability for the LLM.
        routes_json = json.dumps(route_descriptions, indent=2)

        # Inject route descriptions into the system prompt template
        system_message_content = self._prompt_template.format(routes_json=routes_json)

        return [
            {"role": "system", "content": system_message_content},
            {"role": "user", "content": query},
        ]

    async def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parses the LLM's raw text response into a structured dictionary.
        This method attempts to be robust by finding the first JSON object in the response,
        as LLMs can sometimes include preamble or postamble text.

        Args:
            response_text (str): The raw text output from the LLM.

        Returns:
            Dict[str, Any]: The parsed JSON object from the LLM.

        Raises:
            LLMResponseParsingError: If the response cannot be parsed as valid JSON or
                                     if no JSON object is found.
        """
        try:
            # Attempt to find the first JSON object in the response.
            # LLMs sometimes wrap JSON in text, or add preamble/postamble.
            json_start = response_text.find('{')
            json_end = response_text.rfind('}')
            if json_start == -1 or json_end == -1:
                raise ValueError("No complete JSON object found in LLM response.")
            json_str = response_text[json_start : json_end + 1]
            return json.loads(json_str)
        except (json.JSONDecodeError, ValueError) as e:
            raise LLMResponseParsingError(
                f"Failed to parse LLM response as JSON: {e}\nRaw response: {response_text}"
            ) from e

    async def route(self, query: str) -> RoutingResult[T]:
        """
        Analyzes the given query and routes it to the most semantically relevant component.

        This is the core method of the SemanticRouter. It constructs an LLM prompt,
        sends it to the configured LLM, parses the response, and identifies the
        target component.

        Args:
            query (str): The user's input query or prompt to be routed.

        Returns:
            RoutingResult[T]: An object containing the chosen route's details (name, reasoning, score)
                              and its associated target callable component.

        Raises:
            SemanticRouterError: If no routes are configured.
            LLMResponseParsingError: If the LLM's response cannot be parsed into the expected JSON format.
            NoRouteChosenError: If the LLM's parsed response does not contain a `chosen_route` or
                                if the chosen route name does not correspond to any registered route
                                and no 'fallback' route is defined.
        """
        messages = self._build_routing_prompt(query)

        # Call the underlying LLM client for chat completion
        llm_response_text = await self._llm_client.chat_completion(
            messages=messages,
            model=self._routing_model,
            temperature=self._temperature,
            response_format={"type": "json_object"}, # Hint for JSON output (if LLM provider supports it)
            max_tokens=512, # Limit tokens for routing LLM to prevent overly verbose responses
        )

        # Parse the LLM's raw text response
        parsed_response = await self._parse_llm_response(llm_response_text)

        chosen_route_name = parsed_response.get("chosen_route")
        reasoning = parsed_response.get("reasoning", "No specific reasoning provided by LLM.")
        score = parsed_response.get("score")

        if not chosen_route_name:
            raise NoRouteChosenError(
                f"LLM response did not specify a 'chosen_route' field. Raw: {llm_response_text}"
            )

        selected_route = self._routes.get(chosen_route_name)

        if not selected_route:
            # If the LLM selected an unknown route, attempt to use a 'fallback' route if it exists.
            if "fallback" in self._routes:
                fallback_route = self._routes["fallback"]
                print(
                    f"WARNING: LLM chose unknown route '{chosen_route_name}'. "
                    f"Falling back to 'fallback' route '{fallback_route.name}'."
                )
                return RoutingResult(
                    chosen_route_name=fallback_route.name,
                    target_component=fallback_route.target,
                    reasoning=f"Original choice '{chosen_route_name}' was invalid. {reasoning}",
                    score=0.5 if score is None else score * 0.5 # Indicate reduced confidence
                )
            else:
                # No 'fallback' route, so we must raise an error.
                raise NoRouteChosenError(
                    f"LLM chose route '{chosen_route_name}' which is not registered "
                    "and no 'fallback' route is defined to handle unknown choices."
                )

        return RoutingResult(
            chosen_route_name=chosen_route_name,
            target_component=selected_route.target,
            reasoning=reasoning,
            score=score,
        )

# The following block provides a self-contained demonstration and would typically
# be in a separate `examples/` or `tests/` directory in a real framework.
if __name__ == "__main__":
    class MockLLMClient(AbstractLLMClient):
        def get_default_model(self) -> str:
            return "mock-routing-llm"

        async def chat_completion(
            self,
            messages: List[Dict[str, str]],
            model: Optional[str] = None,
            temperature: float = 0.0,
            max_tokens: Optional[int] = None,
            response_format: Optional[Dict[str, str]] = None,
            **kwargs: Any,
        ) -> str:
            user_query = messages[-1]["content"].lower()
            # Simulate LLM decision based on keywords
            if "capital" in user_query or "fact" in user_query:
                return json.dumps({
                    "chosen_route": "knowledge_base",
                    "reasoning": "The query asks for factual information, suitable for a knowledge base.",
                    "score": 0.98
                })
            elif "restaurants" in user_query or "book" in user_query or "weather" in user_query:
                return json.dumps({
                    "chosen_route": "tool_agent",
                    "reasoning": "The query requires an external tool or API call to fetch dynamic real-time data.",
                    "score": 0.95
                })
            elif "hello" in user_query or "greeting" in user_query or "hi" in user_query:
                return json.dumps({
                    "chosen_route": "greeting_bot",
                    "reasoning": "A simple greeting or conversational opening.",
                    "score": 0.99
                })
            elif "error_route" in user_query: # Simulate LLM choosing a non-existent route
                 return json.dumps({
                    "chosen_route": "non_existent_route",
                    "reasoning": "This route is intentionally chosen to demonstrate error handling.",
                    "score": 0.1
                })
            else:
                return json.dumps({
                    "chosen_route": "fallback",
                    "reasoning": "Could not determine a specific intent for this query.",
                    "score": 0.6
                })

    # Define mock target components
    async def handle_knowledge_query(query: str) -> str:
        await asyncio.sleep(0.1)
        return f"[KnowledgeBase] Answer for '{query}': The answer is 42!"

    async def handle_tool_execution(query: str) -> str:
        await asyncio.sleep(0.2)
        return f"[ToolExecutor] Executed tool for '{query}': Booking confirmed!"

    async def handle_greeting(query: str) -> str:
        await asyncio.sleep(0.05)
        return f"[GreetingBot] Hello! How can I help you with '{query}'?"

    async def handle_fallback(query: str) -> str:
        await asyncio.sleep(0.05)
        return f"[Fallback] I'm unable to process '{query}' directly. Please rephrase."

    async def _demo_semantic_router():
        print("--- Vishustra Semantic Router Demonstration ---")

        mock_llm_client = MockLLMClient()
        # The TypeVar `T` is specified here for clarity in the demo.
        router = SemanticRouter[Callable[[str], Any]](llm_client=mock_llm_client, temperature=0.0)

        router.add_route(Route(
            name="knowledge_base",
            description="Handles queries requiring factual information or retrieval from a knowledge base (e.g., RAG).",
            target=handle_knowledge_query
        ))
        router.add_route(Route(
            name="tool_agent",
            description="Routes queries that require invoking external tools, APIs, or specialized agents (e.g., booking, weather, search).",
            target=handle_tool_execution
        ))
        router.add_route(Route(
            name="greeting_bot",
            description="Responds to simple greetings and conversational openings.",
            target=handle_greeting
        ))
        router.add_route(Route(
            name="fallback", # It's good practice to always have a fallback route.
            description="A default route for queries that do not match any other defined routes.",
            target=handle_fallback
        ))

        queries = [
            "What is the capital of France?",
            "Find me restaurants near Paris that serve French cuisine.",
            "Hello, how are you today?",
            "Tell me a story about a wizard.", # Should hit fallback with mock
            "What's the weather like in London tomorrow?",
            "Process this error_route for me." # Should demonstrate fallback for unknown route chosen by LLM
        ]

        for q in queries:
            print(f"\n--- Query: '{q}' ---")
            try:
                result = await router.route(q)
                print(f"  --> Chosen route: '{result.chosen_route_name}' (Score: {result.score})")
                print(f"  --> Reasoning: {result.reasoning}")
                component_response = await result.target_component(q)
                print(f"  --> Component response: {component_response}")
            except SemanticRouterError as e:
                print(f"  --> Error routing query: {e}")

    asyncio.run(_demo_semantic_router())