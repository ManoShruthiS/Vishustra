import asyncio
import json
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Protocol,
    Tuple,
    Union,
    runtime_checkable,
)

from pydantic import BaseModel, Field, ValidationError

# --- Vishustra Core Component Protocols ---
# These protocols define the expected interfaces for Vishustra's internal
# embedding and LLM clients, ensuring modularity and easy integration.

@runtime_checkable
class EmbeddingClient(Protocol):
    """Protocol for Vishustra's embedding client."""
    async def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        Embeds a list of texts into vector representations.

        Args:
            texts: A single string or a list of strings to embed.

        Returns:
            A list of embedding vectors, where each vector is a list of floats.
        """
        ...

@runtime_checkable
class LLMClient(Protocol):
    """Protocol for Vishustra's LLM client."""
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 50,
        json_mode: bool = False,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """
        Generates text based on a given prompt.

        Args:
            prompt: The text prompt to send to the LLM.
            temperature: Controls the randomness of the output (0.0 for deterministic).
            max_tokens: The maximum number of tokens to generate.
            json_mode: If True, instructs the LLM to attempt to return valid JSON.
            stop_sequences: A list of sequences that will stop generation if encountered.

        Returns:
            The generated text response from the LLM.
        """
        ...

@runtime_checkable
class VishustraComponent(Protocol):
    """
    Base protocol for any Vishustra-specific component (e.g., Chain, Agent, Tool)
    that can be invoked and potentially routed to.
    """
    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Invokes the Vishustra component.

        Returns:
            The result of the component's execution.
        """
        ...

# --- Core Semantic Router Components ---

class Route(BaseModel):
    """
    Defines a specific routing destination within the Vishustra framework.

    Each route encapsulates a target component (e.g., a chain, an agent, a tool)
    and metadata describing its purpose, allowing the SemanticRouter to
    intelligently direct user queries.
    """
    name: str = Field(..., description="A unique identifier for this route (e.g., 'weather_tool', 'search_chain').")
    description: str = Field(
        ...,
        description="A detailed natural language description of what this route does "
                    "and when it should be used. Crucial for LLM-based routing "
                    "and often for embedding similarity."
    )
    target: Union[Callable[..., Any], VishustraComponent] = Field(
        ...,
        description="The callable Python object (e.g., a standard function, a Vishustra Chain, "
                    "or an Agent instance) that this route directs to. This will be invoked "
                    "with the query and any additional kwargs."
    )
    embedding_input: Optional[str] = Field(
        None,
        description="An optional, concise string representation for embedding purposes. "
                    "If not provided, the 'description' will be used. "
                    "Useful when the description is verbose but a shorter phrase "
                    "better captures the core intent for rapid semantic similarity matching."
    )

    class Config:
        """Pydantic configuration for handling arbitrary types like callables."""
        arbitrary_types_allowed = True

class EmbeddedRoute(NamedTuple):
    """Internal representation of a route with its pre-computed embedding."""
    route: Route
    embedding: List[float]

# --- Custom Exceptions for SemanticRouter ---

class SemanticRouterError(Exception):
    """Base exception for SemanticRouter related errors."""
    pass

class NoRouteFoundError(SemanticRouterError):
    """Raised when the router cannot determine a suitable route for a query."""
    pass

class LLMRoutingError(SemanticRouterError):
    """Raised when the decision LLM fails to provide a valid or parseable routing decision."""
    pass

class SemanticRouter:
    """
    An advanced routing mechanism for Vishustra, capable of intelligently directing
    user queries to appropriate backend components (chains, agents, tools) based on
    semantic understanding.

    It employs a hybrid routing strategy:
    1.  **Embedding-based similarity**: For quick, confidence-driven matches based
        on the semantic similarity between the user query and pre-defined route
        representations (descriptions or `embedding_input`).
    2.  **LLM-based decision-making**: For complex, nuanced queries or when
        embedding similarity is ambiguous (below a configurable threshold),
        leveraging a powerful LLM to choose the best route based on detailed
        route descriptions.

    This hybrid approach balances performance (fast embedding lookups) with
    flexibility and accuracy (LLM's understanding of complex intent).
    """

    DEFAULT_SIMILARITY_THRESHOLD: float = 0.75
    DEFAULT_LLM_TEMPERATURE: float = 0.0
    DEFAULT_LLM_MAX_TOKENS: int = 200 # Sufficient for a JSON response with route name

    def __init__(
        self,
        routes: List[Route],
        embedding_model: EmbeddingClient,
        decision_llm: LLMClient,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        llm_temperature: float = DEFAULT_LLM_TEMPERATURE,
        llm_max_tokens: int = DEFAULT_LLM_MAX_TOKENS,
    ):
        """
        Initializes the SemanticRouter with a set of routes and necessary clients.

        Args:
            routes: A list of `Route` objects defining all possible destinations.
                    These routes will be analyzed for semantic matching.
            embedding_model: An instance of Vishustra's `EmbeddingClient` for
                             converting text into vector representations for
                             semantic similarity computations.
            decision_llm: An instance of Vishustra's `LLMClient` used for
                          LLM-driven routing decisions when embedding similarity
                          is insufficient or for complex disambiguation.
            similarity_threshold: A float between 0.0 and 1.0. If the cosine
                                  similarity between the query and the best
                                  embedding-matched route is below this,
                                  the LLM will be consulted for a decision.
            llm_temperature: The temperature setting for the decision LLM. Lower
                             values (e.g., 0.0) make the output more deterministic.
            llm_max_tokens: The maximum number of tokens the decision LLM is
                            allowed to generate for its routing response.

        Raises:
            ValueError: If an empty list of routes is provided.
        """
        if not routes:
            raise ValueError("SemanticRouter must be initialized with at least one route.")

        self.routes = routes
        self.embedding_model = embedding_model
        self.decision_llm = decision_llm
        self.similarity_threshold = similarity_threshold
        self.llm_temperature = llm_temperature
        self.llm_max_tokens = llm_max_tokens

        self._embedded_routes: List[EmbeddedRoute] = []
        # A quick lookup map for routes by name, useful after LLM decision
        self._route_name_map: Dict[str, Route] = {route.name: route for route in routes}

    async def _precompute_route_embeddings(self) -> None:
        """
        Asynchronously pre-computes embeddings for all routes.
        This method should typically be called once after router initialization
        to prepare for efficient embedding-based routing.
        """
        texts_to_embed = [
            route.embedding_input if route.embedding_input else route.description
            for route in self.routes
        ]
        # Handle cases where texts_to_embed might be empty, though `routes` check
        # in __init__ makes this less likely for the initial call.
        if not texts_to_embed:
            self._embedded_routes = []
            return

        embeddings = await self.embedding_model.embed(texts_to_embed)

        self._embedded_routes = [
            EmbeddedRoute(route=self.routes[i], embedding=embeddings[i])
            for i in range(len(self.routes))
        ]

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculates the cosine similarity between two given vectors.

        Args:
            vec1: The first vector.
            vec2: The second vector.

        Returns:
            The cosine similarity score (between -1.0 and 1.0).
            Returns 0.0 if either vector is empty or a zero vector.
        """
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude_vec1 = sum(a * a for a in vec1) ** 0.5
        magnitude_vec2 = sum(b * b for b in vec2) ** 0.5

        if magnitude_vec1 == 0 or magnitude_vec2 == 0:
            return 0.0

        return dot_product / (magnitude_vec1 * magnitude_vec2)

    async def _get_embedding_match(self, query_embedding: List[float]) -> Tuple[Optional[Route], float]:
        """
        Finds the best matching route based on cosine similarity with the query embedding.

        Args:
            query_embedding: The embedding vector of the input query.

        Returns:
            A tuple containing the best `Route` object and its cosine similarity score.
            Returns (None, 0.0) if no routes are available or no good match is found.
        """
        # Ensure embeddings are precomputed. If not, trigger precomputation once.
        if not self._embedded_routes:
            await self._precompute_route_embeddings()

        if not self._embedded_routes or not query_embedding:
            return None, 0.0

        best_match: Optional[Route] = None
        highest_similarity: float = 0.0

        for embedded_route in self._embedded_routes:
            similarity = self._cosine_similarity(query_embedding, embedded_route.embedding)
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = embedded_route.route

        return best_match, highest_similarity

    def _generate_llm_routing_prompt(self, query: str) -> str:
        """
        Generates a detailed prompt for the LLM to make an explicit routing decision.
        The prompt instructs the LLM to select a route name and return it in JSON format.
        """
        route_descriptions = []
        for route in self.routes:
            route_descriptions.append(
                f"- Name: '{route.name}'\n"
                f"  Description: {route.description}\n"
            )

        prompt = (
            "You are an expert routing system for a large language model orchestration framework. "
            "Your task is to analyze a user's query and determine the SINGLE most appropriate "
            "backend component (route) to handle it. "
            "Below is the user's query and a list of available routes, each with a unique name and a detailed description. "
            "Carefully read the query and the descriptions of all routes. "
            "Respond ONLY with a JSON object containing a single key 'chosen_route' "
            "whose value is the 'name' of the best matching route. "
            "If no route perfectly matches, choose the one that is most conceptually relevant based on its description. "
            "Do NOT include any other text, explanations, or markdown outside the JSON object.\n\n"
            "--- Available Routes ---\n"
            f"{''.join(route_descriptions)}\n"
            "--- User Query ---\n"
            f"Query: \"{query}\"\n\n"
            "--- Your Decision (JSON ONLY) ---\n"
            "```json\n"
            "{ \"chosen_route\": \"<ROUTE_NAME_HERE>\" }\n"
            "```"
        )
        return prompt

    async def _get_llm_decision(self, query: str) -> Route:
        """
        Consults the decision LLM to determine the best route when embedding
        similarity is not confident enough.

        Args:
            query: The user's input query string.

        Returns:
            The `Route` object chosen by the LLM.

        Raises:
            LLMRoutingError: If the LLM fails to provide a valid or parseable decision,
                             or if there's an issue during LLM interaction.
            NoRouteFoundError: If the LLM's chosen route name does not correspond
                               to any of the configured routes.
        """
        prompt = self._generate_llm_routing_prompt(query)
        llm_response_text = "" # Initialize for error reporting

        try:
            llm_response_text = await self.decision_llm.generate(
                prompt=prompt,
                temperature=self.llm_temperature,
                max_tokens=self.llm_max_tokens,
                json_mode=True # Leverage LLMClient's JSON enforcement
            )

            # Defensive parsing, even if json_mode is active, due to LLM potential failures
            response_json = json.loads(llm_response_text)
            chosen_route_name = response_json.get("chosen_route")

            if not chosen_route_name or not isinstance(chosen_route_name, str):
                raise LLMRoutingError(
                    f"LLM response malformed or missing 'chosen_route' key. "
                    f"Raw response: '{llm_response_text}'"
                )

            chosen_route = self._route_name_map.get(chosen_route_name)
            if not chosen_route:
                raise NoRouteFoundError(
                    f"LLM chose route '{chosen_route_name}' which does not exist in "
                    f"the available routes: {list(self._route_name_map.keys())}"
                )
            return chosen_route
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            raise LLMRoutingError(
                f"Failed to parse LLM routing decision JSON. Error: {e}. "
                f"Raw response: '{llm_response_text}'"
            ) from e
        except Exception as e:
            raise LLMRoutingError(
                f"An unexpected error occurred during LLM routing decision: {e}. "
                f"Raw response: '{llm_response_text}'"
            ) from e


    async def route(self, query: str, **kwargs: Any) -> Any:
        """
        Routes the input query to the most appropriate component and invokes it.

        This method first attempts to route based on embedding similarity between
        the query and available routes. If a confident match is not found
        (i.e., similarity is below `self.similarity_threshold`), it defers
        to the decision LLM for a more nuanced and context-aware choice.
        Finally, it executes the `target` callable of the chosen route, passing
        the original `query` and any additional keyword arguments.

        Args:
            query: The user's input query string that needs to be routed.
            **kwargs: Additional keyword arguments to pass directly to the
                      `target` callable of the chosen route.

        Returns:
            The result of invoking the chosen target component. The type of this
            result depends entirely on the `target`'s return type.

        Raises:
            NoRouteFoundError: If no suitable route can be determined by either
                               embedding matching or the LLM.
            LLMRoutingError: If the LLM encounters an error or returns an
                             unparseable or invalid decision during its routing attempt.
            RuntimeError: If the chosen target component raises an exception during
                          its execution. The original exception is wrapped.
            ValueError: If the query is empty.
        """
        if not query:
            raise ValueError("Input query cannot be empty for routing.")

        # Ensure embeddings are ready; if not, precompute on first use.
        if not self._embedded_routes:
            await self._precompute_route_embeddings()

        if not self._embedded_routes:
            raise NoRouteFoundError("No routes configured or embeddings could not be computed.")

        # 1. Attempt embedding-based routing for quick, high-confidence matches
        query_embedding_list = await self.embedding_model.embed([query])
        query_embedding = query_embedding_list[0] if query_embedding_list else []

        best_embedding_route, similarity = await self._get_embedding_match(query_embedding)

        chosen_route: Optional[Route] = None
        if best_embedding_route and similarity >= self.similarity_threshold:
            chosen_route = best_embedding_route
        else:
            # 2. Fallback to LLM-based routing for complex or ambiguous queries
            # This is more computationally intensive but offers greater flexibility.
            try:
                chosen_route = await self._get_llm_decision(query)
            except (LLMRoutingError, NoRouteFoundError) as e:
                # Re-raise specific routing errors for clarity
                raise e
            except Exception as e:
                # Catch any other unexpected errors during LLM decision process
                raise LLMRoutingError(
                    f"An unexpected error occurred while consulting the decision LLM: {e}"
                ) from e

        if not chosen_route:
            raise NoRouteFoundError(f"No suitable route could be determined for query: '{query}'")

        # 3. Invoke the target of the chosen route
        try:
            # Determine if the target is an async callable or a regular one
            is_async_callable = asyncio.iscoroutinefunction(chosen_route.target) or (
                isinstance(chosen_route.target, VishustraComponent) and asyncio.iscoroutinefunction(chosen_route.target.__call__)
            )

            if is_async_callable:
                return await chosen_route.target(query=query, **kwargs)
            else:
                return chosen_route.target(query=query, **kwargs)
        except Exception as e:
            # Wrap any exceptions raised by the target component for better debugging context
            raise RuntimeError(
                f"Error executing target of route '{chosen_route.name}' "
                f"for query '{query}': {e}"
            ) from e