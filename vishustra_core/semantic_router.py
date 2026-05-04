import logging
from typing import Any, Dict, List, Literal, Optional, Protocol, Tuple, Type
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field, ValidationError, create_model

logger = logging.getLogger(__name__)

# --- Protocols/Interfaces ---

class LLMProvider(Protocol):
    """
    Protocol for an LLM provider capable of generating structured outputs.

    This interface ensures that different LLM integrations (e.g., OpenAI, Anthropic, local models)
    can be swapped out seamlessly, as long as they adhere to the contract of generating
    a Pydantic-schema-compliant JSON response.
    """
    async def generate_structured(self, prompt: str, schema: Type[BaseModel], **kwargs: Any) -> BaseModel:
        """
        Generates a structured output adhering to the provided Pydantic schema.

        Args:
            prompt: The full prompt string to send to the LLM.
            schema: The Pydantic model type that the LLM's response must conform to.
            **kwargs: Additional keyword arguments to pass to the underlying LLM client
                      (e.g., temperature, model_name).

        Returns:
            An instance of the provided Pydantic schema populated with the LLM's response.

        Raises:
            Exception: If the LLM call fails or if its output cannot be parsed
                       into the specified schema.
        """
        ...

# --- Data Models ---

class RouteConfig(BaseModel):
    """
    Configuration for a single routing destination within the Vishustra framework.

    Each route defines a distinct intent or capability that the system can handle.
    """
    name: str = Field(..., description="Unique identifier for this route. E.g., 'document_search', 'weather_query'.")
    description: str = Field(..., description="A clear, concise description of what this route handles or what its purpose is.")
    examples: List[str] = Field(default_factory=list,
                                description="A list of example user queries or phrases that should map to this route. "
                                            "These examples are used to train the underlying LLM for few-shot intent detection.")
    # In more advanced versions, 'input_schema' or 'output_schema' could be added
    # to guide the LLM on parameter extraction or expected response format for this specific route.

class RouterDecision(BaseModel):
    """
    Represents the output of the SemanticRouter, indicating the chosen route
    and supplementary information about the decision.
    """
    route_name: Optional[str] = Field(None, description="The name of the detected route. None if no route could be confidently determined.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="A confidence score (0.0 to 1.0) indicating the certainty of the routing decision.")
    reasoning: str = Field(..., description="A brief explanation provided by the LLM for why this specific route was chosen.")
    parameters: Dict[str, Any] = Field(default_factory=dict,
                                       description="Extracted parameters relevant to the detected route. "
                                                   "Note: Parameter extraction is not implemented in this version "
                                                   "but is included for future extensibility.")

# --- Router Implementation ---

class SemanticRouter:
    """
    The Vishustra Semantic Router orchestrates incoming user requests
    by intelligently detecting their intent and routing them to the
    most appropriate handler or chain within the framework.

    It leverages an underlying LLM to perform semantic intent analysis
    based on predefined route configurations and examples, ensuring
    flexible and robust request dispatching.
    """
    # A special token used internally for the LLM to signal a "default" choice.
    # This allows the LLM to explicitly say "I don't know" or "fall back" without
    # needing to know the actual default route's name.
    _DEFAULT_FALLBACK_TOKEN = "vishustra_default_fallback_route_token"

    def __init__(self, llm: LLMProvider, routes: List[RouteConfig], default_route_name: Optional[str] = None):
        """
        Initializes the SemanticRouter with a set of predefined routes.

        Args:
            llm: An instance of an LLMProvider capable of structured generation.
                 This LLM will be used for intent detection.
            routes: A list of `RouteConfig` objects defining the available routing paths.
            default_route_name: The name of a route to fall back to if no specific
                                  intent can be confidently detected by the LLM,
                                  or if the LLM explicitly suggests a fallback.
                                  Must correspond to a `name` in the `routes` list.

        Raises:
            ValueError: If `routes` is empty, or if `default_route_name` is specified
                        but does not exist in the provided routes.
        """
        if not routes:
            raise ValueError("SemanticRouter requires at least one route configuration to function.")

        self._llm = llm
        # Store routes in a dictionary for quick lookup by name
        self._routes: Dict[str, RouteConfig] = {r.name: r for r in routes}
        self._default_route_name = default_route_name

        if self._default_route_name and self._default_route_name not in self._routes:
            raise ValueError(f"Default route '{self._default_route_name}' not found in the provided route configurations.")

        # Dynamically create the Pydantic schema that the LLM is expected to output.
        # This schema enforces the LLM to select one of the defined route names.
        self._llm_routing_output_schema = self._create_llm_routing_output_schema()
        logger.info(f"Initialized SemanticRouter with {len(routes)} routes. Default route: {default_route_name or 'None'}.")

    def _create_llm_routing_output_schema(self) -> Type[BaseModel]:
        """
        Dynamically creates a Pydantic schema that specifies the expected JSON structure
        from the LLM for its routing decision.

        This schema constrains the LLM's output to ensure it provides a valid
        `route_name` from the configured routes, along with a `confidence` score
        and `reasoning`.
        """
        # Ensure that route names are unique and valid for Literal type
        route_names = list(self._routes.keys())
        if not route_names:
            # Should not happen if __init__ check passed, but for robustness.
            logger.error("No routes defined when creating LLM routing schema.")
            raise RuntimeError("Cannot create routing schema without defined routes.")

        # If a default route is configured, allow the LLM to output a special token
        # to indicate it couldn't find a confident match for any specific route.
        if self._default_route_name:
            # The LLM sees the token, we map it back to the actual route name later.
            route_names.append(self._DEFAULT_FALLBACK_TOKEN)

        # Create a Literal type from the list of available route names (and fallback token)
        RouteNameLiteral = Literal[tuple(route_names)] # type: ignore

        # Dynamically define the fields for the Pydantic model the LLM should return.
        dynamic_schema_fields = {
            "route_name": (RouteNameLiteral, Field(..., description="The name of the most appropriate route, chosen from the available routes. "
                                                                    f"If no specific route is confidently matched, select `{self._DEFAULT_FALLBACK_TOKEN}` "
                                                                    "to indicate a fallback to the default route.")),
            "confidence": (float, Field(..., ge=0.0, le=1.0, description="A confidence score (0.0 to 1.0) for the routing decision.")),
            "reasoning": (str, Field(..., description="A brief explanation for why this particular route was chosen.")),
        }

        # Use pydantic.create_model to generate the schema dynamically.
        return create_model("VishustraRoutingDecisionSchema", **dynamic_schema_fields) # type: ignore

    async def route(self, query: str, **llm_kwargs: Any) -> RouterDecision:
        """
        Analyzes a user query using the configured LLM and determines the most suitable
        route based on the defined `RouteConfig`s.

        Args:
            query: The incoming user query or message string.
            **llm_kwargs: Optional keyword arguments to pass directly to the
                          underlying LLMProvider's `generate_structured` method
                          (e.g., `temperature=0.0`, `model="gpt-4"`).

        Returns:
            A `RouterDecision` object containing the chosen route's name,
            the confidence level, and the LLM's reasoning.

        Raises:
            ValueError: If the LLM returns an invalid structured response
                        that does not conform to the expected Pydantic schema.
            RuntimeError: If an unexpected error occurs during the LLM call
                          or processing of its response.
        """
        prompt_components = [
            "You are an intelligent routing system, part of the 'Vishustra' LLM orchestration framework.",
            "Your primary task is to precisely analyze an incoming user query and determine the most appropriate operational route "
            "from a predefined set of options. Your decision must be based solely on the intent expressed in the user query.",
            "",
            "Carefully consider the description and examples for each available route.",
            "Always aim for the highest confidence in your routing decision.",
            "",
            "--- AVAILABLE ROUTES ---",
        ]

        # Add each route's description and examples to the prompt for few-shot learning.
        for name, config in self._routes.items():
            prompt_components.append(f"Route Name: `{config.name}`")
            prompt_components.append(f"Description: {config.description}")
            if config.examples:
                prompt_components.append(f"Examples: {'; '.join(f'\"{ex}\"' for ex in config.examples)}")
            prompt_components.append("---") # Separator for clarity

        if self._default_route_name:
            prompt_components.append(f"If, after careful consideration, you cannot confidently match the query "
                                     f"to any specific route, or if the intent is ambiguous or unknown, "
                                     f"you MUST select the special token `{self._DEFAULT_FALLBACK_TOKEN}`. "
                                     f"This indicates a fallback to the general purpose route '{self._default_route_name}'.")
        else:
            prompt_components.append("It is critical that you always select one of the provided route names, "
                                     "even if confidence is low. Do not indicate a fallback.")

        prompt_components.append("\n--- USER QUERY ---")
        prompt_components.append(f"User Query: \"{query}\"")
        prompt_components.append("\n--- INSTRUCTIONS ---")
        prompt_components.append("Based on the User Query and the Available Routes, determine the single best route.")
        prompt_components.append("Your output must be a JSON object, strictly adhering to the specified Pydantic schema.")

        prompt = "\n".join(prompt_components)
        logger.debug(f"Sending prompt to LLM for routing decision:\n{prompt}")

        try:
            # Call the LLM with the generated prompt and the dynamic schema.
            llm_raw_decision = await self._llm.generate_structured(prompt, self._llm_routing_output_schema, **llm_kwargs)

            # Resolve the special fallback token back to the actual default route name.
            resolved_route_name: Optional[str] = llm_raw_decision.route_name
            if resolved_route_name == self._DEFAULT_FALLBACK_TOKEN:
                if self._default_route_name:
                    resolved_route_name = self._default_route_name
                    logger.info(f"LLM indicated fallback. Routing to configured default route: '{resolved_route_name}'.")
                else:
                    # This case means LLM returned fallback token but no default was configured.
                    # This implies a potential prompt or LLM issue, or misconfiguration.
                    # For robustness, we can try to pick a route, or set to None.
                    logger.warning(f"LLM returned '{self._DEFAULT_FALLBACK_TOKEN}' but no default_route_name was configured. "
                                   "Returning no specific route. Consider configuring a default or refining routes.")
                    resolved_route_name = None # Or pick the most confident non-fallback if LLM provided other scores.

            # Construct the final RouterDecision object.
            # (Note: 'parameters' field is currently empty as parameter extraction
            # is outside the scope of this initial routing implementation).
            return RouterDecision(
                route_name=resolved_route_name,
                confidence=llm_raw_decision.confidence,
                reasoning=llm_raw_decision.reasoning,
                parameters={} # Future extension: populate with extracted parameters if applicable
            )
        except ValidationError as e:
            logger.error(f"LLM output failed Pydantic validation for routing decision: {e.errors()}")
            # Critical error: LLM did not provide a valid structured response.
            raise ValueError("LLM returned an invalid structured response for routing, likely due to schema mismatch.") from e
        except Exception as e:
            logger.error(f"An unexpected error occurred during LLM routing: {type(e).__name__}: {e}")
            raise RuntimeError("Failed to obtain a routing decision from the LLM due to an internal error.") from e