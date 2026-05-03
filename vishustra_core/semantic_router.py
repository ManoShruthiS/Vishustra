import asyncio
import numpy as np
import json
import re # Only used in example main block, can be removed if not in production
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError

# --- Framework-level Abstract Base Classes (assumed to exist elsewhere in Vishustra) ---

class BaseEmbeddingModel(ABC):
    """
    Abstract base class for an embedding model within Vishustra.
    Concrete implementations would integrate with various embedding providers
    (e.g., OpenAI, Cohere, local models).
    """
    @abstractmethod
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of texts into vector representations.

        Args:
            texts: A list of strings to embed.

        Returns:
            A list of lists of floats, where each inner list is the embedding
            vector for the corresponding text.
        """
        pass

class BaseLLMClient(ABC):
    """
    Abstract base class for an LLM client within Vishustra.
    Concrete implementations would integrate with various LLM providers
    (e.g., OpenAI, Anthropic, custom local LLMs).
    """
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """
        Generates a response from the LLM based on the prompt.

        Args:
            prompt: The prompt string to send to the LLM.
            **kwargs: Additional parameters specific to the LLM provider
                      (e.g., `temperature`, `max_tokens`, `response_format`).

        Returns:
            The generated response string from the LLM.
        """
        pass

# --- Semantic Router Specific Models and Classes ---

class Route(BaseModel):
    """
    Represents a potential routing destination with example queries.

    Attributes:
        name: A unique identifier for the route (e.g., "customer_service", "payments").
        description: A brief explanation of what this route handles.
        examples: A list of example user queries that should map to this route.
    """
    name: str = Field(..., description="Unique name of the route.")
    description: str = Field(..., description="Brief description of the route's purpose.")
    examples: List[str] = Field(..., description="List of example queries for this route.")

class RouterDecision(BaseModel):
    """
    The result of a routing decision.

    Attributes:
        route_name: The name of the chosen route, or 'no_match' if no route was found.
        confidence: A confidence score (0.0 to 1.0) for the decision.
        reason: An explanation for why this route was chosen.
    """
    route_name: str = Field(..., description="Name of the chosen route or 'no_match'.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for the decision.")
    reason: Optional[str] = Field(None, description="Explanation for the routing decision.")

class BaseSemanticRouter(ABC):
    """
    Abstract base class for a semantic router.
    """
    @abstractmethod
    async def route(self, query: str) -> Optional[RouterDecision]:
        """
        Determines the most appropriate route for a given user query.

        Args:
            query: The user's input query.

        Returns:
            An optional RouterDecision object, or None if no route could be determined.
        """
        pass

class LLMSemanticRouter(BaseSemanticRouter):
    """
    A semantic router that leverages an embedding model for initial similarity matching
    and an optional LLM for nuanced decision-making and disambiguation.

    This router maintains a collection of predefined routes, each with example queries.
    When a new query comes in, it first uses an embedding model to find the most
    semantically similar routes. If an LLM client is provided, it can then use
    the LLM to make a more intelligent decision, especially when multiple routes
    are semantically close or require contextual understanding beyond simple similarity.
    """
    _LLM_DECISION_PROMPT = """
    Given the user query below, select the most appropriate route from the provided options.
    Each route includes a description and example queries that fall under it.
    If none of the routes are a good fit, respond with 'no_match'.

    Your response MUST be a JSON object with a 'route' key and a 'reason' key.
    The 'route' value should be the exact name of one of the available routes or 'no_match'.
    The 'reason' value should be a brief, concise explanation for your choice.

    User Query: "{query}"

    Available Routes:
    {route_options_str}

    Please respond in JSON format:
    {{
        "route": "route_name_here" | "no_match",
        "reason": "explanation here"
    }}
    """

    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        llm_client: Optional[BaseLLMClient] = None,
        similarity_threshold: float = 0.7,
        top_k_for_llm: int = 3,
        llm_supports_json_mode: bool = False
    ):
        """
        Initializes the LLMSemanticRouter.

        Args:
            embedding_model: An instance of BaseEmbeddingModel for semantic similarity.
            llm_client: An optional instance of BaseLLMClient for advanced decision-making.
                        If None, routing relies solely on embedding similarity.
            similarity_threshold: The minimum cosine similarity score to consider a route
                                  a potential match based on embedding similarity alone.
            top_k_for_llm: When using an LLM, the number of top candidate routes (based on
                           embedding similarity) to present to the LLM for final decision.
            llm_supports_json_mode: Set to True if the LLM client supports a native JSON
                                    response format parameter (e.g., OpenAI's 'json_object').
        """
        if not isinstance(embedding_model, BaseEmbeddingModel):
            raise TypeError("embedding_model must be an instance of BaseEmbeddingModel")
        if llm_client is not None and not isinstance(llm_client, BaseLLMClient):
            raise TypeError("llm_client must be an instance of BaseLLMClient or None")

        self._embedding_model = embedding_model
        self._llm_client = llm_client
        self._similarity_threshold = similarity_threshold
        self._top_k_for_llm = top_k_for_llm
        self._llm_supports_json_mode = llm_supports_json_mode

        # Stores pre-computed embeddings and their associated route names and example texts
        # Format: [(route_name, example_text), ...]
        self._all_route_examples_meta: List[Tuple[str, str]] = []
        # Stores numpy array of all example embeddings for efficient similarity search
        self._all_example_embeddings: Optional[np.ndarray] = None
        # Maps route_name to the full Route object for accessing descriptions etc.
        self._routes_map: Dict[str, Route] = {}

    async def add_routes(self, routes: List[Route]):
        """
        Adds a list of routes to the router. This method will pre-compute
        embeddings for all example queries within each route.

        Args:
            routes: A list of Route objects to add.
        """
        if not routes:
            return

        new_example_texts: List[str] = []
        new_example_meta: List[Tuple[str, str]] = [] # (route_name, example_text)

        for route in routes:
            if route.name in self._routes_map:
                print(f"Warning: Route with name '{route.name}' already exists. Overwriting definition.")
            self._routes_map[route.name] = route
            
            for example in route.examples:
                new_example_texts.append(example)
                new_example_meta.append((route.name, example))

        if not new_example_texts:
            return

        # Embed all new examples in a single call for efficiency
        new_embeddings = await self._embedding_model.embed(new_example_texts)
        if not new_embeddings:
            raise ValueError("Embedding model returned empty embeddings for new routes.")

        new_embeddings_np = np.array(new_embeddings)

        # Append new embeddings and metadata
        if self._all_example_embeddings is None or self._all_example_embeddings.size == 0:
            self._all_example_embeddings = new_embeddings_np
        else:
            self._all_example_embeddings = np.vstack(
                (self._all_example_embeddings, new_embeddings_np)
            )
        self._all_route_examples_meta.extend(new_example_meta)

    @staticmethod
    def _calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculates cosine similarity between two numpy vectors."""
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0 # Handle zero vectors gracefully
        return np.dot(vec1, vec2) / (norm_vec1 * norm_vec2)

    async def route(self, query: str) -> Optional[RouterDecision]:
        """
        Determines the most appropriate route for a given user query.

        Args:
            query: The user's input query.

        Returns:
            An optional RouterDecision object, or None if no route could be determined.
        """
        if self._all_example_embeddings is None or len(self._all_route_examples_meta) == 0:
            print("Warning: No routes have been added to the router. Cannot route query.")
            return None

        query_embedding_list = await self._embedding_model.embed([query])
        if not query_embedding_list:
            print("Error: Embedding model returned empty embedding for the query.")
            return None
        query_embedding = np.array(query_embedding_list[0])

        # Calculate similarity with all stored example embeddings
        similarities = [
            self._calculate_cosine_similarity(query_embedding, ex_emb)
            for ex_emb in self._all_example_embeddings
        ]
        
        # Get indices of top K most similar *examples*
        # Use min(len, top_k) to handle cases where there are fewer examples than top_k
        num_candidates_to_consider = min(len(similarities), self._top_k_for_llm * 2) # Get more examples than routes
        top_k_example_indices = np.argsort(similarities)[::-1][:num_candidates_to_consider]

        # Aggregate similarities by route name. Keep track of the highest similarity score
        # for a route and a few top examples for that route.
        candidate_routes_data: Dict[str, Tuple[float, List[str]]] = {} # {route_name: (max_similarity, [top_examples_text])}
        
        for i in top_k_example_indices:
            route_name, example_text = self._all_route_examples_meta[i]
            similarity_score = similarities[i]

            current_max_sim, current_examples = candidate_routes_data.get(route_name, (0.0, []))

            if similarity_score > current_max_sim:
                candidate_routes_data[route_name] = (similarity_score, [example_text]) # Reset examples if new max
            elif len(current_examples) < 2: # Keep up to 2 examples per route
                if example_text not in current_examples:
                    current_examples.append(example_text)
                    candidate_routes_data[route_name] = (current_max_sim, current_examples)

        # Sort candidate routes by their highest example similarity score for presentation to LLM
        # and for fallback similarity-only routing
        sorted_candidates = sorted(candidate_routes_data.items(), key=lambda item: item[1][0], reverse=True)[:self._top_k_for_llm]

        if not sorted_candidates:
            return None # No candidates found based on similarity

        # --- Decision Making Strategy ---
        if self._llm_client:
            return await self._route_with_llm(query, sorted_candidates)
        else:
            return self._route_with_similarity_only(sorted_candidates)

    def _route_with_similarity_only(self, sorted_candidates: List[Tuple[str, Tuple[float, List[str]]]]) -> Optional[RouterDecision]:
        """
        Routes the query based solely on embedding similarity if no LLM client is available.
        """
        if not sorted_candidates:
            return None

        top_route_name, (top_similarity, top_examples) = sorted_candidates[0]

        if top_similarity >= self._similarity_threshold:
            return RouterDecision(
                route_name=top_route_name,
                confidence=top_similarity,
                reason=f"Highest embedding similarity ({top_similarity:.2f}) to example '{top_examples[0]}'."
            )
        else:
            return None # No route met the similarity threshold

    async def _route_with_llm(self, query: str, sorted_candidates: List[Tuple[str, Tuple[float, List[str]]]]) -> Optional[RouterDecision]:
        """
        Routes the query using the LLM for advanced decision-making, considering top semantic candidates.
        """
        if not self._llm_client:
            raise RuntimeError("LLM client not provided for LLM-based routing.")

        route_options_str_parts = []
        for route_name, (similarity, examples) in sorted_candidates:
            route_obj = self._routes_map.get(route_name)
            if not route_obj:
                continue # This should ideally not happen if data integrity is maintained

            route_options_str_parts.append(f"Route Name: \"{route_name}\" (Similarity Score: {similarity:.2f})")
            route_options_str_parts.append(f"Description: {route_obj.description}")
            route_options_str_parts.append(f"Examples: {', '.join(f'\"{ex}\"' for ex in examples)}")
            route_options_str_parts.append("") # Newline for separation

        route_options_str = "\n".join(route_options_str_parts).strip()

        llm_prompt = self._LLM_DECISION_PROMPT.format(
            query=query,
            route_options_str=route_options_str
        )
        
        llm_kwargs = {"temperature": 0.0} # Keep LLM deterministic for routing
        if self._llm_supports_json_mode:
            llm_kwargs["response_format"] = {"type": "json_object"}

        try:
            llm_response_str = await self._llm_client.generate(prompt=llm_prompt, **llm_kwargs)
            llm_decision_data = json.loads(llm_response_str)

            chosen_route_name = llm_decision_data.get("route")
            reason = llm_decision_data.get("reason")

            if chosen_route_name and chosen_route_name != "no_match" and chosen_route_name in self._routes_map:
                # Determine confidence: use the highest similarity score for the chosen route among candidates.
                # If LLM chooses a route not in top_k_for_llm, we can still use a base confidence.
                confidence = 0.0
                for r_name, (sim, _) in sorted_candidates:
                    if r_name == chosen_route_name:
                        confidence = sim
                        break
                confidence = max(confidence, self._similarity_threshold) # Ensure minimum threshold if LLM chose it.

                return RouterDecision(
                    route_name=chosen_route_name,
                    confidence=confidence,
                    reason=reason or "LLM selected this route based on contextual understanding."
                )
            elif chosen_route_name == "no_match":
                 return RouterDecision(
                    route_name="no_match",
                    confidence=0.0, # No match typically implies very low confidence
                    reason=reason or "LLM determined no suitable route among the options."
                )
            else:
                # LLM returned an invalid route name or unexpected format, fallback
                print(f"Warning: LLM returned an invalid route '{chosen_route_name}' or bad format. Falling back to similarity-only decision.")
                return self._route_with_similarity_only(sorted_candidates)

        except (json.JSONDecodeError, ValidationError) as e:
            print(f"Warning: Failed to parse LLM response or validate decision: {e}. Raw response: '{llm_response_str}'. Falling back to similarity-only decision.")
            return self._route_with_similarity_only(sorted_candidates)
        except Exception as e:
            print(f"Error during LLM routing: {e}. Falling back to similarity-only decision.")
            return self._route_with_similarity_only(sorted_candidates)


# --- Example Usage (for demonstration purposes) ---
if __name__ == "__main__":
    import asyncio
    
    # Mock implementations for demonstration purposes
    class MockEmbeddingModel(BaseEmbeddingModel):
        async def embed(self, texts: List[str]) -> List[List[float]]:
            # A deterministic, simple mock embedding for testing similarity
            embeddings = []
            for text in texts:
                char_sum = sum(ord(c) for c in text)
                # Create a 3-dim vector based on text properties, then normalize
                vec = [
                    len(text) * 0.1,
                    (char_sum % 100) * 0.01,
                    ((len(text) + char_sum) % 50) * 0.02
                ]
                np_vec = np.array(vec)
                norm = np.linalg.norm(np_vec)
                embeddings.append((np_vec / (norm if norm != 0 else 1.0)).tolist())
            return embeddings

    class MockLLMClient(BaseLLMClient):
        async def generate(self, prompt: str, **kwargs) -> str:
            print(f"\n--- Mock LLM Call (Truncated Prompt) ---\nPrompt: {prompt[:400]}...") # Truncate long prompts
            
            # Simulate LLM decision based on prompt content using simple keyword checks
            if "refund" in prompt.lower() and "customer service" in prompt.lower():
                return '{"route": "customer_service", "reason": "Query about refund requires customer service."}'
            elif "billing" in prompt.lower() and "payment" in prompt.lower():
                 return '{"route": "payments", "reason": "Query clearly about billing and payments."}'
            elif "technical issue" in prompt.lower() or "bug" in prompt.lower() or "crashed" in prompt.lower():
                 return '{"route": "technical_support", "reason": "User is reporting a technical problem."}'
            elif "product features" in prompt.lower() or "what can it do" in prompt.lower() or "capabilities" in prompt.lower():
                 return '{"route": "product_info", "reason": "User is asking about product capabilities."}'
            elif "no_match" in prompt.lower(): # If prompt hints no_match from low similarity
                return '{"route": "no_match", "reason": "No clear route based on the options provided."}'
            else:
                # If LLM doesn't have a specific rule, try to pick the highest similarity one from prompt
                if "Available Routes:" in prompt:
                    routes_section_match = re.search(r'Available Routes:(.*?)\nPlease respond in JSON format:', prompt, re.DOTALL)
                    if routes_section_match:
                        routes_section = routes_section_match.group(1)
                        first_route_name_match = re.search(r'Route Name: "([^"]+)"', routes_section)
                        if first_route_name_match:
                            chosen_route = first_route_name_match.group(1)
                            return f'{{"route": "{chosen_route}", "reason": "LLM picked the first highly similar candidate from the provided list."}}'
                return '{"route": "no_match", "reason": "LLM could not confidently determine a route based on ambiguous input."}'


    async def main():
        embedding_model = MockEmbeddingModel()
        llm_client = MockLLMClient()

        # Define some routes for the router
        routes = [
            Route(
                name="customer_service",
                description="Handles all customer inquiries, complaints, and refunds.",
                examples=["I need a refund", "My order is delayed", "How do I contact support?", "I have a complaint about my service"]
            ),
            Route(
                name="payments",
                description="Manages billing, invoices, and payment method updates.",
                examples=["How do I pay my bill?", "Where can I see my invoices?", "Update payment method", "Payment failed for my subscription"]
            ),
            Route(
                name="product_info",
                description="Provides information about Vishustra's features, capabilities, and usage.",
                examples=["What can Vishustra do?", "How to use agent tools?", "Explain LLM orchestration", "Product capabilities and integrations"]
            ),
            Route(
                name="technical_support",
                description="Assists with technical issues, bugs, and system outages affecting Vishustra.",
                examples=["I found a bug in the API", "The system is down", "Technical issue with the dashboard", "Error message X when deploying"]
            )
        ]

        # --- Test Similarity-only Router ---
        print("\n--- Testing Similarity-only Router ---")
        router_similarity_only = LLMSemanticRouter(
            embedding_model=embedding_model,
            similarity_threshold=0.6 # Lower threshold for mock
        )
        await router_similarity_only.add_routes(routes)

        queries_similarity = [
            "I want my money back",
            "Where is my invoice?",
            "Tell me about the new features",
            "My account is not loading, it crashed",
            "Hello, how are you today?", # Should be no_match or low confidence
            "How do I update my credit card information?"
        ]

        for query in queries_similarity:
            decision = await router_similarity_only.route(query)
            print(f"\nQuery: '{query}'")
            if decision:
                print(f"  Decision (Similarity): Route='{decision.route_name}', Confidence={decision.confidence:.2f}, Reason: {decision.reason}")
            else:
                print("  Decision (Similarity): No matching route found.")

        # --- Test LLM-enhanced Router ---
        print("\n--- Testing LLM-enhanced Router ---")
        router_with_llm = LLMSemanticRouter(
            embedding_model=embedding_model,
            llm_client=llm_client,
            similarity_threshold=0.6,
            top_k_for_llm=2, # Present top 2 candidate routes to LLM
            llm_supports_json_mode=False # Mock LLM doesn't need native JSON mode
        )
        await router_with_llm.add_routes(routes) # Add routes again

        queries_llm = [
            "I need to talk to someone about getting a refund for a recent service outage.", # Combines customer_service and tech_support
            "What's the process for changing my payment details and seeing past bills?", # Combines payments and product_info slightly
            "My application crashed, what's wrong? I see an error 500.", # Clearly technical
            "Can Vishustra integrate with custom vector databases and what are its capabilities?", # Product info
            "What is the current weather in London?", # Definitely no_match
            "I have a general query about an API, where should I go?" # Could be tech_support or product_info, LLM disambiguates
        ]

        for query in queries_llm:
            decision = await router_with_llm.route(query)
            print(f"\nQuery: '{query}'")
            if decision:
                print(f"  Decision (LLM Enhanced): Route='{decision.route_name}', Confidence={decision.confidence:.2f}, Reason: {decision.reason}")
            else:
                print("  Decision (LLM Enhanced): No matching route found.")
    
    asyncio.run(main())