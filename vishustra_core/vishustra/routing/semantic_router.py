import numpy as np
from typing import List, Dict, Any, Protocol, Optional, Self
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

# --- Vishustra Core Interfaces (simplified for standalone demonstration) ---
# In a full Vishustra project, these would typically reside in `vishustra.core.embeddings.base`
# and `vishustra.core.utils.similarity_metrics` modules to maintain modularity.

class Embedder(Protocol):
    """
    Protocol for an embedding model within Vishustra.
    Concrete implementations would wrap various embedding providers (e.g., OpenAI, HuggingFace
    local models, custom internal models).
    """
    def embed_text(self, text: str) -> List[float]:
        """
        Embeds a single string of text into a list of floats (vector representation).
        Implementations should handle batching internally for efficiency if needed.

        Args:
            text: The input string to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        ...

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of strings into a list of vectors. This method is often
        more efficient for providers that support batch processing.

        Args:
            texts: A list of input strings to embed.

        Returns:
            A list of lists of floats, where each inner list is the embedding
            vector for the corresponding input string.
        """
        ...

# --- Semantic Router Components ---

@dataclass(frozen=True)
class Route:
    """
    Represents a predefined route or destination within Vishustra's orchestration framework.
    Each route has a unique name, a descriptive text, and a list of example phrases
    that semantically align with this route.

    The internal `_embedding` attribute stores the pre-computed average embedding
    of all examples (and description), used for efficient similarity lookups by
    the `SemanticRouter`. This attribute is populated by the `SemanticRouter`
    upon registration and is hidden from direct instantiation to ensure consistency.
    """
    name: str
    description: str
    examples: List[str]
    _embedding: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        """
        Initializes the _embedding attribute to an empty numpy array after the dataclass
        is created. It is later populated by the SemanticRouter.
        We use object.__setattr__ because the dataclass is frozen.
        """
        object.__setattr__(self, '_embedding', np.array([]))

    def _set_embedding(self, embedding: np.ndarray):
        """
        Internal method to set the pre-computed embedding for this route.
        This method is exclusively for use by the `SemanticRouter` to ensure
        the embedding is correctly generated and assigned.

        Args:
            embedding: A 1D numpy array representing the average embedding of the route.

        Raises:
            ValueError: If the provided embedding is not a 1D numpy array.
        """
        if not isinstance(embedding, np.ndarray) or embedding.ndim != 1:
            raise ValueError("Embedding must be a 1D numpy array.")
        object.__setattr__(self, '_embedding', embedding)

    @property
    def embedding(self) -> np.ndarray:
        """
        Returns the pre-computed embedding for this route.
        Raises an AttributeError if the embedding has not yet been initialized by a router.
        """
        if self._embedding.size == 0:
            raise AttributeError(f"Route '{self.name}' embedding has not been initialized. "
                                 "Ensure it's registered with a SemanticRouter.")
        return self._embedding

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculates the cosine similarity between two 1D numpy arrays.
    Cosine similarity measures the cosine of the angle between two vectors.
    A score of 1 indicates identical direction, -1 indicates opposite, 0 indicates orthogonality.

    Args:
        vec1: The first vector (1D numpy array).
        vec2: The second vector (1D numpy array).

    Returns:
        The cosine similarity score, ranging from -1.0 to 1.0.
        Returns 0.0 if either vector has zero magnitude to prevent division by zero.
    """
    if vec1.size == 0 or vec2.size == 0:
        logger.warning("Attempted to compute similarity with an empty vector. Returning 0.0.")
        return 0.0

    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    if norm_vec1 == 0 or norm_vec2 == 0:
        logger.warning("Attempted to compute similarity with a zero-magnitude vector. Returning 0.0.")
        return 0.0

    return float(dot_product / (norm_vec1 * norm_vec2))

class SemanticRouter:
    """
    The Vishustra SemanticRouter orchestrates the routing of incoming text queries
    to the most semantically similar predefined 'Route' using an embedding model.

    This component is crucial for building highly modular LLM applications where
    different user intents can dynamically trigger distinct downstream processes,
    such as invoking specific agents, tools, memory modules, or even different LLM chains.
    """
    def __init__(self, embedder: Embedder, routes: Optional[List[Route]] = None,
                 similarity_threshold: float = 0.75):
        """
        Initializes the SemanticRouter.

        Args:
            embedder: An instance of an `Embedder` implementation (e.g., `OpenAIEmbedder`,
                      `LocalHFEmbedder`). This is used to generate embeddings for both
                      the predefined routes and incoming user queries.
            routes: An optional list of `Route` objects to register upon initialization.
            similarity_threshold: The minimum cosine similarity score required for a
                                  query to be considered a match for a route. If no
                                  route meets this threshold, the `route` method will
                                  return `None`. Defaults to 0.75.
        """
        if not isinstance(embedder, Embedder):
            raise TypeError("Embedder must implement the Embedder protocol.")
        self.embedder: Embedder = embedder
        self._routes: Dict[str, Route] = {}
        self.similarity_threshold: float = similarity_threshold

        if routes:
            self.add_routes(routes)

        logger.info(f"SemanticRouter initialized with {len(self._routes)} routes "
                    f"and similarity threshold: {self.similarity_threshold:.2f}.")

    def add_route(self, route: Route) -> Self:
        """
        Adds a single `Route` to the router. The examples and description within the route
        are embedded and averaged to create a representative embedding for the route,
        which is then stored internally for efficient similarity comparisons.

        Args:
            route: The `Route` object to add to the router.

        Returns:
            The `SemanticRouter` instance, allowing for method chaining.

        Raises:
            ValueError: If a route with the same name already exists in the router.
        """
        if route.name in self._routes:
            raise ValueError(f"Route with name '{route.name}' already exists.")

        # Embed all examples and the description to get a robust representation
        texts_to_embed = []
        if route.examples:
            texts_to_embed.extend(route.examples)
        texts_to_embed.append(route.description) # Always include description for context

        if not texts_to_embed:
            logger.warning(f"Route '{route.name}' has no examples or description. "
                           "Cannot create a meaningful embedding.")
            # Create a placeholder or raise an error depending on desired behavior
            route._set_embedding(np.zeros(self._get_embedding_dimension())) # Assuming a way to get dim
        else:
            try:
                example_embeddings_list = self.embedder.embed_texts(texts_to_embed)
                if not example_embeddings_list:
                    raise RuntimeError("Embedder returned empty list.")
                avg_embedding = np.mean(example_embeddings_list, axis=0)
                route._set_embedding(avg_embedding) # Use the internal setter to populate the frozen dataclass
            except Exception as e:
                logger.error(f"Failed to embed examples for route '{route.name}': {e}")
                # Set a dummy embedding to prevent errors later, but route will likely fail to match
                route._set_embedding(np.zeros(self._get_embedding_dimension()))

        self._routes[route.name] = route
        logger.debug(f"Route '{route.name}' added successfully with {len(texts_to_embed)} texts embedded.")
        return self

    def add_routes(self, routes: List[Route]) -> Self:
        """
        Adds multiple `Route` objects to the router.

        Args:
            routes: A list of `Route` objects to add.

        Returns:
            The `SemanticRouter` instance, allowing for method chaining.
        """
        for route in routes:
            self.add_route(route)
        return self

    def remove_route(self, route_name: str) -> Optional[Route]:
        """
        Removes a route by its unique name.

        Args:
            route_name: The name of the route to remove.

        Returns:
            The `Route` object that was removed if found, otherwise `None`.
        """
        if route_name in self._routes:
            route = self._routes.pop(route_name)
            logger.debug(f"Route '{route_name}' removed successfully.")
            return route
        logger.warning(f"Attempted to remove non-existent route: '{route_name}'.")
        return None

    def _get_embedding_dimension(self) -> int:
        """
        Helper to infer embedding dimension, useful for creating dummy vectors.
        In a real scenario, the embedder might expose this directly.
        """
        if self._routes:
            for route in self._routes.values():
                if route.embedding.size > 0:
                    return route.embedding.shape[0]
        # Fallback if no routes exist or no embeddings are set yet
        # This is a heuristic; real embedders usually have a fixed dimension.
        # For a mock, 3 is sufficient. For real systems, it could be 768, 1536, etc.
        logger.warning("Could not determine embedding dimension from existing routes. Defaulting to 3.")
        return 3 # Default to a small arbitrary number for the example mock.

    def route(self, query: str) -> Optional[Route]:
        """
        Determines the most semantically relevant route for a given input query.
        The query is embedded, and its similarity to all registered route embeddings
        is calculated using cosine similarity. The route with the highest similarity
        score above the `similarity_threshold` is returned.

        Args:
            query: The incoming text query from the user or another system component.

        Returns:
            The best matching `Route` object if its similarity score meets or exceeds
            the configured `similarity_threshold`, otherwise `None`.
        """
        if not self._routes:
            logger.warning("No routes configured in SemanticRouter. Cannot route query. Returning None.")
            return None

        try:
            query_embedding_list = self.embedder.embed_text(query)
            if not query_embedding_list:
                logger.error("Embedder returned an empty embedding for the query. Cannot route.")
                return None
            query_embedding = np.array(query_embedding_list)
        except Exception as e:
            logger.error(f"Failed to embed the query '{query[:100]}...': {e}. Cannot route.")
            return None

        best_route: Optional[Route] = None
        max_similarity: float = -1.0

        for route_name, route_obj in self._routes.items():
            try:
                if route_obj.embedding.size == 0:
                    logger.warning(f"Skipping route '{route_name}' due to uninitialized or empty embedding.")
                    continue

                similarity = cosine_similarity(query_embedding, route_obj.embedding)
                logger.debug(f"Query '{query[:50]}...' vs Route '{route_name}': Sim={similarity:.4f}")

                if similarity > max_similarity:
                    max_similarity = similarity
                    best_route = route_obj
            except AttributeError as ae:
                logger.error(f"Error accessing embedding for route '{route_name}': {ae}. Skipping.")
            except Exception as e:
                logger.error(f"Unexpected error during similarity calculation for route '{route_name}': {e}. Skipping.")


        if best_route and max_similarity >= self.similarity_threshold:
            logger.info(f"Query routed to '{best_route.name}' with similarity {max_similarity:.4f} "
                        f"(threshold: {self.similarity_threshold:.2f}).")
            return best_route
        else:
            if best_route:
                logger.info(f"No route found above threshold {self.similarity_threshold:.2f}. "
                            f"Best match was '{best_route.name}' with similarity {max_similarity:.4f}.")
            else:
                logger.info(f"No suitable route found for query '{query[:50]}...'. No routes matched.")
            return None

# --- Example Usage (for testing and demonstration purposes) ---
if __name__ == "__main__":
    import sys
    # Configure basic logging for the example to demonstrate internal workings
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Set this file's logger to DEBUG for more detailed output during routing
    logging.getLogger(__name__).setLevel(logging.DEBUG)

    # A mock embedder for demonstration. In a real Vishustra setup, this would be a
    # concrete implementation wrapping a real embedding API (e.g., OpenAI, Cohere)
    # or a local HuggingFace model.
    class MockEmbedder(Embedder):
        """
        A simple mock embedder that returns fixed vectors for known keywords.
        Simulates distinct semantic spaces for different intents.
        """
        _predefined_vectors = {
            "travel": np.array([0.9, 0.1, 0.1], dtype=np.float32),
            "weather": np.array([0.1, 0.9, 0.1], dtype=np.float32),
            "scheduling": np.array([0.1, 0.1, 0.9], dtype=np.float32),
            "general": np.array([0.5, 0.5, 0.5], dtype=np.float32) # Neutral/broad intent
        }
        _unknown_count = 0

        def embed_text(self, text: str) -> List[float]:
            text_lower = text.lower()
            if "flight" in text_lower or "travel" in text_lower or "ticket" in text_lower or "london" in text_lower or "paris" in text_lower:
                return self._predefined_vectors["travel"].tolist()
            if "weather" in text_lower or "climate" in text_lower or "rain" in text_lower or "forecast" in text_lower or "berlin" in text_lower:
                return self._predefined_vectors["weather"].tolist()
            if "schedule" in text_lower or "meeting" in text_lower or "appointment" in text_lower or "calendar" in text_lower:
                return self._predefined_vectors["scheduling"].tolist()
            if "hello" in text_lower or "hi" in text_lower or "joke" in text_lower or "capital" in text_lower or "vishustra" in text_lower or "chat" in text_lower:
                 return self._predefined_vectors["general"].tolist()
            if "task" in text_lower or "todo" in text_lower or "list" in text_lower or "groceries" in text_lower or "report" in text_lower:
                 # Generate a slightly varied vector for a new domain to avoid direct hit on 'general'
                 self._unknown_count += 1
                 # Blend with general, slightly push towards a 'task' like vector
                 return (self._predefined_vectors["general"] * 0.7 + np.array([0.2, 0.4, 0.6]) * 0.3 +
                         np.random.rand(3) * 0.05).tolist() # Add some noise

            logger.warning(f"MockEmbedder: Unknown text pattern '{text}'. Using general vector with noise.")
            # Add some slight noise to differentiate unknown general queries
            return (self._predefined_vectors["general"] + np.random.rand(3) * 0.01).tolist()

        def embed_texts(self, texts: List[str]) -> List[List[float]]:
            return [self.embed_text(text) for text in texts]

    mock_embedder = MockEmbedder()

    # Define various routes with descriptive names, descriptions, and example queries.
    # The examples help the router learn the semantic space of each route.
    flight_booking_route = Route(
        name="flight_booking_agent",
        description="Handles user requests related to booking flights and general travel arrangements.",
        examples=[
            "I need to book a flight to London.",
            "Find me a plane ticket.",
            "Can you help me with travel plans for my vacation?",
            "Book a flight from New York to Paris next month.",
            "What's the best time to fly to Tokyo?"
        ]
    )

    weather_query_route = Route(
        name="weather_information_tool",
        description="Provides current weather conditions and forecasts for any location.",
        examples=[
            "What's the weather like today in Berlin?",
            "Will it rain tomorrow in Seattle?",
            "Give me the weather forecast for next week in London.",
            "How hot is it in Miami right now?",
            "Is it sunny in Rome?"
        ]
    )

    meeting_scheduler_route = Route(
        name="meeting_scheduler_service",
        description="Assists with scheduling meetings, setting reminders, and managing calendar events.",
        examples=[
            "Schedule a meeting for me next Tuesday at 10 AM.",
            "Set up a call with John at 3 PM about the new project.",
            "Can you add an event to my calendar for the product launch?",
            "Reschedule my appointment with Dr. Smith.",
            "Remind me about the team sync."
        ]
    )

    general_chatbot_route = Route(
        name="general_chatbot",
        description="A versatile chatbot for casual conversation, general knowledge queries, and fallback.",
        examples=[
            "Hello, how are you doing today?",
            "Tell me a joke.",
            "What is the capital of France?",
            "Who are you, and what can you do?",
            "Tell me about Vishustra, your framework.",
            "Hi there, I just want to chat."
        ]
    )

    # Initialize the router with the mock embedder and initial routes.
    # A slightly lower similarity threshold is chosen to accommodate the simplicity of the MockEmbedder.
    router = SemanticRouter(
        embedder=mock_embedder,
        routes=[
            flight_booking_route,
            weather_query_route,
            meeting_scheduler_route,
            general_chatbot_route
        ],
        similarity_threshold=0.70 # Adjust based on actual embedder performance
    )

    print("\n--- Testing Initial Routing ---")

    test_queries = [
        "I want to book a flight to Tokyo next month.",             # -> flight_booking_agent
        "What's the weather going to be like in New York City tomorrow?", # -> weather_information_tool
        "Can you schedule a meeting for me with the team next week?", # -> meeting_scheduler_service
        "Tell me something interesting about space.",               # -> general_chatbot
        "How do I book a hotel?",                                  # -> flight_booking_agent (travel related)
        "What's the temperature outside?",                          # -> weather_information_tool
        "Set up an appointment for me for a dental checkup.",      # -> meeting_scheduler_service
        "Hi there, can we chat for a bit?",                        # -> general_chatbot
        "I need to buy a train ticket for my trip.",              # -> flight_booking_agent (travel related, though train)
        "Who built you?",                                          # -> general_chatbot
        "I need to cancel my flight."                              # -> flight_booking_agent
    ]

    for i, query in enumerate(test_queries):
        print(f"\nQuery {i+1}: '{query}'")
        matched_route = router.route(query)
        if matched_route:
            print(f"  --> Matched Route: '{matched_route.name}'")
            print(f"      Description: {matched_route.description}")
        else:
            print("  --> No suitable route found.")

    print("\n--- Dynamically Adding/Removing Routes ---")

    # Add a new route for task management
    new_task_route = Route(
        name="task_management_tool",
        description="Manages user tasks, to-do lists, and reminders for specific actions.",
        examples=[
            "Add 'buy groceries' to my to-do list.",
            "What are my pending tasks for this week?",
            "Mark 'finish report' as complete.",
            "Remind me to call mom at 5 PM.",
            "Create a new task."
        ]
    )
    router.add_route(new_task_route)
    print(f"Router now has {len(router._routes)} routes after adding 'task_management_tool'.")

    query_new = "Can you add 'send email to boss' to my task list?"
    print(f"\nQuery: '{query_new}'")
    matched_route = router.route(query_new)
    if matched_route:
        print(f"  --> Matched Route: '{matched_route.name}'")
    else:
        print("  --> No suitable route found.")

    # Remove the newly added route
    removed_route = router.remove_route("task_management_tool")
    if removed_route:
        print(f"Removed route: '{removed_route.name}'.")
    print(f"Router now has {len(router._routes)} routes after removal.")

    # Test the same query again; it should no longer match the removed route
    print(f"\nQuery: '{query_new}' (after 'task_management_tool' removal)")
    matched_route = router.route(query_new)
    if matched_route:
        print(f"  --> Matched Route: '{matched_route.name}'")
    else:
        print("  --> No suitable route found.") # Expected output

    # Test behavior with no routes configured in a new router instance
    empty_router = SemanticRouter(mock_embedder, routes=[], similarity_threshold=0.7)
    print("\n--- Testing empty router ---")
    matched = empty_router.route("hello world, what's up?")
    print(f"Result from empty router: {matched} (Expected: None)")

    # Test with a very low similarity threshold to ensure all routes are considered
    router_low_threshold = SemanticRouter(
        embedder=mock_embedder,
        routes=[flight_booking_route, weather_query_route],
        similarity_threshold=0.1
    )
    print("\n--- Testing with very low similarity threshold ---")
    matched_low_threshold = router_low_threshold.route("Completely unrelated query string.")
    if matched_low_threshold:
        print(f"  --> Matched Route with low threshold: '{matched_low_threshold.name}'")
    else:
        print("  --> No route matched even with low threshold.")
