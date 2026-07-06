import logging
from typing import Any, Dict, List

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class IntentClassifierNode(BaseNode):
    """
    A Vishustra processing node that classifies the intent of a given text input.
    This node simulates intent classification based on configurable keyword rules.
    It's designed to identify a primary intent and assign a confidence score.
    """

    def __init__(self,
                 intent_keywords: Dict[str, List[str]] = None,
                 default_unclear_confidence: float = 0.4,
                 default_classified_confidence: float = 0.9):
        """
        Initializes the IntentClassifierNode with a mapping of intents to keywords
        and default confidence scores.

        Args:
            intent_keywords (Dict[str, List[str]], optional): A dictionary where keys are intent names
                                                     and values are lists of keywords/phrases associated with that intent.
                                                     Keywords are case-insensitive. If None, a default set of keywords will be used.
            default_unclear_confidence (float, optional): The confidence score to assign when no clear intent is found.
                                                          Defaults to 0.4.
            default_classified_confidence (float, optional): The confidence score to assign when an intent is classified.
                                                             Defaults to 0.9.
        """
        self._intent_keywords = intent_keywords if intent_keywords is not None else {
            "book_flight": ["book flight", "flight ticket", "travel to", "reservations"],
            "get_weather": ["weather forecast", "how is the weather", "temperature in", "climate"],
            "get_help": ["help me", "support", "customer service", "assistance"],
            "cancel_order": ["cancel my order", "revoke purchase", "return item", "undo transaction"]
        }
        self._default_unclear_confidence = default_unclear_confidence
        self._default_classified_confidence = default_classified_confidence

        # Convert keywords to lowercase for case-insensitive matching and pre-sort by length
        # for more specific matches first.
        self._processed_intent_keywords: List[Tuple[str, str]] = []
        for intent, phrases in self._intent_keywords.items():
            for phrase in phrases:
                self._processed_intent_keywords.append((intent, phrase.lower()))
        self._processed_intent_keywords.sort(key=lambda x: len(x[1]), reverse=True)

        logger.debug(
            f"IntentClassifierNode initialized with {len(self._intent_keywords)} intents. "
            f"Default unclear confidence: {self._default_unclear_confidence}, "
            f"classified confidence: {self._default_classified_confidence}."
        )

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "IntentClassifier"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classifies the intent of the input data, which is expected to be a string query.
        It uses keyword matching to determine the most likely intent.

        Args:
            data (Any): The input data, expected to be a string representing a user query.
            context (Dict[str, Any]): A dictionary containing contextual information.
                                     This node currently does not use context for processing,
                                     but it's available for future enhancements (e.g., dynamic intent loading).

        Returns:
            Dict[str, Any]: A dictionary containing the original query, the classified intent,
                            and a confidence score.
                            Example: {"query": "How's the weather?", "intent": "get_weather", "confidence": 0.9}

        Raises:
            TypeError: If the input 'data' is not a string.
        """
        if not isinstance(data, str):
            error_msg = (
                f"Invalid input data type for IntentClassifierNode. Expected 'str', "
                f"but received '{type(data).__name__}'."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

        original_query = data
        # Process the query for case-insensitive matching
        processed_query = original_query.lower()

        classified_intent = "unclear"
        confidence = self._default_unclear_confidence

        if not processed_query.strip():
            logger.warning("Input query is empty or only whitespace. Classifying as 'empty_query'.")
            return {
                "query": original_query,
                "intent": "empty_query",
                "confidence": self._default_unclear_confidence
            }

        logger.debug(f"Starting intent classification for query: '{original_query}'")

        # Iterate through pre-sorted keywords to find the most specific match first
        for intent, keyword_phrase in self._processed_intent_keywords:
            if keyword_phrase in processed_query:
                classified_intent = intent
                confidence = self._default_classified_confidence
                logger.info(
                    f"Identified intent '{classified_intent}' for query: '{original_query}' "
                    f"using keyword phrase: '{keyword_phrase}'."
                )
                break  # Exit loop on first specific match

        if classified_intent == "unclear":
            logger.info(
                f"Could not clearly identify intent for query: '{original_query}'. "
                f"Classified as '{classified_intent}' with confidence {confidence}."
            )

        return {
            "query": original_query,
            "intent": classified_intent,
            "confidence": confidence
        }

# Example usage (for internal testing, not part of the required output)
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG) # Configure logging for testing
    
    # Mock BaseNode path for standalone testing if needed
    try:
        from vishustra_core.nodes.base_node import BaseNode
    except ImportError:
        # Define a mock BaseNode for local testing if the package isn't installed
        from abc import ABC, abstractmethod
        class BaseNode(ABC):
            @abstractmethod
            def process(self, data: Any, context: Dict[str, Any]) -> Any: pass
            @property
            @abstractmethod
            def node_name(self) -> str: pass
        logger.warning("Could not import BaseNode from vishustra_core.nodes.base_node. Using mock BaseNode for testing.")

    classifier = IntentClassifierNode()

    test_queries = [
        "I need to book a flight to London.",
        "What is the weather forecast for tomorrow?",
        "Can you help me with my account?",
        "I'd like to cancel my order.",
        "Tell me a joke.",
        "",
        "   ",
        123, # Invalid type
        "What's the temperature in Paris?",
        "I want to book flight and cancel my order." # Demonstrates first match wins
    ]

    for query in test_queries:
        try:
            result = classifier.process(query, {})
            logger.info(f"Query: '{query}' -> Result: {result}")
        except TypeError as e:
            logger.error(f"Error processing '{query}': {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred for query '{query}': {e}")

    # Test with custom intent keywords
    custom_classifier = IntentClassifierNode(
        intent_keywords={
            "order_pizza": ["order pizza", "pizza delivery", "hungry for pizza"],
            "get_directions": ["how to get to", "directions to"]
        },
        default_classified_confidence=0.98
    )
    logger.info("\n--- Testing with custom keywords ---")
    custom_queries = [
        "I want to order pizza now.",
        "Can you tell me how to get to the nearest Starbucks?",
        "Where is my package?",
        "Order pizza, please!"
    ]
    for query in custom_queries:
        try:
            result = custom_classifier.process(query, {})
            logger.info(f"Query: '{query}' -> Result: {result}")
        except TypeError as e:
            logger.error(f"Error processing '{query}': {e}")
