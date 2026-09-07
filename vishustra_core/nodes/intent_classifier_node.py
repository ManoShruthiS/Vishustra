import logging
from typing import Any, Dict, List

# Simulate the project's base node import path
# In a real project, this would be available from the installed package.
try:
    from vishustra_core.nodes.base_node import BaseNode
except ImportError:
    # This block allows the file to be runnable or testable even if the full vishustra_core
    # is not yet installed or available in the path, by providing a mock BaseNode.
    # In a production Vishustra environment, this ImportError should not happen.
    logging.warning(
        "Could not import BaseNode from vishustra_core.nodes.base_node. "
        "Using a mock BaseNode for development/testing purposes. "
        "Ensure vishustra_core is correctly installed and configured for production use."
    )
    from abc import ABC, abstractmethod
    class BaseNode(ABC):
        @abstractmethod
        def process(self, data: Any, context: Dict[str, Any]) -> Any:
            pass
        @property
        @abstractmethod
        def node_name(self) -> str:
            pass

logger = logging.getLogger(__name__)

class IntentClassifierNode(BaseNode):
    """
    A Vishustra node that classifies the intent of a given user query.

    This node takes a string representing a user's natural language query
    and attempts to categorize it into one of several predefined intents
    based on keyword matching. It's configurable with a dictionary
    of intent patterns.

    Inputs:
        data (str): The user query string to classify.
        context (Dict[str, Any]): A dictionary containing contextual information
                                   (not directly used for classification in this
                                   implementation but available for future extensions).

    Outputs:
        Dict[str, Any]: A dictionary containing:
            'query' (str): The original input query.
            'intent' (str): The classified intent (e.g., "GREETING", "WEATHER_QUERY", "UNKNOWN").

    Configuration (via __init__):
        intent_patterns (Dict[str, List[str]]): An optional dictionary where keys are
            intent names (str) and values are lists of keywords/phrases (str)
            that, if found in the query, trigger that intent.
            Defaults to a set of common patterns if not provided.
    """

    def __init__(self, intent_patterns: Dict[str, List[str]] = None):
        """
        Initializes the IntentClassifierNode with a set of intent patterns.

        Args:
            intent_patterns (Dict[str, List[str]], optional): Custom intent patterns.
                Defaults to a predefined set if None.
        """
        # Default patterns for common intents if not provided by the user
        self._intent_patterns = intent_patterns if intent_patterns is not None else {
            "GREETING": ["hello", "hi", "hey", "good morning", "good evening", "greetings"],
            "WEATHER_QUERY": ["weather", "forecast", "temperature", "how cold", "how hot", "is it raining"],
            "BOOK_FLIGHT": ["book flight", "flights to", "fly me to", "schedule a flight"],
            "CREATE_REMINDER": ["remind me", "set a reminder", "create reminder", "my reminder"],
            "PLAY_MUSIC": ["play music", "music for", "song by", "play the song", "music please"],
            "SEARCH_INFO": ["what is", "who is", "tell me about", "define", "information on"],
            "HELP_REQUEST": ["help", "support", "assist me", "can you help"],
            "FAREWELL": ["bye", "goodbye", "see you", "farewell"],
        }
        logger.debug(f"IntentClassifierNode initialized with {len(self._intent_patterns)} intent patterns.")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "IntentClassifier"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to classify its intent.

        Args:
            data (Any): The user query, expected to be a string.
            context (Dict[str, Any]): A dictionary for contextual information.

        Returns:
            Dict[str, Any]: A dictionary containing the original query and the classified intent.

        Raises:
            TypeError: If the input 'data' is not a string.
        """
        if not isinstance(data, str):
            logger.error(
                f"IntentClassifierNode received invalid data type. "
                f"Expected 'str', got '{type(data).__name__}'."
            )
            raise TypeError(f"{self.node_name} expects string input data, but received {type(data).__name__}.")

        query = data.strip()
        if not query:
            logger.warning(f"Received empty query for intent classification in {self.node_name}.")
            return {"query": query, "intent": "NO_INPUT"}

        normalized_query = query.lower()
        classified_intent = "UNKNOWN"

        # Iterate through predefined intent patterns to find a match
        for intent, patterns in self._intent_patterns.items():
            for pattern in patterns:
                # Simple keyword/phrase matching for simulation.
                # In a more advanced implementation, this could use regex,
                # NLP libraries, or call an external ML model.
                if pattern.lower() in normalized_query:
                    classified_intent = intent
                    logger.debug(f"Query '{query}' matched pattern '{pattern}' for intent '{intent}'.")
                    break  # Found a match for this intent, no need to check further patterns for it
            if classified_intent != "UNKNOWN":
                break  # Found a definitive intent, no need to check other intents

        logger.info(f"Classified intent for query '{query}' as '{classified_intent}'.")
        return {"query": query, "intent": classified_intent}

if __name__ == '__main__':
    # Example usage for testing purposes
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.DEBUG) # Set to DEBUG for more detailed output in example

    # Initialize the node with default patterns
    classifier_node = IntentClassifierNode()

    # Test cases
    test_queries = [
        "Hello there!",
        "What is the weather like today?",
        "I need to book a flight to London.",
        "Remind me to call mom at 5 PM.",
        "Play some classical music.",
        "Tell me about quantum physics.",
        "I need help with my account.",
        "Goodbye for now!",
        "How are you?", # Should match GREETING
        "What's the forecast for tomorrow?", # Should match WEATHER_QUERY
        "find me a song by Queen", # Should match PLAY_MUSIC
        "what time is it?", # No direct match, should be UNKNOWN
        "", # Empty query
        123, # Invalid type
        None, # Invalid type
    ]

    print(f"\n--- Testing {classifier_node.node_name} ---")
    for q in test_queries:
        try:
            result = classifier_node.process(q, {})
            print(f"Query: '{result['query']}' -> Intent: '{result['intent']}'")
        except TypeError as e:
            print(f"Error processing query '{q}': {e}")
        except Exception as e:
            print(f"An unexpected error occurred for query '{q}': {e}")

    # Initialize with custom patterns
    print(f"\n--- Testing {classifier_node.node_name} with custom patterns ---")
    custom_patterns = {
        "ORDER_FOOD": ["order food", "pizza", "sushi", "delivery"],
        "LIGHT_CONTROL": ["turn on lights", "lights off", "dim the lights"],
    }
    custom_classifier = IntentClassifierNode(intent_patterns=custom_patterns)
    custom_queries = [
        "I want to order pizza for dinner.",
        "Can you turn on the living room lights?",
        "What is the capital of France?", # Should be UNKNOWN with custom patterns
    ]
    for q in custom_queries:
        try:
            result = custom_classifier.process(q, {})
            print(f"Query: '{result['query']}' -> Intent: '{result['intent']}'")
        except TypeError as e:
            print(f"Error processing query '{q}': {e}")
        except Exception as e:
            print(f"An unexpected error occurred for query '{q}': {e}")
