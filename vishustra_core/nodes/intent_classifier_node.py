from typing import Any, Dict, List, Optional
import logging

# Assume vishustra_core.nodes.base_node exists as per project context
# For standalone execution/testing, you might need a mock or actual BaseNode definition.
# Here we'll rely on the provided BaseNode signature.
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class IntentClassifierNode(BaseNode):
    """
    A Vishustra processing node that classifies the intent of a given text utterance.

    This node uses a predefined mapping of keywords to identify common user intents.
    It's designed to be a foundational component for conversational AI flows.
    """

    def __init__(self, intent_mapping: Dict[str, List[str]], default_intent: str = "unknown"):
        """
        Initializes the IntentClassifierNode with an intent mapping and a default intent.

        Args:
            intent_mapping (Dict[str, List[str]]): A dictionary where keys are intent names
                                                   (e.g., "greeting", "order_status") and
                                                   values are lists of keywords (strings)
                                                   associated with that intent. Keywords are
                                                   case-insensitive during matching.
            default_intent (str): The intent to return if no specific intent is matched.
                                  Defaults to "unknown".
        """
        if not isinstance(intent_mapping, dict) or not all(
            isinstance(k, str) and isinstance(v, list) and all(isinstance(i, str) for i in v)
            for k, v in intent_mapping.items()
        ):
            logger.error("IntentClassifierNode received invalid intent_mapping. Expected Dict[str, List[str]].")
            raise TypeError("intent_mapping must be a dictionary mapping intent names to lists of keywords.")
        
        if not isinstance(default_intent, str) or not default_intent:
            logger.error("IntentClassifierNode received invalid default_intent. Expected non-empty string.")
            raise ValueError("default_intent must be a non-empty string.")

        self._intent_mapping = {
            intent_name.lower(): [keyword.lower() for keyword in keywords]
            for intent_name, keywords in intent_mapping.items()
        }
        self._default_intent = default_intent
        logger.info(f"Initialized IntentClassifierNode with {len(self._intent_mapping)} defined intents and default '{self._default_intent}'.")

    @property
    def node_name(self) -> str:
        """
        Returns the name of the node.
        """
        return "IntentClassifier"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data (expected to be a string utterance) to classify its intent.

        Args:
            data (Any): The input data to be processed. Expected to be a string representing
                        a user utterance.
            context (Dict[str, Any]): A dictionary containing contextual information for processing.
                                      While available, this node primarily uses 'data'.

        Returns:
            Dict[str, Any]: A dictionary containing the original utterance and the classified intent.
                            Example: {"original_utterance": "Hello there!", "classified_intent": "greeting"}

        Raises:
            ValueError: If the input `data` is not a string.
        """
        if not isinstance(data, str):
            logger.error(f"[{self.node_name}] Received invalid input data type. Expected 'str', got '{type(data).__name__}'. Data: {data}")
            raise ValueError(f"{self.node_name} requires string data for intent classification. Received type: {type(data).__name__}.")

        utterance = data.strip().lower()
        classified_intent = self._default_intent
        matched_keyword: Optional[str] = None

        logger.debug(f"[{self.node_name}] Attempting to classify intent for utterance: '{data}'")

        # Iterate through defined intents and their keywords
        for intent_name, keywords in self._intent_mapping.items():
            for keyword in keywords:
                if keyword in utterance:
                    classified_intent = intent_name
                    matched_keyword = keyword
                    break # Found a keyword for this intent, move to next utterance
            if classified_intent != self._default_intent:
                break # Found an intent, no need to check further intents

        if classified_intent == self._default_intent:
            logger.warning(f"[{self.node_name}] No specific intent matched for utterance '{data}'. Defaulting to '{self._default_intent}'.")
        else:
            logger.debug(f"[{self.node_name}] Classified utterance '{data}' as '{classified_intent}' via keyword '{matched_keyword}'.")

        result = {
            "original_utterance": data,
            "classified_intent": classified_intent,
            "matching_keyword": matched_keyword # Useful for debugging or further processing
        }
        
        return result

# Example of how to use this node (for demonstration, not part of the required output)
if __name__ == '__main__':
    # Setup basic logging for demonstration
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Mock BaseNode for local testing if vishustra_core isn't installed
    # In a real Vishustra environment, vishustra_core.nodes.base_node.BaseNode would be available.
    try:
        from vishustra_core.nodes.base_node import BaseNode
    except ImportError:
        print("Warning: vishustra_core not found. Using a mock BaseNode for local testing.")
        from abc import ABC, abstractmethod
        class BaseNode(ABC):
            @abstractmethod
            def process(self, data: Any, context: Dict[str, Any]) -> Any: pass
            @property
            @abstractmethod
            def node_name(self) -> str: pass
        # Re-define IntentClassifierNode to inherit from the mock BaseNode if necessary
        # This block is illustrative and not part of the required output.
        class IntentClassifierNode(BaseNode):
            def __init__(self, intent_mapping: Dict[str, List[str]], default_intent: str = "unknown"):
                if not isinstance(intent_mapping, dict) or not all(
                    isinstance(k, str) and isinstance(v, list) and all(isinstance(i, str) for i in v)
                    for k, v in intent_mapping.items()
                ):
                    raise TypeError("intent_mapping must be a dictionary mapping intent names to lists of keywords.")
                if not isinstance(default_intent, str) or not default_intent:
                    raise ValueError("default_intent must be a non-empty string.")

                self._intent_mapping = {
                    intent_name.lower(): [keyword.lower() for keyword in keywords]
                    for intent_name, keywords in intent_mapping.items()
                }
                self._default_intent = default_intent
                logger.info(f"Initialized IntentClassifierNode with {len(self._intent_mapping)} defined intents and default '{self._default_intent}'.")

            @property
            def node_name(self) -> str:
                return "IntentClassifier"

            def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
                if not isinstance(data, str):
                    logger.error(f"[{self.node_name}] Received invalid input data type. Expected 'str', got '{type(data).__name__}'. Data: {data}")
                    raise ValueError(f"{self.node_name} requires string data for intent classification. Received type: {type(data).__name__}.")

                utterance = data.strip().lower()
                classified_intent = self._default_intent
                matched_keyword: Optional[str] = None

                logger.debug(f"[{self.node_name}] Attempting to classify intent for utterance: '{data}'")

                for intent_name, keywords in self._intent_mapping.items():
                    for keyword in keywords:
                        if keyword in utterance:
                            classified_intent = intent_name
                            matched_keyword = keyword
                            break
                    if classified_intent != self._default_intent:
                        break

                if classified_intent == self._default_intent:
                    logger.warning(f"[{self.node_name}] No specific intent matched for utterance '{data}'. Defaulting to '{self._default_intent}'.")
                else:
                    logger.debug(f"[{self.node_name}] Classified utterance '{data}' as '{classified_intent}' via keyword '{matched_keyword}'.")

                result = {
                    "original_utterance": data,
                    "classified_intent": classified_intent,
                    "matching_keyword": matched_keyword
                }
                
                return result


    intent_map = {
        "greeting": ["hello", "hi", "hey"],
        "order_status": ["where is my order", "track order", "status of order"],
        "cancel_order": ["cancel my order", "stop order"],
        "product_inquiry": ["tell me about", "what is", "info on"],
        "goodbye": ["bye", "goodbye", "see you"]
    }

    classifier = IntentClassifierNode(intent_mapping=intent_map, default_intent="general_query")

    test_cases = [
        "Hello there!",
        "Where is my order 12345?",
        "Can you cancel my order?",
        "Tell me about your latest product.",
        "I need help with something.",
        "Goodbye for now.",
        123 # Invalid input
    ]

    for test_data in test_cases:
        try:
            result = classifier.process(test_data, {})
            print(f"Input: '{test_data}' -> Result: {result}")
        except ValueError as e:
            print(f"Input: '{test_data}' -> Error: {e}")
        except TypeError as e:
            print(f"Input: '{test_data}' -> Error: {e}")

    # Test with an empty mapping to check default behavior
    empty_classifier = IntentClassifierNode(intent_mapping={}, default_intent="catch_all")
    result_empty = empty_classifier.process("Any phrase at all.", {})
    print(f"Input with empty map: 'Any phrase at all.' -> Result: {result_empty}")

    # Test with invalid constructor arguments
    try:
        invalid_classifier = IntentClassifierNode(intent_mapping="not a dict")
    except (TypeError, ValueError) as e:
        print(f"Constructor error test 1: {e}")

    try:
        invalid_classifier_2 = IntentClassifierNode(intent_mapping={"greet": ["hi"]}, default_intent=None)
    except (TypeError, ValueError) as e:
        print(f"Constructor error test 2: {e}")