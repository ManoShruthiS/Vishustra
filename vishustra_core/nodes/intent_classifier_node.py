import logging
from typing import Any, Dict, Optional

# Assuming vishustra_core is a package at the project root
# For local development/testing, you might need to adjust the import path
# or ensure the vishustra_core package is discoverable.
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class IntentClassifierNode(BaseNode):
    """
    A processing node that classifies the intent of a given text input.

    This node simulates intent classification based on predefined keywords.
    In a real-world scenario, this would integrate with an actual ML model
    or a rule-based system.
    """

    def __init__(self):
        """
        Initializes the IntentClassifierNode with its intent mapping.
        """
        self._intent_map = {
            "greeting": ["hello", "hi", "hey", "good morning", "good evening"],
            "farewell": ["bye", "goodbye", "see you", "later"],
            "order_status": ["order", "status", "track", "delivery"],
            "product_info": ["product", "information", "details", "specs", "features"],
            "support_request": ["help", "support", "issue", "problem", "ticket"],
            "thank_you": ["thank you", "thanks", "appreciate"]
        }
        logger.debug(f"IntentClassifierNode initialized with intent map: {self._intent_map}")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "IntentClassifier"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to classify its intent.

        Expects `data` to be a dictionary containing at least a 'text' key with
        the string to be classified.

        Args:
            data: The input data, expected as `Dict[str, Any]` with a 'text' key.
            context: A dictionary for shared contextual information across nodes.

        Returns:
            A dictionary containing the original data, the classified intent,
            and an optional confidence score.

        Raises:
            ValueError: If the input data is not a dictionary or lacks the 'text' key.
            TypeError: If the value associated with the 'text' key is not a string.
        """
        if not isinstance(data, dict):
            logger.error(f"Invalid input data type for {self.node_name}. Expected dict, got {type(data)}.")
            raise ValueError(f"Input data for {self.node_name} must be a dictionary.")

        if 'text' not in data:
            logger.error(f"Missing 'text' key in input data for {self.node_name}: {data.keys()}")
            raise ValueError(f"Input data for {self.node_name} must contain a 'text' key.")

        text_to_classify = data['text']
        if not isinstance(text_to_classify, str):
            logger.error(f"Invalid type for 'text' key in {self.node_name}. Expected str, got {type(text_to_classify)}.")
            raise TypeError(f"The 'text' value in input data for {self.node_name} must be a string.")

        normalized_text = text_to_classify.lower()
        classified_intent: Optional[str] = None
        confidence: float = 0.0

        for intent, keywords in self._intent_map.items():
            for keyword in keywords:
                if keyword in normalized_text:
                    classified_intent = intent
                    confidence = 1.0  # Simple simulation: 1.0 if matched, else 0.0 for unclear
                    logger.debug(f"Classified '{text_to_classify}' as '{intent}' based on keyword '{keyword}'.")
                    break
            if classified_intent:
                break

        if classified_intent is None:
            classified_intent = "unclear_intent"
            confidence = 0.5 # A default confidence for unknown intents
            logger.info(f"Could not clearly classify intent for text: '{text_to_classify}'. Defaulting to '{classified_intent}'.")

        result = {
            "original_input": data,
            "classified_intent": classified_intent,
            "confidence": confidence
        }
        logger.info(f"Processed text: '{text_to_classify}' -> Intent: '{classified_intent}' (Confidence: {confidence:.2f})")
        return result

# Example usage (for testing, not part of the required output file)
if __name__ == "__main__":
    # Configure basic logging for local testing
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Mock the BaseNode for local testing if vishustra_core isn't fully set up
    # In a real setup, vishustra_core would be installed.
    # This is just to make the example runnable without the full framework.
    try:
        from vishustra_core.nodes.base_node import BaseNode # Try normal import
    except ImportError:
        # Fallback for local testing if the package structure isn't there
        print("Vishustra_core not found, mocking BaseNode for local test.")
        class BaseNode(ABC):
            @abstractmethod
            def process(self, data: Any, context: Dict[str, Any]) -> Any:
                pass
            @property
            @abstractmethod
            def node_name(self) -> str:
                pass

    classifier = IntentClassifierNode()

    test_data = [
        {"text": "Hello there, how are you?"},
        {"text": "What is the status of my recent order?"},
        {"text": "I'm looking for product specifications."},
        {"text": "Goodbye for now!"},
        {"text": "I have a problem with my account."},
        {"text": "Thank you for your help!"},
        {"text": "Just some random text without clear intent."},
        {"query": "This has no 'text' key."} # Invalid input
    ]

    for item in test_data:
        try:
            processed_result = classifier.process(item, {})
            print(f"\nInput: {item}")
            print(f"Output: {processed_result}")
        except (ValueError, TypeError) as e:
            print(f"\nError processing input {item}: {e}")

    # Test with invalid text type
    try:
        classifier.process({"text": 123}, {})
    except (ValueError, TypeError) as e:
        print(f"\nError processing input {{'text': 123}}: {e}")

    # Test with non-dict input
    try:
        classifier.process("just a string", {})
    except (ValueError, TypeError) as e:
        print(f"\nError processing input 'just a string': {e}")