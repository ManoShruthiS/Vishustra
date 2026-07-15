import logging
from typing import Any, Dict, Union

from vishustra_core.nodes.base_node import BaseNode # Assuming this path is correctly set up in the project

logger = logging.getLogger(__name__)

class IntentClassifierNode(BaseNode):
    """
    A Vishustra node that classifies the intent of a given text input.

    This node simulates intent classification based on predefined keywords.
    In a production environment, this would typically integrate with an
    external NLP service or a local ML model.

    Input (data):
        - A string representing the user utterance.
        - A dictionary containing a 'text' key with the user utterance as its value.

    Output:
        - A dictionary containing:
            - 'text': The original input utterance.
            - 'intent': The classified intent (e.g., 'order_management', 'delivery_inquiry', 'unknown_intent').
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "IntentClassifier"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classifies the intent of the input text.

        Args:
            data (Any): The input data, expected to be a string utterance
                        or a dictionary with a 'text' key.
            context (Dict[str, Any]): A dictionary for shared context or state
                                      across nodes in the orchestration.

        Returns:
            Dict[str, Any]: A dictionary containing the original text and the classified intent.

        Raises:
            ValueError: If the input data is not in the expected format.
        """
        utterance: str = ""
        
        if isinstance(data, str):
            utterance = data
        elif isinstance(data, dict):
            if 'text' in data and isinstance(data['text'], str):
                utterance = data['text']
            else:
                logger.error(
                    "IntentClassifierNode received a dictionary without a 'text' key "
                    "or with a non-string value for 'text'. Data: %s", data
                )
                raise ValueError(
                    "Invalid input data: Dictionary must contain a 'text' key with a string value."
                )
        else:
            logger.error(
                "IntentClassifierNode received invalid data type. Expected string or dict, got %s. Data: %s",
                type(data).__name__, data
            )
            raise ValueError(
                f"Invalid input data type for IntentClassifierNode. Expected string or dictionary, got {type(data).__name__}."
            )

        if not utterance.strip():
            logger.warning("IntentClassifierNode received an empty or whitespace-only utterance.")
            return {"text": utterance, "intent": "empty_utterance"}

        classified_intent = self._classify_intent_from_text(utterance)
        
        logger.debug(
            "Classified intent for utterance '%s': %s",
            utterance[:50] + "..." if len(utterance) > 50 else utterance, classified_intent
        )

        return {"text": utterance, "intent": classified_intent}

    def _classify_intent_from_text(self, text: str) -> str:
        """
        Internal method to simulate intent classification based on keywords.
        In a real-world scenario, this would involve NLP model inference.
        """
        text_lower = text.lower()

        if any(keyword in text_lower for keyword in ["order", "buy", "purchase", "item", "product"]):
            return "order_management"
        if any(keyword in text_lower for keyword in ["delivery", "shipment", "track", "package"]):
            return "delivery_inquiry"
        if any(keyword in text_lower for keyword in ["help", "support", "issue", "problem", "assist"]):
            return "customer_support"
        if any(keyword in text_lower for keyword in ["hello", "hi", "hey", "good morning", "good evening"]):
            return "greeting"
        if any(keyword in text_lower for keyword in ["cancel", "change", "amend"]):
            return "cancellation_modification"
        if any(keyword in text_lower for keyword in ["price", "cost", "how much"]):
            return "price_inquiry"

        return "unknown_intent"

# Example of how to use it (for internal testing, not part of Vishustra execution flow)
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    classifier_node = IntentClassifierNode()
    context_data: Dict[str, Any] = {}

    test_cases = [
        "I want to buy a new laptop.",
        {"text": "Track my last order."},
        "I need help with my account.",
        "Hello there!",
        "What is the price of this item?",
        "Can I cancel my subscription?",
        "Please provide support.",
        "Some random text that doesn't match.",
        "",
        "   ",
        {"not_text_key": "some value"},
        123,
    ]

    for i, test_input in enumerate(test_cases):
        try:
            logger.info(f"\n--- Test Case {i+1} ---")
            result = classifier_node.process(test_input, context_data)
            logger.info(f"Input: {test_input}")
            logger.info(f"Output: {result}")
        except ValueError as e:
            logger.error(f"Input: {test_input} -> Error: {e}")
        except Exception as e:
            logger.critical(f"Unhandled error for input: {test_input} -> {e}", exc_info=True)
