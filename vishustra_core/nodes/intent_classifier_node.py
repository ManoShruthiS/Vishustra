import logging
from typing import Any, Dict

# Assuming vishustra_core.nodes.base_node exists in the project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class IntentClassifierNode(BaseNode):
    """
    A Vishustra processing node that classifies the intent of an input text.

    This node simulates intent classification based on predefined keyword rules.
    In a real-world scenario, this would typically integrate with a machine
    learning model (e.g., a fine-tuned transformer model, an NLU service, etc.).
    """

    _DEFAULT_INTENT_RULES = {
        "track_order": ["track my order", "where is my package", "delivery status"],
        "customer_support": ["i need help", "customer service", "technical issue", "support ticket"],
        "greeting": ["hello", "hi there", "good morning"],
        "product_inquiry": ["tell me about", "product details", "specifications", "what is the price"],
        "account_management": ["change password", "update profile", "my account"],
    }
    _DEFAULT_FALLBACK_INTENT = "general_query"
    _SIMULATED_CONFIDENCE_MATCH = 0.95
    _SIMULATED_CONFIDENCE_FALLBACK = 0.50

    def __init__(self):
        """
        Initializes the IntentClassifierNode.

        In a production system, this constructor would be responsible for
        loading an intent classification model, configuration files,
        or setting up client connections to an external NLU service.
        """
        logger.debug(f"[{self.node_name}] Initializing IntentClassifierNode instance.")

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this node.
        """
        return "IntentClassifier"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to classify its intent.

        The `data` input is expected to be a string containing the text
        for intent classification. The `context` dictionary can optionally
        provide dynamic intent rules or model configuration, though for
        this simulation, default rules are primarily used.

        Args:
            data: The input text string that needs its intent classified.
            context: A dictionary containing contextual information. This can
                     be used to pass dynamic parameters like 'intent_rules'
                     to override default classification rules.

        Returns:
            A dictionary containing:
            - "original_text": The input text string.
            - "classified_intent": A string representing the identified intent.
            - "confidence": A float representing the simulated confidence score
                            for the classification.

        Raises:
            ValueError: If the input 'data' is not a string, as this node
                        specifically operates on text input.
        """
        if not isinstance(data, str):
            error_msg = (
                f"[{self.node_name}] Invalid input data type for intent classification. "
                f"Expected 'str', but received '{type(data).__name__}'."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        text_lower = data.lower()
        classified_intent = self._DEFAULT_FALLBACK_INTENT
        confidence = self._SIMULATED_CONFIDENCE_FALLBACK

        logger.info(f"[{self.node_name}] Attempting to classify intent for text: '{data[:75]}{'...' if len(data) > 75 else ''}'")

        # Allow intent rules to be overridden by context for flexibility
        intent_rules_to_use = context.get("intent_rules", self._DEFAULT_INTENT_RULES)

        # Simulate intent classification using keyword matching
        for intent, keywords in intent_rules_to_use.items():
            if any(keyword.lower() in text_lower for keyword in keywords):
                classified_intent = intent
                confidence = self._SIMULATED_CONFIDENCE_MATCH
                logger.debug(f"[{self.node_name}] Matched intent '{intent}' based on keywords for text: '{data[:75]}'.")
                break # Take the first matching intent

        if classified_intent == self._DEFAULT_FALLBACK_INTENT:
            logger.info(f"[{self.node_name}] No specific intent matched. Falling back to '{self._DEFAULT_FALLBACK_INTENT}'.")

        result = {
            "original_text": data,
            "classified_intent": classified_intent,
            "confidence": confidence,
        }
        logger.debug(f"[{self.node_name}] Intent classification complete. Result: {result}")
        return result

# Example of how to configure logging for standalone testing:
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.DEBUG)
#     classifier = IntentClassifierNode()
#
#     print("\n--- Test Case 1: Track Order ---")
#     res1 = classifier.process("I want to track my order. Where is my package?", {})
#     print(res1)
#
#     print("\n--- Test Case 2: Customer Support ---")
#     res2 = classifier.process("I need help with a technical issue.", {})
#     print(res2)
#
#     print("\n--- Test Case 3: Greeting ---")
#     res3 = classifier.process("Hello there!", {})
#     print(res3)
#
#     print("\n--- Test Case 4: General Query (Fallback) ---")
#     res4 = classifier.process("What is the weather like today?", {})
#     print(res4)
#
#     print("\n--- Test Case 5: Product Inquiry ---")
#     res5 = classifier.process("Tell me about the new product specifications.", {})
#     print(res5)
#
#     print("\n--- Test Case 6: Override Rules via Context ---")
#     custom_rules = {
#         "farewell": ["goodbye", "see you later"],
#         "food_order": ["i want to order pizza", "get me a burger"],
#     }
#     res6 = classifier.process("Goodbye for now!", {"intent_rules": custom_rules})
#     print(res6)
#     res7 = classifier.process("I want to order pizza for dinner.", {"intent_rules": custom_rules})
#     print(res7)
#     res8 = classifier.process("Where is my package?", {"intent_rules": custom_rules}) # Should fallback now
#     print(res8)
#
#     print("\n--- Test Case 7: Invalid Input ---")
#     try:
#         classifier.process(123, {})
#     except ValueError as e:
#         print(f"Caught expected error: {e}")
#     try:
#         classifier.process(["list", "of", "strings"], {})
#     except ValueError as e:
#         print(f"Caught expected error: {e}")