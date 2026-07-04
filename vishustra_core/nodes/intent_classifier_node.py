import logging
from typing import Any, Dict, Optional, List

# Assuming vishustra_core is available in the project's Python path
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class IntentClassifierNode(BaseNode):
    """
    A Vishustra processing node that classifies the intent of a given text input.

    This node simulates intent classification based on predefined keywords.
    It expects a string as input data and returns a dictionary containing
    the classified intent, a confidence score, and the original text.
    """

    def __init__(self, model_config: Optional[Dict[str, List[str]]] = None):
        """
        Initializes the IntentClassifierNode with a specific intent model configuration.

        Args:
            model_config (Optional[Dict[str, List[str]]]): A dictionary where keys
                are intent names (str) and values are lists of keywords (List[str])
                associated with that intent. If None, a default configuration is used.
        """
        if model_config is None:
            self._intent_model = {
                "greeting": ["hello", "hi", "hey", "good morning", "good evening"],
                "order_status": ["where is my order", "order status", "track my package", "delivery date"],
                "product_query": ["tell me about", "what is", "product details", "specifications", "features of"],
                "customer_support": ["help me", "support", "technical issue", "contact support"],
                "farewell": ["goodbye", "bye", "see you later", "thanks for your help"]
            }
            logger.info("IntentClassifierNode initialized with default model configuration.")
        else:
            if not isinstance(model_config, dict):
                raise TypeError("model_config must be a dictionary if provided.")
            # Basic validation for model_config structure
            for intent, keywords in model_config.items():
                if not isinstance(intent, str) or not isinstance(keywords, list) or \
                   not all(isinstance(k, str) for k in keywords):
                    raise ValueError(
                        f"Invalid model_config format. Expected Dict[str, List[str]], "
                        f"but found non-string intent '{intent}' or non-list/non-string keywords '{keywords}'."
                    )
            self._intent_model = {k.lower(): [kw.lower() for kw in v] for k, v in model_config.items()}
            logger.info("IntentClassifierNode initialized with custom model configuration.")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "IntentClassifier"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to classify its intent.

        Args:
            data (Any): The input data to classify. Expected to be a string representing
                        the user's utterance.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the current processing flow.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - 'intent' (str): The classified intent, e.g., "greeting", "unknown".
                - 'confidence' (float): A confidence score for the classification (0.0 to 1.0).
                - 'original_text' (str): The original input text.
                - 'error' (Optional[str]): An error message if processing failed.
        """
        if not isinstance(data, str):
            error_msg = f"IntentClassifierNode received invalid input type. Expected 'str', got '{type(data).__name__}'."
            logger.error(error_msg)
            return {
                "intent": "error",
                "confidence": 0.0,
                "original_text": str(data),
                "error": error_msg
            }

        text_input = data.strip()
        if not text_input:
            warning_msg = "IntentClassifierNode received empty string input. Classifying as 'unknown'."
            logger.warning(warning_msg)
            return {
                "intent": "unknown",
                "confidence": 0.5,
                "original_text": data,
                "error": None
            }

        processed_text = text_input.lower()
        classified_intent = "unknown"
        confidence = 0.5  # Default confidence for unknown intent

        for intent, keywords in self._intent_model.items():
            for keyword in keywords:
                if keyword in processed_text:
                    classified_intent = intent
                    confidence = 0.95  # High confidence for a direct keyword match
                    logger.debug(f"Intent '{intent}' detected for text: '{text_input}' via keyword '{keyword}'.")
                    break  # Found an intent, no need to check further keywords for this text
            if classified_intent != "unknown":
                break  # Found an intent, no need to check further intents

        logger.info(f"Classified intent for '{text_input}': '{classified_intent}' with confidence {confidence:.2f}.")

        return {
            "intent": classified_intent,
            "confidence": confidence,
            "original_text": data,
            "error": None
        }

# Example Usage (for local testing, not part of the committed code)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Default model
    classifier_default = IntentClassifierNode()
    print(f"Node Name: {classifier_default.node_name}")
    print(f"Result: {classifier_default.process('Hello, how are you?', {})}")
    print(f"Result: {classifier_default.process('Where is my order number 12345?', {})}")
    print(f"Result: {classifier_default.process('Tell me about the new product features.', {})}")
    print(f"Result: {classifier_default.process('I need some help with my account.', {})}")
    print(f"Result: {classifier_default.process('This is a completely random sentence.', {})}")
    print(f"Result: {classifier_default.process('', {})}")
    print(f"Result: {classifier_default.process(123, {})}") # Invalid input

    # Custom model
    custom_model = {
        "book_flight": ["book a flight", "flight reservation", "find flights"],
        "check_weather": ["what's the weather", "weather forecast", "temperature in"],
        "play_music": ["play some music", "start music", "next song"]
    }
    classifier_custom = IntentClassifierNode(model_config=custom_model)
    print("\n--- Custom Model Classifier ---")
    print(f"Node Name: {classifier_custom.node_name}")
    print(f"Result: {classifier_custom.process('I want to book a flight to London.', {})}")
    print(f"Result: {classifier_custom.process('What is the weather like in Paris?', {})}")
    print(f"Result: {classifier_custom.process('Play some jazz music.', {})}")
    print(f"Result: {classifier_custom.process('How much does it cost?', {})}") # Should be unknown for custom model
