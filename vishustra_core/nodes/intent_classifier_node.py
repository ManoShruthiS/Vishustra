import logging
from typing import Any, Dict, List

# Assuming this path exists in the Vishustra project structure
# For local testing/development, you might need to adjust sys.path or use a mock.
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class IntentClassifierNode(BaseNode):
    """
    A Vishustra processing node that simulates classifying the intent of a given text utterance.
    This node processes a string input and determines a primary intent along with a confidence score.
    """

    def __init__(self, supported_intents: List[str] = None, default_intent: str = "unknown"):
        """
        Initializes the IntentClassifierNode.

        Args:
            supported_intents (List[str], optional): A list of intents this classifier is configured to recognize.
                                                     Defaults to a common set if not provided.
            default_intent (str, optional): The intent to return if no specific intent can be classified.
                                            Defaults to "unknown".
        """
        self._supported_intents = supported_intents if supported_intents is not None else [
            "greet", "order", "cancel", "query", "thank", "goodbye"
        ]
        if default_intent not in self._supported_intents:
            # Ensure default intent is always considered supported if not already explicitly listed
            self._supported_intents.append(default_intent)
        self._default_intent = default_intent
        
        logger.info(f"IntentClassifierNode initialized with supported intents: {self._supported_intents}, default: '{self._default_intent}'")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "IntentClassifier"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data, expected to be a string utterance, to classify its intent.
        This method simulates intent classification based on simple keyword matching.

        Args:
            data (Any): The input data, expected to be a string utterance to be classified.
            context (Dict[str, Any]): The processing context, which may contain additional
                                      information relevant to classification (e.g., user history),
                                      though not heavily used in this simulation.

        Returns:
            Dict[str, Any]: A dictionary containing the classified intent and a simulated confidence score.
                            Example: {"intent": "greet", "confidence": 0.95}

        Raises:
            TypeError: If the input data is not a string.
        """
        if not isinstance(data, str):
            logger.error(f"IntentClassifierNode received non-string input. Type: {type(data)}. Data: {data}")
            raise TypeError(f"IntentClassifierNode expects a string utterance for classification, but received {type(data)}.")

        utterance = data.lower().strip()
        classified_intent = self._default_intent
        confidence = 0.5  # Default low confidence for unclassified or ambiguous cases

        # Simulate intent classification based on keywords
        if any(keyword in utterance for keyword in ["hello", "hi", "hey", "good morning", "good evening"]):
            classified_intent = "greet"
            confidence = 0.95
        elif any(keyword in utterance for keyword in ["order", "purchase", "buy", "place an order", "i want to order"]):
            classified_intent = "order"
            confidence = 0.90
        elif any(keyword in utterance for keyword in ["cancel", "revoke", "stop order", "undo", "cancel my"]):
            classified_intent = "cancel"
            confidence = 0.88
        elif any(keyword in utterance for keyword in ["what is", "how to", "information about", "tell me about", "explain"]):
            classified_intent = "query"
            confidence = 0.85
        elif any(keyword in utterance for keyword in ["thank you", "thanks", "appreciate it", "cheers"]):
            classified_intent = "thank"
            confidence = 0.92
        elif any(keyword in utterance for keyword in ["goodbye", "bye", "see you", "farewell"]):
            classified_intent = "goodbye"
            confidence = 0.90
        
        # Validate if the classified intent is among the supported ones.
        # This acts as a safeguard, especially if keyword logic leads to an unexpected string.
        if classified_intent not in self._supported_intents:
            logger.warning(f"Simulated classification resulted in unexpected intent '{classified_intent}' for utterance '{utterance[:100]}...'. Falling back to default intent: '{self._default_intent}'.")
            classified_intent = self._default_intent
            confidence = 0.5 # Reset confidence if falling back

        logger.info(f"Classified intent for utterance '{utterance[:100]}...' as '{classified_intent}' with confidence {confidence:.2f}")

        return {"intent": classified_intent, "confidence": confidence}

# Example of how it might be used (for testing purposes, not part of the node file)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize the node
    classifier_node = IntentClassifierNode(
        supported_intents=["greet", "order", "cancel", "query", "thank", "goodbye", "custom_action"],
        default_intent="unknown"
    )

    test_utterances = [
        "Hello there!",
        "I'd like to place an order for a coffee.",
        "Can I cancel my last purchase?",
        "What is the capital of France?",
        "Thank you for your help!",
        "See you later!",
        "This is a random sentence.",
        123, # Invalid input type
        "", # Empty string
        "How do I sign up for the custom_action plan?"
    ]

    for i, utterance in enumerate(test_utterances):
        try:
            result = classifier_node.process(utterance, {})
            logger.info(f"Processed utterance {i+1}: '{utterance}' -> {result}")
        except TypeError as e:
            logger.error(f"Error processing utterance {i+1}: '{utterance}' -> {e}")
        except Exception as e:
            logger.critical(f"An unexpected error occurred for utterance {i+1}: '{utterance}' -> {e}")

    # Test with a node initialized with fewer custom intents
    custom_classifier = IntentClassifierNode(supported_intents=["start", "end"])
    try:
        result = custom_classifier.process("Start the engine", {})
        logger.info(f"Custom classifier result: {result}") # Should be 'start'
        result = custom_classifier.process("Saying hello", {})
        logger.info(f"Custom classifier result: {result}") # Should be 'unknown' due to default
    except Exception as e:
        logger.error(f"Error with custom classifier: {e}")