import logging
from typing import Any, Dict, List

# Assuming 'vishustra_core' is available in the Python path
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class IntentClassifierNode(BaseNode):
    """
    A Vishustra processing node that classifies the intent of a user query string.

    This node simulates intent classification based on a set of keywords.
    In a production environment, this would typically integrate with an LLM
    or a dedicated machine learning model to provide more sophisticated and
    accurate intent recognition.
    """

    def __init__(self, default_intent: str = "general_query"):
        """
        Initializes the IntentClassifierNode.

        Args:
            default_intent (str): The intent to return if no specific intent is detected
                                  based on the defined rules. Defaults to "general_query".
        """
        self._default_intent = default_intent
        logger.debug(f"IntentClassifierNode initialized with default intent: '{self._default_intent}'")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "IntentClassifierNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data (expected to be a user query string) to classify its intent.

        The classification logic is based on keyword matching. The `context` dictionary
        can optionally provide a custom set of `classification_rules` to dynamically
        configure the node's behavior.

        Args:
            data (Any): The input data. Expected to be a string representing the user's query.
            context (Dict[str, Any]): A dictionary containing contextual information.
                                     Can include 'classification_rules' (Dict[str, List[str]])
                                     to override default rules for keyword-based intent detection.

        Returns:
            Dict[str, Any]: A dictionary containing:
                            - "query" (str): The original input query.
                            - "intent" (str): The classified intent (e.g., "account_management", "product_inquiry").
                            - "confidence" (float): A confidence score for the classified intent (0.0 to 1.0).

        Raises:
            ValueError: If the input `data` is not a string.
            RuntimeError: If an unexpected error occurs during the classification process.
        """
        if not isinstance(data, str):
            logger.error(f"Invalid input data type for IntentClassifierNode. Expected 'str', got '{type(data).__name__}'.")
            raise ValueError("Input data must be a string for intent classification.")

        query = data.strip().lower()
        classified_intent = self._default_intent
        confidence = 0.0 # Initialize with lowest confidence, updated upon detection

        if not query:
            logger.warning("Received empty or whitespace-only query for intent classification.")
            # For an empty query, return default intent with minimal confidence to indicate ambiguity
            return {"query": data, "intent": self._default_intent, "confidence": 0.05}

        # Define default classification rules. These can be dynamically overridden via `context`.
        default_classification_rules: Dict[str, List[str]] = {
            "account_management": ["reset password", "change email", "login issue", "account settings", "update profile"],
            "product_inquiry": ["what is", "features of", "how does it work", "tell me about", "product details", "specifications"],
            "technical_support": ["error", "bug", "troubleshoot", "fix problem", "technical issue", "not working"],
            "billing_inquiry": ["invoice", "payment", "subscription", "price", "bill", "charge", "refund"],
            "general_greeting": ["hello", "hi", "hey", "good morning", "good evening"],
            "thank_you": ["thank you", "thanks", "appreciate it", "cheers"]
        }

        # Allow context to provide custom classification rules, falling back to defaults
        classification_rules = context.get('classification_rules', default_classification_rules)

        try:
            detected_intents: List[str] = []
            for intent_name, keywords in classification_rules.items():
                if any(keyword in query for keyword in keywords):
                    detected_intents.append(intent_name)

            if detected_intents:
                # Simple strategy: take the first detected intent. In a more complex system,
                # this could involve intent ranking, multi-intent detection, or LLM-based disambiguation.
                classified_intent = detected_intents[0]
                confidence = 0.95 # High confidence for a direct rule match
                logger.debug(f"Matched intent '{classified_intent}' for query: '{data[:50]}...'")
            else:
                logger.debug(f"No specific intent detected for query: '{data[:50]}...'. Assigning default intent.")
                confidence = 0.5 # Medium confidence for default assignment when no rule matches

            result = {
                "query": data,
                "intent": classified_intent,
                "confidence": confidence
            }
            logger.info(f"Classified query '{data[:50]}...' as '{classified_intent}' (Confidence: {confidence:.2f})")
            return result
        except Exception as e:
            logger.exception(f"An unexpected error occurred during intent classification for query: '{data}'")
            raise RuntimeError(f"Failed to classify intent due to an internal error: {e}") from e