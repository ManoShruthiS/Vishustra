
import logging
from typing import Any, Dict, List

# Assuming vishustra_core.nodes.base_node exists as per project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class IntentClassifierNode(BaseNode):
    """
    A Vishustra processing node responsible for classifying the intent of
    incoming text data. This node simulates intent detection based on
    predefined keywords.
    """

    def __init__(self, intent_map: Dict[str, List[str]] = None, default_intent: str = "general_query"):
        """
        Initializes the IntentClassifierNode with an optional intent mapping and
        a default intent.

        Args:
            intent_map: An optional dictionary where keys are intent names (str)
                        and values are lists of keywords (List[str]) associated
                        with that intent. If None, a default mapping is used.
            default_intent: The intent string to assign if no specific intent
                            is detected based on the provided keywords.
        """
        self._intent_map = intent_map if intent_map is not None else self._build_default_intent_map()
        self._default_intent = default_intent
        logger.info(
            f"IntentClassifierNode initialized. Using default intent: '{self._default_intent}'. "
            f"Configured intents: {list(self._intent_map.keys())}"
        )

    def _build_default_intent_map(self) -> Dict[str, List[str]]:
        """
        Builds a default mapping of intents to keywords for demonstration purposes.
        """
        return {
            "book_flight": ["book flight", "plane ticket", "fly to", "reservation", "travel"],
            "check_status": ["status", "tracking", "order", "delivery", "check my"],
            "customer_support": ["help", "support", "contact us", "problem", "issue"],
            "greeting": ["hello", "hi", "hey", "good morning", "good evening"],
            "farewell": ["bye", "goodbye", "see you", "later"],
            "price_inquiry": ["how much", "cost", "price", "charge"],
            "product_info": ["tell me about", "what is", "information on"],
        }

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "IntentClassifierNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classifies the intent of the input text data based on the initialized
        keyword mapping.

        The `context` parameter is available for future extensions but is not
        used in this basic keyword-based classification.

        Args:
            data: The input text (e.g., user query) as a string.
            context: A dictionary of contextual information, potentially
                     including user history, session data, etc.

        Returns:
            A dictionary containing the detected intent and a simulated
            confidence score, e.g., {"intent": "book_flight", "confidence": 0.95}.

        Raises:
            TypeError: If the input `data` is not a string.
            ValueError: If the input `data` is an empty string after stripping whitespace.
        """
        logger.debug(f"[{self.node_name}] Starting process for data: '{data}'")

        if not isinstance(data, str):
            error_msg = (
                f"[{self.node_name}] Invalid input data type. Expected 'str', "
                f"but received '{type(data).__name__}'."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

        text = data.strip()
        if not text:
            error_msg = f"[{self.node_name}] Received an empty string for classification."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Normalize text for keyword matching
        normalized_text = text.lower()

        detected_intent = self._default_intent
        confidence = 0.5  # Default confidence for the default intent

        for intent, keywords in self._intent_map.items():
            for keyword in keywords:
                if keyword in normalized_text:
                    detected_intent = intent
                    confidence = 0.95  # Simulated higher confidence for matched intent
                    logger.debug(
                        f"[{self.node_name}] Matched keyword '{keyword}' for intent '{intent}' "
                        f"in text: '{data[:50]}...'"
                    )
                    break  # Found a match for this intent, no need to check further keywords
            if detected_intent != self._default_intent:
                break  # An intent was found, no need to check other intents

        result = {"intent": detected_intent, "confidence": confidence}
        logger.info(f"[{self.node_name}] Classified input as: {result} for text: '{data}'")
        return result

