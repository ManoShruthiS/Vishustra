
import logging
from typing import Any, Dict

# Assuming vishustra_core.nodes.base_node exists as per project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class IntentClassifierNode(BaseNode):
    """
    A Vishustra processing node that simulates intent classification for a given text input.

    This node takes a user query (string) and, based on a set of simple keyword rules,
    classifies its likely intent. In a production environment, this would integrate
    with an actual ML model or external NLP service.
    """

    def __init__(self):
        """
        Initializes the IntentClassifierNode.
        In a real scenario, this might load a pre-trained intent model
        or configure API clients.
        """
        logger.debug("IntentClassifierNode initialized.")

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this node.
        """
        return "IntentClassifierNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to classify its intent.

        Expects `data` to be a string representing a user query.
        Uses a simple keyword-based approach for demonstration.

        Args:
            data (Any): The input data to be processed, expected to be a string.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       relevant to the processing.

        Returns:
            Dict[str, Any]: A dictionary containing the detected intent, confidence,
                            and the original query.

        Raises:
            TypeError: If the input `data` is not a string.
            ValueError: If `data` is an empty string after stripping whitespace.
        """
        logger.info(f"[{self.node_name}] Starting intent classification.")
        logger.debug(f"[{self.node_name}] Received context: {context}")

        if not isinstance(data, str):
            error_msg = (
                f"[{self.node_name}] Invalid input data type. "
                f"Expected string, but received {type(data).__name__}."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

        user_query = data.strip()
        if not user_query:
            error_msg = f"[{self.node_name}] Received an empty query after stripping whitespace."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Normalize for case-insensitive matching
        lower_query = user_query.lower()

        detected_intent = "general_inquiry"
        confidence = 0.5  # Default low confidence

        # Simple keyword-based intent detection
        if any(keyword in lower_query for keyword in ["order", "delivery", "shipment", "track"]):
            detected_intent = "order_management"
            confidence = 0.95
        elif any(keyword in lower_query for keyword in ["support", "help", "issue", "problem", "contact us"]):
            detected_intent = "customer_support"
            confidence = 0.90
        elif any(keyword in lower_query for keyword in ["account", "profile", "settings", "login", "password"]):
            detected_intent = "account_management"
            confidence = 0.85
        elif any(keyword in lower_query for keyword in ["price", "cost", "features", "product", "buy"]):
            detected_intent = "product_inquiry"
            confidence = 0.80
        elif any(keyword in lower_query for keyword in ["weather", "time", "news"]):
            detected_intent = "informational"
            confidence = 0.75

        result = {
            "original_query": user_query,
            "detected_intent": detected_intent,
            "confidence": confidence,
            "node_processed_by": self.node_name,
        }

        logger.info(
            f"[{self.node_name}] Classified intent: '{detected_intent}' "
            f"(Confidence: {confidence:.2f}) for query: '{user_query[:70]}{'...' if len(user_query) > 70 else ''}'"
        )
        logger.debug(f"[{self.node_name}] Returning result: {result}")
        return result

