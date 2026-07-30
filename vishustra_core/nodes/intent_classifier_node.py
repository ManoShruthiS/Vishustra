import logging
from typing import Any, Dict, Optional

# Assuming BaseNode is located at vishustra_core/nodes/base_node.py
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class IntentClassifierNode(BaseNode):
    """
    A Vishustra node that simulates intent classification from input text.

    This node takes a string (e.g., a user query) and attempts to classify its
    underlying intent based on a predefined keyword map. It returns a dictionary
    containing the classified intent and a simulated confidence score.
    """

    def __init__(self, intent_map: Optional[Dict[str, str]] = None, default_intent: str = "fallback_intent"):
        """
        Initializes the IntentClassifierNode with a specific intent mapping
        or a default one.

        Args:
            intent_map (Optional[Dict[str, str]]): A dictionary where keys are
                                                   keywords/phrases (case-insensitive)
                                                   and values are the corresponding intent names.
                                                   If None, a default map is used.
            default_intent (str): The intent to return if no specific keyword match
                                  is found in the input data.
        """
        self._intent_map = intent_map if intent_map is not None else self._get_default_intent_map()
        self._default_intent = default_intent
        logger.debug(f"{self.node_name} initialized with custom map (size: {len(self._intent_map)}) "
                     f"and default intent: '{self._default_intent}'.")

    def _get_default_intent_map(self) -> Dict[str, str]:
        """
        Provides a default set of keyword-to-intent mappings.
        """
        return {
            "order status": "check_order_status",
            "delivery date": "check_order_status",
            "track my order": "check_order_status",
            "product information": "get_product_info",
            "details about": "get_product_info",
            "customer support": "customer_support",
            "contact us": "customer_support",
            "account settings": "manage_account",
            "profile update": "manage_account",
            "cancel order": "cancel_order",
            "return item": "return_item",
            "refund process": "return_item",
            "invoice request": "get_invoice",
            "payment issue": "payment_support",
        }

    @property
    def node_name(self) -> str:
        """Returns the name of this processing node."""
        return "IntentClassifierNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to classify its intent based on internal logic.

        Args:
            data (Any): The input data to be classified. Expected to be a string
                        representing a user query or similar text.
            context (Dict[str, Any]): A dictionary containing contextual information
                                      that might influence or be influenced by the
                                      classification process. For this node, it's primarily
                                      passed through but could be used for advanced features.

        Returns:
            Dict[str, Any]: A dictionary containing the classified intent, a simulated
                            confidence score, and the source node name.
                            Example: {"intent": "check_order_status", "confidence": 0.95, "source": "IntentClassifierNode"}

        Raises:
            TypeError: If the input 'data' is not a string.
        """
        if not isinstance(data, str):
            logger.error(f"[{self.node_name}] Received non-string data of type '{type(data).__name__}'. "
                         "Expected 'str' for intent classification.")
            raise TypeError(f"Input 'data' must be a string for {self.node_name} intent classification.")

        query = data.strip().lower()
        classified_intent = self._default_intent
        confidence = 0.5  # Default confidence for no specific match or empty query

        if not query:
            logger.warning(f"[{self.node_name}] Received an empty string for classification. Returning default intent.")
            return {"intent": classified_intent, "confidence": confidence, "source": self.node_name}

        # Simple keyword-based matching for demonstration.
        # In a real-world scenario, this would involve ML models (e.g., BERT, Sentence Transformers).
        for keyword, intent in self._intent_map.items():
            if keyword in query:
                classified_intent = intent
                confidence = 0.95  # Higher confidence for a direct keyword match
                logger.debug(f"[{self.node_name}] Query '{data}' matched keyword '{keyword}', "
                             f"classified as '{intent}'.")
                break  # Prioritize the first matching keyword

        if classified_intent == self._default_intent:
            logger.info(f"[{self.node_name}] No specific intent matched for query '{data}'. "
                        f"Falling back to default intent: '{self._default_intent}'.")
        else:
            logger.info(f"[{self.node_name}] Classified intent for query '{data}' as '{classified_intent}' "
                        f"with confidence {confidence}.")

        return {"intent": classified_intent, "confidence": confidence, "source": self.node_name}
