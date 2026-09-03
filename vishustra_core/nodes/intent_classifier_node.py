import logging
from typing import Any, Dict, Optional

# Assuming vishustra_core.nodes.base_node exists in the project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class IntentClassifierNode(BaseNode):
    """
    A Vishustra node designed to classify the intent of a given text query.

    This node uses a keyword-based approach to simulate intent classification.
    It can be configured with a custom mapping of keywords (or phrases) to
    specific intent names. If no specific intent is detected, it defaults
    to a predefined 'general_inquiry' intent.
    """

    def __init__(self, intent_map: Optional[Dict[str, str]] = None, default_intent: str = "general_inquiry"):
        """
        Initializes the IntentClassifierNode.

        Args:
            intent_map (Optional[Dict[str, str]]): A dictionary where keys are keywords
                                                    (or phrases) and values are the
                                                    corresponding intent names. Keys
                                                    will be matched case-insensitively.
                                                    Example: {"book appointment": "schedule_appointment",
                                                              "cancel order": "cancel_order"}.
                                                    If None, a default internal map is used.
            default_intent (str): The intent to assign if no specific intent is detected
                                  based on the provided `intent_map`.
        """
        self._intent_map = intent_map if intent_map is not None else self._get_default_intent_map()
        self._default_intent = default_intent
        # Normalize intent map keys to lowercase for robust matching
        self._normalized_intent_map = {k.lower(): v for k, v in self._intent_map.items()}
        logger.debug(
            f"{self.node_name} initialized with intent map: {self._normalized_intent_map} "
            f"and default intent: '{self._default_intent}'."
        )

    def _get_default_intent_map(self) -> Dict[str, str]:
        """
        Provides a default intent map if none is specified during initialization.
        In a production system, this data would typically be loaded from a
        configuration service or file.
        """
        return {
            "schedule appointment": "schedule_appointment",
            "book appointment": "schedule_appointment",
            "cancel order": "cancel_order",
            "check status": "check_order_status",
            "shipping info": "check_order_status",
            "contact support": "customer_support",
            "help me": "customer_support",
            "reset password": "account_management",
            "change password": "account_management",
            "pricing": "product_inquiry",
            "features": "product_inquiry",
            "talk to human": "customer_support",
            "complaint": "customer_support",
        }

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "IntentClassifierNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data, classifying its intent based on configured keywords.

        Args:
            data (Any): The input data, expected to be a string representing the user query.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       relevant to the current execution flow. While
                                       this node's core logic doesn't strictly depend
                                       on context, it can be used for logging, metrics,
                                       or to pass configurations in more advanced scenarios.

        Returns:
            Dict[str, Any]: A dictionary containing the original query and the classified intent.
                            Example: {"query": "I want to book an appointment", "intent": "schedule_appointment"}

        Raises:
            ValueError: If the input data is not a string, as this node expects text for classification.
            Exception: For any unexpected errors occurring during the intent classification process.
        """
        if not isinstance(data, str):
            logger.error(f"[{self.node_name}] Invalid input type. Expected a string, received {type(data)}.")
            raise ValueError(f"Input data for {self.node_name} must be a string, got {type(data).__name__}.")

        query = data.strip().lower()
        classified_intent = self._default_intent
        matched_keyword = None

        logger.debug(f"[{self.node_name}] Starting intent classification for query: '{query}'")

        try:
            # Iterate through the normalized map for case-insensitive keyword matching
            for keyword_phrase, intent_name in self._normalized_intent_map.items():
                if keyword_phrase in query:
                    classified_intent = intent_name
                    matched_keyword = keyword_phrase
                    logger.debug(
                        f"[{self.node_name}] Matched keyword '{matched_keyword}' "
                        f"leading to intent '{classified_intent}' for query: '{query}'."
                    )
                    break  # First match found, consider it the primary intent
            else:
                logger.debug(f"[{self.node_name}] No specific keyword matched. Assigning default intent: '{self._default_intent}' for query: '{query}'.")

            # Update context with classification result for traceability or downstream nodes
            context.setdefault('classified_intents', []).append({
                'node': self.node_name,
                'query': data,
                'intent': classified_intent,
                'matched_keyword': matched_keyword
            })

            result = {"query": data, "intent": classified_intent}
            logger.info(f"[{self.node_name}] Successfully classified query '{data}' as intent: '{classified_intent}'.")
            return result

        except Exception as e:
            logger.exception(f"[{self.node_name}] An unexpected error occurred during "
                             f"intent classification for query '{data}'. Error: {e}")
            # Depending on the framework's error handling policy,
            # we might wrap this in a custom framework exception.
            raise