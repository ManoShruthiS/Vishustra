
import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class IntentClassifierNode(BaseNode):
    """
    A processing node that classifies the intent of a given text input.

    This simulated node uses a simple keyword-based matching approach to
    determine the intent. In a real-world scenario, this would involve
    loading and using a trained NLP model (e.g., a BERT-based classifier).
    """

    def __init__(self, classification_rules: Dict[str, str] = None, default_intent: str = "GeneralInquiry"):
        """
        Initializes the IntentClassifierNode with classification rules.

        Args:
            classification_rules (Dict[str, str], optional): A dictionary
                where keys are keywords/phrases and values are the corresponding
                intent names. If None, a set of default rules is used.
            default_intent (str, optional): The intent to assign if no
                specific rule matches. Defaults to "GeneralInquiry".
        """
        self._classification_rules = classification_rules if classification_rules is not None else {
            "order status": "OrderStatus",
            "delivery time": "DeliveryInquiry",
            "refund request": "RefundRequest",
            "account settings": "AccountManagement",
            "billing issue": "BillingIssue",
            "technical support": "TechnicalSupport",
            "product information": "ProductInformation",
            "customer service": "CustomerService",
        }
        self._default_intent = default_intent
        logger.info(
            f"[{self.node_name}] Initialized with {len(self._classification_rules)} "
            f"classification rules and default intent '{self._default_intent}'."
        )

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "IntentClassifier"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input text to classify its intent.

        Expects `data` to be a string containing the text to be classified.
        The `context` dictionary can be used for runtime configuration, though
        this simulated node primarily uses its internal rules.

        Args:
            data (Any): The input data, expected to be a string.
            context (Dict[str, Any]): A dictionary containing contextual information.

        Returns:
            Dict[str, Any]: A dictionary containing the original text and the
            classified intent.
            Example: {"original_text": "What is my order status?", "classified_intent": "OrderStatus"}

        Raises:
            TypeError: If the input `data` is not a string.
            ValueError: If the input `data` is an empty string.
        """
        logger.debug(f"[{self.node_name}] Starting intent classification for data: '{data}'")

        if not isinstance(data, str):
            logger.error(f"[{self.node_name}] Invalid input data type. Expected str, got {type(data)}.")
            raise TypeError(f"[{self.node_name}] Input data must be a string for intent classification.")

        if not data.strip():
            logger.warning(f"[{self.node_name}] Received empty string for classification.")
            raise ValueError(f"[{self.node_name}] Input data cannot be an empty string.")

        text_lower = data.lower()
        classified_intent = self._default_intent
        matched_keyword = None

        for keyword, intent in self._classification_rules.items():
            if keyword in text_lower:
                classified_intent = intent
                matched_keyword = keyword
                logger.debug(
                    f"[{self.node_name}] Classified intent as '{intent}' "
                    f"based on keyword '{keyword}' in text."
                )
                break  # Found a match, no need to check further

        if matched_keyword is None:
            logger.info(f"[{self.node_name}] No specific intent matched. Assigning default intent '{classified_intent}'.")

        result = {
            "original_text": data,
            "classified_intent": classified_intent,
            "matched_keyword": matched_keyword,
        }
        logger.debug(f"[{self.node_name}] Finished intent classification. Result: {result}")
        return result

