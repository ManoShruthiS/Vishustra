import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class IntentClassifierNode(BaseNode):
    """
    A Vishustra processing node that classifies the intent of a given text input.

    This node simulates intent classification based on keyword matching.
    In a real-world scenario, this would typically integrate with a machine
    learning model or a more sophisticated natural language processing service.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "IntentClassifierNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to determine its intent based on keyword matching.

        Expected `data` format:
        A dictionary containing at least a "text" key with the user's query string.
        Example: `{"text": "What is my account balance?", "user_id": "123"}`

        The method extracts the "text" field, performs a basic keyword-based
        classification, and augments the input `data` with an "intent" key.

        Args:
            data: The input data, expected to be a dictionary containing a "text" key.
                  Other keys in `data` will be preserved in the output.
            context: A dictionary containing contextual information relevant to
                     the current orchestration run (e.g., configuration, session state).
                     This node does not currently use the `context` for classification logic
                     but relies on it for potential future extensibility.

        Returns:
            A dictionary that is a copy of the input `data` but augmented with an
            "intent" key, indicating the classified intent (e.g., "check_balance",
            "transfer_funds", "unknown").

        Raises:
            TypeError: If the input `data` is not a dictionary, or if the value
                       associated with the "text" key is not a string.
            ValueError: If the "text" key is missing from the input `data`.
        """
        logger.debug(f"[{self.node_name}] Starting intent classification process. Input data keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")

        if not isinstance(data, dict):
            error_msg = f"[{self.node_name}] Invalid input data type. Expected dict, but received {type(data)}. Cannot classify intent."
            logger.error(error_msg)
            raise TypeError(error_msg)

        text_to_classify = data.get("text")

        if text_to_classify is None:
            error_msg = f"[{self.node_name}] 'text' key missing in input data. A text field is required for intent classification. Data received: {data.keys()}"
            logger.warning(error_msg)
            raise ValueError(error_msg)

        if not isinstance(text_to_classify, str):
            error_msg = f"[{self.node_name}] Invalid type for 'text' field. Expected str, but received {type(text_to_classify)}. Cannot classify intent."
            logger.error(error_msg)
            raise TypeError(error_msg)

        # --- Simulate intent classification using basic keyword matching ---
        # In a production environment, this logic would typically involve:
        # 1. Loading a pre-trained machine learning model (e.g., transformer-based).
        # 2. Sending the text to an external NLP service or API.
        # 3. Using a more sophisticated rule-based engine.
        lower_text = text_to_classify.lower()
        classified_intent = "unknown"

        if any(keyword in lower_text for keyword in ["balance", "account status", "my money"]):
            classified_intent = "check_balance"
        elif any(keyword in lower_text for keyword in ["transfer", "send money", "move funds"]):
            classified_intent = "transfer_funds"
        elif any(keyword in lower_text for keyword in ["pay bill", "utility payment", "invoice"]):
            classified_intent = "pay_bill"
        elif any(keyword in lower_text for keyword in ["help", "support", "contact us", "assistance"]):
            classified_intent = "get_support"
        elif "order" in lower_text and any(keyword in lower_text for keyword in ["status", "track", "where is"]):
            classified_intent = "check_order_status"
        elif any(keyword in lower_text for keyword in ["cancel", "unsubscribe", "stop service"]):
            classified_intent = "cancel_service"
        elif any(keyword in lower_text for keyword in ["settings", "preferences", "update profile"]):
            classified_intent = "manage_settings"
        elif any(keyword in lower_text for keyword in ["hello", "hi", "hey"]):
            classified_intent = "greeting"

        # Create a new dictionary to avoid modifying the input `data` directly
        # for immutability, though deep copying might be overkill here.
        result_data = {**data, "intent": classified_intent}
        logger.info(f"[{self.node_name}] Classified intent for text '{text_to_classify[:70]}{'...' if len(text_to_classify) > 70 else ''}' as: '{classified_intent}'")
        return result_data