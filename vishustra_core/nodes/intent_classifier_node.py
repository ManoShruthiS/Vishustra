import logging
from typing import Any, Dict, List, Optional

# Assuming this import path exists in the Vishustra framework
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class IntentClassifierNode(BaseNode):
    """
    A Vishustra processing node designed to classify the intent of a given text input.

    This node simulates intent classification based on predefined keywords
    or patterns within the input data. It's built to identify a primary
    user intent from a natural language utterance, which is crucial for
    orchestrating subsequent LLM operations or routing requests.
    """

    def __init__(self, intent_rules: Optional[Dict[str, List[str]]] = None):
        """
        Initializes the IntentClassifierNode with optional, customizable intent rules.

        Args:
            intent_rules: A dictionary where keys are intent names (str) and
                          values are lists of keywords (str) associated with that intent.
                          These keywords are used to detect the intent within the input text.
                          If `None`, a robust set of default rules will be applied.
        """
        self._intent_rules = intent_rules if intent_rules is not None else self._get_default_intent_rules()
        logger.debug(f"[{self.node_name}] Initialized with intent rules: {self._intent_rules}")

    def _get_default_intent_rules(self) -> Dict[str, List[str]]:
        """
        Provides a default set of intent classification rules.

        This method centralizes the default intent definitions, making them
        easy to review and extend.
        """
        return {
            "greeting": ["hello", "hi", "hey", "good morning", "good evening"],
            "place_order": ["order", "buy", "purchase", "add to cart", "get me"],
            "check_status": ["status", "track", "where is", "update"],
            "cancel_order": ["cancel", "revoke", "undo", "stop order"],
            "customer_support": ["help", "support", "assist", "contact", "agent"],
            "goodbye": ["bye", "goodbye", "see you", "farewell"],
            "inquire_product": ["product info", "details about", "tell me about"],
            "schedule_appointment": ["schedule", "book appointment", "set up meeting"]
        }

    @property
    def node_name(self) -> str:
        """
        Returns the unique and descriptive name of this processing node.
        """
        return "IntentClassifierNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to classify its underlying intent.

        The `data` input is expected to be a string representing a user query
        or natural language utterance. The method attempts to match keywords
        within the data against the node's configured intent rules.

        Args:
            data: The input text (str) that needs its intent classified.
                  This is typically a user's prompt to an LLM.
            context: A dictionary containing contextual information relevant to
                     the current processing flow (e.g., `user_id`, `session_id`,
                     `transaction_id`). While not directly used for classification
                     in this simplified version, it's a vital part of the Vishustra
                     framework for context propagation.

        Returns:
            A dictionary containing the classification results:
            - 'intent': The identified intent as a string (e.g., "place_order").
                        Returns "unknown" if no intent is matched.
            - 'confidence': A numerical confidence score (float). Currently 1.0 for
                            a direct keyword match, 0.0 otherwise, but extensible
                            for more sophisticated models.
            - 'original_query': The verbatim input data received for classification.
            - 'context_passthrough': The original `context` dictionary, facilitating
                                     contextual flow to subsequent nodes.

        Raises:
            TypeError: If the input `data` is not a string, violating the expected
                       input contract for this node.
            ValueError: If `data` is an empty string after stripping whitespace,
                        indicating no meaningful input to classify.
        """
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected 'str', "
                f"but received '{type(data).__name__}'. Unable to classify intent."
            )
            raise TypeError(
                f"Input data for {self.node_name} must be a string. "
                f"Received type: {type(data).__name__}."
            )

        # Normalize the query for case-insensitive and whitespace-robust matching
        query = data.strip().lower()

        if not query:
            logger.warning(
                f"[{self.node_name}] Received an empty query after stripping. "
                f"No intent can be classified from an empty input."
            )
            raise ValueError(f"Input query for {self.node_name} cannot be empty.")

        detected_intent: str = "unknown"
        confidence: float = 0.0

        # Iterate through defined intents and their keywords to find a match
        for intent_name, keywords in self._intent_rules.items():
            for keyword in keywords:
                if keyword in query:
                    detected_intent = intent_name
                    confidence = 1.0  # Assign full confidence for a direct keyword match
                    logger.debug(
                        f"[{self.node_name}] Successfully identified intent '{detected_intent}' "
                        f"for query: '{data}' using keyword '{keyword}'."
                    )
                    break  # Found a keyword for this intent, move to the next intent
            if detected_intent != "unknown":
                break  # An intent has been found, no need to check further

        if detected_intent == "unknown":
            logger.info(
                f"[{self.node_name}] No specific intent identified for query: '{data}'. "
                f"Defaulting to 'unknown'."
            )

        # Construct the result payload for downstream nodes
        result = {
            "intent": detected_intent,
            "confidence": confidence,
            "original_query": data,
            "context_passthrough": context  # Ensure context is preserved and passed along
        }
        return result
