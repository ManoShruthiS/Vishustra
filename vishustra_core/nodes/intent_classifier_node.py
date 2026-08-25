import logging
from typing import Any, Dict, List, Optional

# Assuming BaseNode is available at this path within the project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class IntentClassifierNode(BaseNode):
    """
    A Vishustra processing node designed to classify the intent of an input text.

    This node simulates intent classification based on a configurable set of keywords
    associated with different intents. It identifies the most probable intent from
    the input query and returns a structured result.
    """

    def __init__(self,
                 intent_mapping: Optional[Dict[str, List[str]]] = None,
                 default_intent: str = "unclassified",
                 confidence_score: float = 0.9):
        """
        Initializes the IntentClassifierNode with classification rules.

        Args:
            intent_mapping: An optional dictionary where keys are intent names (str)
                            and values are lists of keywords (str) that trigger that intent.
                            If None, a default mapping is used.
            default_intent: The string intent name to assign if no specific intent
                            is detected based on the provided keywords.
            confidence_score: A simulated float score representing the confidence of
                              the classification. This is static for simulation.
        """
        self._default_intent: str = default_intent
        self._confidence_score: float = confidence_score

        # Establish default intent mapping if none is provided
        self._intent_mapping: Dict[str, List[str]] = intent_mapping if intent_mapping is not None else {
            "greeting": ["hello", "hi", "hey", "good morning", "good evening", "how are you"],
            "farewell": ["bye", "goodbye", "see you", "later", "talk to you soon"],
            "ask_product_info": ["product", "item", "what is", "tell me about", "details on"],
            "place_order": ["order", "buy", "purchase", "add to cart", "get me"],
            "check_order_status": ["status", "track", "where is my", "my delivery"],
            "support_request": ["help", "support", "issue", "problem", "assist me"]
        }
        logger.debug(f"[{self.node_name}] Initialized with intent mapping: {self._intent_mapping}")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this node."""
        return "IntentClassifierNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classifies the intent of the input data (expected to be a string query).

        Args:
            data: The input data, typically a string representing a user query
                  or a piece of text requiring intent classification.
            context: A dictionary containing contextual information relevant to the
                     current execution flow. Can be used for dynamic configuration
                     or additional logging details.

        Returns:
            A dictionary containing the classification results:
            - 'original_query': The input query string.
            - 'classified_intent': The determined intent as a string.
            - 'confidence': A simulated confidence score (float).

        Raises:
            TypeError: If the input 'data' is not a string, as this node expects text.
        """
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected 'str', "
                f"got '{type(data).__name__}'. Data: {data!r}"
            )
            raise TypeError(f"Input 'data' for {self.node_name} must be a string. Got {type(data).__name__}.")

        query_lower = data.lower()
        classified_intent = self._default_intent

        for intent_name, keywords in self._intent_mapping.items():
            for keyword in keywords:
                if keyword in query_lower:
                    classified_intent = intent_name
                    logger.debug(
                        f"[{self.node_name}] Matched keyword '{keyword}' for intent '{intent_name}' "
                        f"in query: '{data}'"
                    )
                    break  # Found a match for this intent, move to the next intent category
            if classified_intent != self._default_intent:
                break  # An intent was found, stop checking other intent categories

        if classified_intent == self._default_intent:
            logger.warning(
                f"[{self.node_name}] No specific intent found for query: '{data}'. "
                f"Defaulting to '{self._default_intent}'."
            )
        else:
            logger.info(
                f"[{self.node_name}] Classified intent '{classified_intent}' for query: '{data}'."
            )

        return {
            "original_query": data,
            "classified_intent": classified_intent,
            "confidence": self._confidence_score
        }
