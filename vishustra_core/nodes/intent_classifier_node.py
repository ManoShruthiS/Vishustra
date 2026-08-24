import logging
from typing import Any, Dict, List, Optional

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class IntentClassifierNode(BaseNode):
    """
    A Vishustra processing node that simulates intent classification for user queries.

    This node takes a user query (string) and attempts to classify its intent
    based on a set of predefined keywords or a mock classification model.
    It returns a dictionary containing the detected intent and a confidence score.
    """

    def __init__(self,
                 intents_map: Optional[Dict[str, List[str]]] = None,
                 default_intent: str = "unknown"):
        """
        Initializes the IntentClassifierNode with a mapping of keywords to intents.

        Args:
            intents_map (Optional[Dict[str, List[str]]]): A dictionary where keys are intent names
                                                         and values are lists of keywords associated with that intent.
                                                         If None, a default map is used.
            default_intent (str): The intent to return if no specific intent is detected.
        """
        self._intents_map = intents_map if intents_map is not None else self._get_default_intents_map()
        self._default_intent = default_intent
        logger.debug(f"IntentClassifierNode initialized with intents: {list(self._intents_map.keys())} "
                     f"and default intent: '{self._default_intent}'")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "Intent Classifier"

    def _get_default_intents_map(self) -> Dict[str, List[str]]:
        """
        Provides a default set of intents and associated keywords for simulation.
        """
        return {
            "book_flight": ["book flight", "flight ticket", "travel to", "plane reservation"],
            "check_status": ["check status", "order status", "delivery status", "where is my package"],
            "customer_support": ["help", "support", "contact us", "problem with", "issue with my order"],
            "greet": ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"],
            "goodbye": ["bye", "goodbye", "see you later"],
            "thank_you": ["thanks", "thank you very much"]
        }

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data (user query) to classify its intent.

        The classification uses a simple keyword matching strategy.
        The `context` dictionary can optionally override the `intents_map`
        and `default_intent` for a specific processing call.

        Args:
            data (Any): The input data, expected to be a string representing the user's query.
            context (Dict[str, Any]): A dictionary containing contextual information.
                                      Can include 'intents_map' (Dict[str, List[str]])
                                      and 'default_intent' (str) to override
                                      the node's initialized configuration.

        Returns:
            Dict[str, Any]: A dictionary containing 'intent' (str) and 'confidence' (float).
                            Returns the configured default intent with 0.0 confidence
                            if no specific intent is detected.

        Raises:
            ValueError: If the input 'data' is not a string.
        """
        logger.info("IntentClassifierNode starting processing.")

        if not isinstance(data, str):
            error_msg = f"Invalid input data type for IntentClassifierNode. Expected string, got {type(data)}."
            logger.error(error_msg)
            raise ValueError(error_msg)

        query = data.lower().strip()
        detected_intent = self._default_intent
        confidence = 0.0

        # Allow context to override classification parameters if provided
        current_intents_map = context.get('intents_map', self._intents_map)
        current_default_intent = context.get('default_intent', self._default_intent)

        # Simulate intent classification using simple keyword matching
        for intent, keywords in current_intents_map.items():
            for keyword in keywords:
                if keyword in query:
                    detected_intent = intent
                    confidence = 0.9  # Assign high confidence for a direct keyword match
                    logger.debug(f"Detected intent '{detected_intent}' for query '{query}' via keyword '{keyword}'.")
                    break  # Found a match, exit inner loop
            if detected_intent != current_default_intent: # An intent was detected, exit outer loop
                break

        if detected_intent == current_default_intent and confidence == 0.0:
            logger.debug(f"No specific intent detected for query '{query}'. Falling back to default intent: '{current_default_intent}'.")

        result = {
            "intent": detected_intent,
            "confidence": confidence
        }

        logger.info(f"IntentClassifierNode finished processing. Query: '{data}', Result: {result}")
        return result