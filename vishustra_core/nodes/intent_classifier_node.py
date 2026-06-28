import logging
from typing import Any, Dict, List, Optional
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class IntentClassifierNode(BaseNode):
    """
    A Vishustra processing node that classifies the intent of an input text query.

    This node simulates intent classification using a keyword-based approach.
    It takes a string query as input and attempts to categorize its underlying
    user intent (e.g., "book_flight", "get_weather").
    """

    def __init__(self, intent_map: Optional[Dict[str, List[str]]] = None):
        """
        Initializes the IntentClassifierNode.

        Args:
            intent_map (Optional[Dict[str, List[str]]]): An optional dictionary
                mapping intent names (str) to lists of associated keywords (List[str]).
                If not provided, a default mapping is used.
        """
        self._intent_map = intent_map if intent_map is not None else {
            "book_flight": ["flight", "book", "travel", "ticket"],
            "get_weather": ["weather", "forecast", "temperature", "climate"],
            "play_music": ["play", "music", "song", "tune", "playlist"],
            "greet": ["hello", "hi", "hey", "good morning", "good evening"],
            "goodbye": ["bye", "goodbye", "see you", "farewell"]
        }
        logger.debug(f"IntentClassifierNode initialized with intent map: {list(self._intent_map.keys())}")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "IntentClassifier"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to classify its intent.

        Args:
            data (Any): The input data to be classified. Expected to be a string
                        representing a user query.
            context (Dict[str, Any]): A dictionary containing shared pipeline context
                                     or configuration, not directly used in this
                                     simulation but available for future extensions.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - 'original_query' (str): The original input query.
                - 'intent' (str): The classified intent (e.g., "book_flight", "unknown").
                - 'confidence' (float): A simulated confidence score (0.0 to 1.0).
                - 'error_message' (Optional[str]): An error message if processing failed.
        """
        if not isinstance(data, str):
            error_msg = (
                f"Invalid input data type for IntentClassifier. Expected 'str', "
                f"but received '{type(data).__name__}'. Unable to process intent."
            )
            logger.error(error_msg)
            return {
                "original_query": str(data),
                "intent": "error",
                "confidence": 0.0,
                "error_message": error_msg
            }

        query = data.lower()
        detected_intent = "unknown"
        confidence = 0.0

        for intent, keywords in self._intent_map.items():
            for keyword in keywords:
                if keyword in query:
                    detected_intent = intent
                    confidence = 0.85  # Simulate a reasonable confidence for a match
                    logger.info(
                        f"Query '{data}' classified as intent '{detected_intent}' "
                        f"based on keyword '{keyword}'."
                    )
                    # In a simple keyword-based system, we take the first match.
                    # A real system would use more sophisticated matching or ML models.
                    break
            if detected_intent != "unknown":
                break

        if detected_intent == "unknown":
            logger.info(f"No specific intent detected for query: '{data}'. Falling back to 'unknown'.")

        return {
            "original_query": data,
            "intent": detected_intent,
            "confidence": confidence,
            "error_message": None
        }