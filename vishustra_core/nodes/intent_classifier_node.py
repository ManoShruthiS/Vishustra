from vishustra_core.nodes.base_node import BaseNode
from typing import Any, Dict, Union
import logging

logger = logging.getLogger(__name__)

class IntentClassifierNode(BaseNode):
    """
    A Vishustra processing node responsible for classifying the intent
    of a given text input.

    This node simulates intent classification based on keywords for demonstration
    purposes. In a production environment, this would typically integrate
    with an actual machine learning model or a dedicated Natural Language
    Understanding (NLU) service to provide high-fidelity intent recognition.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node.
        """
        return "IntentClassifierNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Union[str, float]]:
        """
        Processes the input data (expected to be a string utterance) to classify
        its underlying intent and assign a confidence score.

        Args:
            data (Any): The primary input data for the node. For this node, it is
                        expected to be a string representing a user utterance or query.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       relevant to the current orchestration flow.
                                       This can include session data, user preferences,
                                       or previous node outputs.

        Returns:
            Dict[str, Union[str, float]]: A dictionary containing the classified intent
                                          as a string and its associated confidence score
                                          as a float.
                                          Example: {"intent": "book_flight", "confidence": 0.98}

        Raises:
            TypeError: If the input `data` is not a string, indicating an invalid
                       input type for intent classification.
        """
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected 'str', got '{type(data).__name__}'. "
                f"Failed to classify intent. Context keys: {list(context.keys())}"
            )
            raise TypeError(
                f"IntentClassifierNode requires input 'data' to be a string. "
                f"Received type: {type(data).__name__}."
            )

        utterance = data.lower().strip()
        logger.debug(f"[{self.node_name}] Starting intent classification for utterance: '{utterance[:100]}{'...' if len(utterance) > 100 else ''}'")

        # Simulate intent classification based on a simple keyword matching logic.
        # This can be expanded or replaced with an external NLU model integration.
        intent: str
        confidence: float

        if ("book" in utterance and "flight" in utterance) or \
           ("reserve" in utterance and ("ticket" in utterance or "seat" in utterance)):
            intent = "book_flight"
            confidence = 0.95
        elif "weather" in utterance or "forecast" in utterance or "temperature" in utterance:
            intent = "get_weather"
            confidence = 0.90
        elif "help" in utterance or "support" in utterance or "assist" in utterance or "faq" in utterance:
            intent = "get_help"
            confidence = 0.88
        elif ("cancel" in utterance or "change" in utterance) and \
             ("booking" in utterance or "reservation" in utterance or "flight" in utterance):
            intent = "manage_booking"
            confidence = 0.92
        elif "price" in utterance or "cost" in utterance or "how much" in utterance:
            intent = "inquire_price"
            confidence = 0.85
        else:
            intent = "general_query"
            confidence = 0.70

        result = {"intent": intent, "confidence": confidence}
        logger.info(
            f"[{self.node_name}] Classified utterance as intent: '{intent}' "
            f"with confidence: {confidence:.2f}"
        )
        return result