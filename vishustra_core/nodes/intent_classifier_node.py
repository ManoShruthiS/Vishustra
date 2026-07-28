import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class IntentClassifierNode(BaseNode):
    """
    A Vishustra node designed to classify the user's intent from an input utterance.

    This node takes a string representing user input and, through its `process` method,
    determines a high-level intent. For demonstration and modularity, this
    implementation uses a simple keyword-based heuristic. In a production
    environment, this would typically interface with an advanced NLU model.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "IntentClassifierNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data (user utterance) to classify its primary intent.

        Args:
            data (Any): The input data, expected to be a string representing
                        a user's natural language utterance.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       that might be relevant for intent classification,
                                       though not directly used in this simulated version.

        Returns:
            Dict[str, Any]: A dictionary containing:
                            - "original_utterance": The cleaned input utterance.
                            - "classified_intent": A string representing the determined intent
                                                   (e.g., "BOOK_FLIGHT", "GET_WEATHER",
                                                   "UNCLEAR_INTENT").
                            - Optionally, an "error" key if input validation fails.
        """
        if not isinstance(data, str):
            logger.warning(
                f"[{self.node_name}] Invalid input type received. Expected 'str', got '{type(data).__name__}'."
            )
            return {
                "original_utterance": data,
                "classified_intent": "INVALID_INPUT_TYPE",
                "error": "Input data must be a string utterance.",
            }

        utterance = data.strip()
        if not utterance:
            logger.info(f"[{self.node_name}] Received empty or whitespace-only utterance.")
            return {
                "original_utterance": data,
                "classified_intent": "EMPTY_UTTERANCE",
            }

        # Simulate intent classification with a simple keyword-based approach
        classified_intent = "UNCLEAR_INTENT"
        utterance_lower = utterance.lower()

        if any(keyword in utterance_lower for keyword in ["book", "flight", "ticket", "travel"]):
            classified_intent = "BOOK_FLIGHT"
        elif any(keyword in utterance_lower for keyword in ["weather", "forecast", "temperature", "climate"]):
            classified_intent = "GET_WEATHER"
        elif any(keyword in utterance_lower for keyword in ["hello", "hi", "hey", "greetings"]):
            classified_intent = "GREETING"
        elif any(keyword in utterance_lower for keyword in ["thank", "thanks", "appreciate"]):
            classified_intent = "THANK_YOU"
        elif any(keyword in utterance_lower for keyword in ["help", "support", "assist", "problem"]):
            classified_intent = "REQUEST_HELP"
        elif any(keyword in utterance_lower for keyword in ["cancel", "stop", "abort"]):
            classified_intent = "CANCEL_OPERATION"
        elif any(keyword in utterance_lower for keyword in ["what is", "tell me about", "information on"]):
            classified_intent = "GET_INFORMATION"

        logger.debug(
            f"[{self.node_name}] Classified intent for utterance '{utterance}' as '{classified_intent}'."
        )

        return {
            "original_utterance": utterance,
            "classified_intent": classified_intent,
        }