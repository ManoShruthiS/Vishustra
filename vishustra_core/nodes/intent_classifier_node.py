import logging
import re
from typing import Any, Dict, Optional

# Assuming this import path based on the project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class IntentClassifierNode(BaseNode):
    """
    A Vishustra processing node that classifies the intent of an input text.

    This node simulates intent detection by matching keywords or regex patterns
    against a predefined intent map. In a production system, this would typically
    involve an NLP model or a dedicated intent classification service, potentially
    configured via the 'context' or during node initialization.
    """

    def __init__(self, intent_map: Optional[Dict[str, str]] = None):
        """
        Initializes the IntentClassifierNode with an optional intent map.

        Args:
            intent_map (Optional[Dict[str, str]]): A dictionary where keys are
                                                   regex patterns and values are
                                                   corresponding intent names.
                                                   If None, a default map is used.
        """
        # Default simple intent map with regex for robust keyword matching
        self._intent_map = intent_map if intent_map is not None else {
            r"\b(order|buy|purchase|procure)\b": "PLACE_ORDER",
            r"\b(support|help|assistance|issue|trouble)\b": "GET_SUPPORT",
            r"\b(cancel|return|refund|discontinue)\b": "MANAGE_SUBSCRIPTION_OR_ORDER",
            r"\b(status|track|where is|delivery)\b": "CHECK_STATUS",
            r"\b(hello|hi|hey|greeting|good morning)\b": "GREETING",
            r"\b(thank you|thanks|appreciate)\b": "THANK_YOU",
        }
        logger.debug(f"IntentClassifierNode initialized with intent map: {list(self._intent_map.keys())}")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "IntentClassifierNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data (expected to be a string) to classify its intent.

        The method iterates through the configured intent map, applying regex patterns
        to the input text to identify the most probable intent.

        Args:
            data (Any): The input data, expected to be a string representing a user query
                        or utterance.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the processing node. This could include
                                       session data, user preferences, or global
                                       configuration relevant to intent classification.

        Returns:
            Dict[str, Any]: A dictionary containing 'intent' (str) and 'confidence' (float).
                            Returns 'UNKNOWN' intent with low confidence if no match is found.

        Raises:
            TypeError: If the input data is not a string, indicating an invalid input type.
            Exception: For any other unexpected errors encountered during the classification process.
        """
        if not isinstance(data, str):
            error_msg = (
                f"Invalid input data type for {self.node_name}. "
                f"Expected 'str', received '{type(data).__name__}'."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

        text_lower = data.lower()
        classified_intent = "UNKNOWN"
        confidence = 0.1  # Default low confidence for an unknown intent

        try:
            for pattern, intent_name in self._intent_map.items():
                if re.search(pattern, text_lower):
                    classified_intent = intent_name
                    confidence = 0.9  # High confidence for a direct pattern match
                    logger.debug(f"Intent '{classified_intent}' detected for input: '{data}' using pattern '{pattern}'.")
                    break  # Found the first matching intent, prioritizing order in map
            else:
                logger.info(f"No specific intent pattern matched for input: '{data}'. Classified as '{classified_intent}'.")

        except re.error as e:
            # Handle potential issues with regex patterns themselves
            logger.exception(f"Regex error encountered in {self.node_name} while processing data: '{data}'. Error: {e}")
            raise ValueError(f"Invalid regex pattern in intent map configuration: {e}") from e
        except Exception as e:
            # Catch any other unforeseen issues during the classification loop
            logger.exception(f"An unexpected error occurred during intent classification in {self.node_name} for data: '{data}'.")
            raise RuntimeError(f"Failed to classify intent due to an internal error: {e}") from e

        result = {
            "intent": classified_intent,
            "confidence": confidence
        }
        logger.info(f"IntentClassifierNode processed input: '{data}' -> Result: {result}")
        return result