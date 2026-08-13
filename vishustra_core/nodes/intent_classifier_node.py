import logging
from typing import Any, Dict, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class IntentClassifierNode(BaseNode):
    """
    A Vishustra processing node designed to classify the intent of an input text.

    This node simulates intent classification based on configurable rules or
    a mock model invocation. It expects the input data to contain text,
    either directly as a string or embedded within a dictionary under a 'text' key.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "IntentClassifierNode"

    def process(self, data: Union[str, Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to classify its underlying intent.

        The classification logic is simulated using a keyword-based matching approach.
        Behavior and classification rules can be dynamically configured via the `context`
        dictionary for each execution.

        Args:
            data: The input data payload. This can be either a string containing
                  the text to classify, or a dictionary that must contain a 'text'
                  key whose value is the target text.
            context: A dictionary providing runtime context and configuration parameters
                     for the classification process. Expected keys include:
                     - 'intent_mapping' (Dict[str, str], optional): A mapping where keys are
                       keywords or phrases (case-insensitive) and values are the
                       corresponding intent names to be assigned if a match is found.
                       Example: `{'book flight': 'travel_booking', 'hello': 'greeting'}`.
                     - 'default_intent_name' (str, optional): The intent name to assign
                       if no specific intent is detected through the mapping. Defaults
                       to 'general_query'.
                     - 'default_confidence' (float, optional): The confidence score to
                       assign for classification. A higher value (e.g., 0.8) is used
                       if an intent is matched, and a lower value (e.g., 0.6) if the
                       default intent is assigned.

        Returns:
            A dictionary representing the processed output. This dictionary will contain
            either the original input data (if it was a dict) or the input text (if it
            was a string, wrapped in a 'text' key), augmented with two new keys:
            'classified_intent' (str) and 'intent_confidence' (float).

        Raises:
            TypeError: If the input `data` is neither a string nor a dictionary,
                       indicating an unsupported data type.
            ValueError: If `data` is a dictionary but critically lacks the 'text' key
                        required for intent extraction.
        """
        input_text: str
        processed_output: Dict[str, Any]

        if isinstance(data, str):
            input_text = data
            processed_output = {"text": data}  # Wrap string data for consistent dictionary output
            logger.debug("Input data identified as a string; wrapping it for consistent output structure.")
        elif isinstance(data, dict):
            if 'text' not in data:
                logger.error("Input data is a dictionary but does not contain the mandatory 'text' key.")
                raise ValueError("Input dictionary for IntentClassifierNode must contain a 'text' key.")
            input_text = str(data['text'])
            processed_output = data.copy()  # Create a mutable copy to augment
        else:
            logger.error(f"Received invalid input data type: {type(data)}. Expected str or dict.")
            raise TypeError(f"IntentClassifierNode requires input 'data' to be a string or a dictionary, got {type(data)}.")

        # Ensure extracted text is a string for processing, handling potential non-string 'text' values
        if not isinstance(input_text, str):
            logger.warning(f"Extracted 'text' value is not a string (type: {type(input_text)}). Attempting conversion.")
            try:
                input_text = str(input_text)
            except Exception as e:
                logger.error(f"Failed to convert extracted text to string: {e}. Proceeding with empty string.")
                input_text = "" # Fallback to prevent further processing errors

        # Retrieve configuration from context with sensible defaults
        intent_mapping: Dict[str, str] = context.get('intent_mapping', {})
        default_intent: str = context.get('default_intent_name', 'general_query')
        default_matched_confidence: float = context.get('default_confidence', 0.8)
        default_unmatched_confidence: float = context.get('default_confidence', 0.6)

        classified_intent: str = default_intent
        intent_confidence: float = default_unmatched_confidence
        
        lower_case_input_text = input_text.lower()
        intent_detected = False

        # Simulate intent detection using keyword matching
        for keyword, assigned_intent in intent_mapping.items():
            if keyword.lower() in lower_case_input_text:
                classified_intent = assigned_intent
                intent_confidence = default_matched_confidence
                intent_detected = True
                logger.info(f"Intent '{assigned_intent}' detected for text '{input_text}' via keyword '{keyword}'.")
                break  # Assign the first matching intent and break

        if not intent_detected:
            logger.info(f"No specific intent detected for text '{input_text}'. Assigning default intent '{default_intent}'.")

        processed_output['classified_intent'] = classified_intent
        processed_output['intent_confidence'] = intent_confidence

        logger.debug(f"Intent classification complete. Assigned intent: '{classified_intent}', confidence: {intent_confidence:.2f}.")
        return processed_output