import logging
from typing import Any, Dict, List

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class IntentClassifierNode(BaseNode):
    """
    A Vishustra node responsible for classifying the primary intent of a given text input.

    This node simulates intent classification based on a configurable mapping of keywords
    to intents. It can be extended to integrate with external Natural Language Understanding (NLU)
    services or local machine learning models for more sophisticated intent detection.
    """

    def __init__(self, intent_map: Dict[str, List[str]] = None, default_intent: str = "unknown_intent"):
        """
        Initializes the IntentClassifierNode with a specific intent mapping and a fallback intent.

        Args:
            intent_map (Dict[str, List[str]], optional): A dictionary where keys are intent names
                                                        (e.g., "greet", "order_status") and values
                                                        are lists of keywords or phrases that,
                                                        if found in the input text, trigger that intent.
                                                        If None, a sensible default mapping is used.
            default_intent (str, optional): The name of the intent to be returned if no specific
                                            intent is matched based on the provided `intent_map`.
                                            Defaults to "unknown_intent".
        """
        self._default_intent = default_intent
        self._intent_map = intent_map if intent_map is not None else self._build_default_intent_map()
        logger.debug(f"IntentClassifierNode initialized with {len(self._intent_map)} intent definitions.")

    def _build_default_intent_map(self) -> Dict[str, List[str]]:
        """
        Provides a default, illustrative mapping of keywords to intents.
        This can be overridden by providing a custom `intent_map` during instantiation.
        """
        return {
            "greet": ["hello", "hi", "hey", "good morning", "good evening", "greetings"],
            "farewell": ["bye", "goodbye", "see you", "later", "adios"],
            "order_status": ["where is my order", "order status", "track order", "my package", "delivery info"],
            "account_info": ["my account", "change password", "update profile", "login issue"],
            "help": ["help me", "support", "assistance", "can you help"],
            "complaint": ["i am unhappy", "this is bad", "complaint about", "dissatisfied"],
            "thanks": ["thank you", "thanks", "much appreciated"]
        }

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this node.
        """
        return "IntentClassifierNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data (expected to be a user query string) to classify its intent.

        The method iterates through the configured intent map, searching for keywords
        within the normalized input text. If a match is found, the corresponding intent
        and a simulated confidence score are returned. If no specific intent is matched,
        the `default_intent` is returned with a lower confidence.

        Args:
            data (Any): The input data, which must be a string representing the text query.
            context (Dict[str, Any]): The current processing context, available for broader
                                      orchestration but not directly used by this specific
                                      classification logic.

        Returns:
            Dict[str, Any]: A dictionary containing the classified intent and a confidence score.
                            Example: `{"intent": "greet", "confidence": 0.9}`

        Raises:
            ValueError: If the input `data` is not a string, indicating an invalid input type.
            Exception: For any unexpected errors encountered during the classification process.
        """
        if not isinstance(data, str):
            logger.error(f"IntentClassifierNode received invalid input type. Expected 'str', got '{type(data).__name__}'.")
            raise ValueError(f"IntentClassifierNode requires string input for classification, received {type(data).__name__}.")

        input_text = data.lower().strip()
        logger.info(f"Attempting intent classification for input: '{input_text[:100]}{'...' if len(input_text) > 100 else ''}'")

        classified_intent = self._default_intent
        confidence = 0.1  # Default low confidence for unknown intent

        try:
            for intent_name, keywords_list in self._intent_map.items():
                for keyword in keywords_list:
                    if keyword in input_text:
                        classified_intent = intent_name
                        confidence = 0.9  # Simulate high confidence for a direct keyword match
                        logger.debug(f"Matched keyword '{keyword}' with intent '{intent_name}'.")
                        break  # Found a keyword for this intent, no need to check others for it
                if classified_intent != self._default_intent:
                    break  # An intent was successfully classified, stop checking other intents

            if classified_intent == self._default_intent:
                logger.info(f"No specific intent matched for input text. Defaulting to '{self._default_intent}'.")
            else:
                logger.info(f"Successfully classified intent as '{classified_intent}' with confidence {confidence:.2f}.")

            return {"intent": classified_intent, "confidence": confidence}

        except Exception as e:
            logger.exception(f"An unexpected error occurred during intent classification for input: '{input_text}'.")
            raise Exception(f"Failed to process intent classification: {e}") from e