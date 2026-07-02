import logging
from typing import Any, Dict, List, Optional

# Assuming this import path based on project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class IntentClassifierNode(BaseNode):
    """
    A processing node designed to classify the intent of a given text input.
    It uses a predefined set of patterns/keywords to map input text to specific intents.
    This provides a configurable and modular way to integrate intent recognition
    into Vishustra workflows.
    """

    def __init__(self, intents_config: Dict[str, List[str]], default_intent: str = "unclear_intent"):
        """
        Initializes the IntentClassifierNode with a configuration for intent classification.

        Args:
            intents_config: A dictionary where keys are intent names (str) and values
                            are lists of keywords or phrases (str) associated with that intent.
                            The node will attempt to match these keywords/phrases in the input data.
                            Matching is case-insensitive.
            default_intent: The intent to return if no specific intent matches the input data.
                            Defaults to "unclear_intent".

        Raises:
            TypeError: If `intents_config` or its elements, or `default_intent` are not of the
                       expected types.
        """
        if not isinstance(intents_config, dict):
            raise TypeError("The 'intents_config' parameter must be a dictionary.")
        
        processed_config: Dict[str, List[str]] = {}
        for intent_name, patterns in intents_config.items():
            if not isinstance(intent_name, str):
                raise TypeError(f"Intent name '{intent_name}' in 'intents_config' must be a string.")
            if not isinstance(patterns, list):
                raise TypeError(f"Patterns for intent '{intent_name}' must be a list of strings.")
            if not all(isinstance(p, str) for p in patterns):
                raise TypeError(f"All patterns within the list for intent '{intent_name}' must be strings.")
            
            processed_config[intent_name.lower()] = [p.lower() for p in patterns]

        if not isinstance(default_intent, str):
            raise TypeError("The 'default_intent' parameter must be a string.")

        self._intents_config = processed_config
        self._default_intent = default_intent.lower()
        logger.debug(f"IntentClassifierNode initialized with {len(self._intents_config)} intents.")

    @property
    def node_name(self) -> str:
        """Returns the programmatic name of the node."""
        return "IntentClassifierNode"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Processes the input data to identify its intent based on configured patterns.

        The method expects the input `data` to be a string representing an utterance
        or query. It performs a case-insensitive keyword match against the
        `intents_config` provided during initialization.

        Args:
            data: The input text (expected to be a string) for intent classification.
            context: A dictionary containing contextual information. While not
                     directly used in this basic implementation, it is available
                     for more complex future enhancements.

        Returns:
            The identified intent as a string. Returns `self._default_intent` if no
            specific intent matches the input data.

        Raises:
            ValueError: If the input `data` is not a string, indicating an
                        unsupported input type for this node.
        """
        if not isinstance(data, str):
            logger.error(f"IntentClassifierNode received invalid data type. Expected 'str', got '{type(data).__name__}'.")
            raise ValueError(
                f"IntentClassifierNode expects string input for data. Received type: {type(data).__name__}"
            )

        text_lower = data.lower()
        matched_intent: Optional[str] = None

        for intent, patterns in self._intents_config.items():
            for pattern in patterns:
                if pattern in text_lower:
                    matched_intent = intent
                    break  # Found a pattern for this intent
            if matched_intent:
                break  # Found an intent, no need to check further

        if matched_intent:
            logger.info(f"Classified intent for input '{data[:75]}...' as '{matched_intent}'.")
            return matched_intent
        else:
            logger.info(f"No specific intent matched for input '{data[:75]}...'. Defaulting to '{self._default_intent}'.")
            return self._default_intent