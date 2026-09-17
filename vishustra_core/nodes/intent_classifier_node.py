import logging
from typing import Any, Dict, List, Optional

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class IntentClassifierNode(BaseNode):
    """
    A Vishustra processing node that classifies the intent of a given text input.

    This node provides a configurable, keyword-based approach for intent classification,
    suitable for initial routing or simpler use cases within an orchestration flow.
    It can be extended or replaced with more sophisticated machine learning models
    for production-grade intent recognition.
    """

    DEFAULT_INTENT_MAP: Dict[str, List[str]] = {
        "greeting": ["hello", "hi", "hey", "namaste", "good morning", "good evening"],
        "question": ["what", "how", "why", "when", "where", "which", "who"],
        "complaint": ["complaint", "not working", "broken", "unhappy", "terrible", "worst", "issue"],
        "praise": ["great", "excellent", "love", "amazing", "awesome", "best", "wonderful"],
        "request": ["please", "can you", "i need", "i want", "could you", "help me"],
        "gratitude": ["thanks", "thank you", "thankyou", "appreciate"],
        "apology": ["sorry", "apologize", "my bad", "regret"],
    }

    def __init__(
        self,
        intent_map: Dict[str, List[str]] = None,
        default_intent: str = "unknown_intent"
    ):
        """
        Initializes the IntentClassifierNode with a mapping of intents to keywords.

        Args:
            intent_map: A dictionary where keys are intent names (str) and values
                        are lists of keywords or phrases (str) associated with that intent.
                        The comparison is case-insensitive. If None, a sensible default
                        map covering common conversation intents is used.
            default_intent: The intent to return if no specific intent is matched
                            by the configured keywords. Defaults to "unknown_intent".

        Raises:
            TypeError: If `intent_map` is not a dictionary, or if its structure
                       (keys, values, elements within value lists) is incorrect.
            TypeError: If `default_intent` is not a string.
        """
        if intent_map is None:
            intent_map = {k: list(v) for k, v in self.DEFAULT_INTENT_MAP.items()}
        if not isinstance(intent_map, dict):
            raise TypeError("Configuration error: intent_map must be a dictionary.")
        
        processed_intent_map = {}
        for intent, keywords in intent_map.items():
            if not isinstance(intent, str):
                raise TypeError(f"Configuration error: Intent names in intent_map must be strings. Got {type(intent)}.")
            if not isinstance(keywords, list):
                raise TypeError(f"Configuration error: Keywords for intent '{intent}' must be a list. Got {type(keywords)}.")
            if not all(isinstance(k, str) for k in keywords):
                raise TypeError(f"Configuration error: All keywords for intent '{intent}' must be strings.")
            
            processed_intent_map[intent.lower()] = [kw.lower() for kw in keywords]

        self._intent_map = processed_intent_map
        
        if not isinstance(default_intent, str):
            raise TypeError("Configuration error: default_intent must be a string.")
        self._default_intent = default_intent.lower()
        
        logger.debug(
            f"[{self.node_name}] Initialized with configured intents: "
            f"{list(self._intent_map.keys())}. Default intent: '{self._default_intent}'."
        )

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "IntentClassifierNode"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Classifies the intent of the input text data based on predefined keywords.

        The classification prioritizes the first intent whose keywords are found
        in the input text. If no keywords match, the configured default intent
        is returned.

        Args:
            data: The input text (string) to be classified.
            context: A dictionary containing contextual information for processing.
                     This node logs the context but does not use it for classification logic.

        Returns:
            A string representing the classified intent.

        Raises:
            ValueError: If the input data is not a string.
            Exception: For any unexpected errors that occur during the classification process.
        """
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected 'str', "
                f"but received '{type(data).__name__}' for processing."
            )
            raise ValueError(
                f"{self.node_name} expects string data for classification, "
                f"but received {type(data).__name__}."
            )

        query_text = data.lower().strip()
        classified_intent: Optional[str] = None

        logger.info(f"[{self.node_name}] Attempting to classify intent for query: '{query_text}'.")
        logger.debug(f"[{self.node_name}] Received context: {context}.")

        try:
            for intent, keywords in self._intent_map.items():
                for keyword in keywords:
                    if keyword in query_text:
                        classified_intent = intent
                        break  # Found a keyword for this intent, proceed with this intent
                if classified_intent:
                    break  # Intent found, no need to check other intents

            if classified_intent:
                logger.info(
                    f"[{self.node_name}] Successfully classified intent "
                    f"as '{classified_intent}' for query: '{query_text}'."
                )
                return classified_intent
            else:
                logger.info(
                    f"[{self.node_name}] No specific intent keywords matched for query: "
                    f"'{query_text}'. Returning default intent: '{self._default_intent}'."
                )
                return self._default_intent
        except Exception as e:
            logger.exception(
                f"[{self.node_name}] An unexpected error occurred during intent classification "
                f"for query: '{query_text}'."
            )
            # Re-raise to ensure orchestration framework can handle or log critical issues
            raise Exception(f"Failed to classify intent in {self.node_name}: {e}") from e

