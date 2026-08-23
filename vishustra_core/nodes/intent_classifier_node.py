import logging
from typing import Any, Dict, Optional

# Assuming BaseNode is located here as per instructions for the import path
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class IntentClassifierNode(BaseNode):
    """
    A Vishustra processing node designed to classify the intent of an input text string.

    This node simulates intent classification using a simple keyword-based matching strategy.
    It identifies an intent by checking if predefined keywords or phrases exist within
    the input text. For production-grade applications, this would typically be replaced
    with integration to a more sophisticated Natural Language Understanding (NLU) service
    or a trained machine learning model.
    """

    def __init__(self, intent_mapping: Optional[Dict[str, str]] = None):
        """
        Initializes the IntentClassifierNode.

        Args:
            intent_mapping (Optional[Dict[str, str]]): A dictionary where keys are
                keywords or phrases (case-insensitive for matching) and values are
                the corresponding intent names (e.g., {"book flight": "book_flight"}).
                If not provided, a default internal mapping will be used.
        """
        self._intent_mapping = intent_mapping if intent_mapping is not None else {
            "hello": "greet",
            "hi": "greet",
            "hey": "greet",
            "book flight": "book_flight",
            "schedule trip": "book_flight",
            "weather forecast": "get_weather",
            "current weather": "get_weather",
            "cancel my order": "cancel_order",
            "remove item": "cancel_order",
            "support": "get_support",
            "help me": "get_support",
            "what's up": "chit_chat",
            "how are you": "chit_chat",
        }
        logger.debug(f"[{self.node_name}] Initialized with intent mapping: {self._intent_mapping}")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "IntentClassifier"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Processes the input data to determine its intent.

        The `data` input is expected to be a string representing a user query or message.
        The method converts the input to lowercase and then attempts to match it
        against the predefined keywords in the `_intent_mapping`. The first matching
        keyword determines the intent. If no keywords match, "unknown_intent" is returned.

        Args:
            data (Any): The input data to classify, typically a string.
            context (Dict[str, Any]): A dictionary providing contextual information.
                                       This node does not actively use the context for
                                       classification but it's available for orchestration
                                       or future extensions (e.g., dynamic config, model access).

        Returns:
            str: The identified intent as a string (e.g., "greet", "book_flight",
                 "unknown_intent").

        Raises:
            TypeError: If the input `data` is not a string.
            RuntimeError: For any unexpected operational failures during classification.
        """
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Received non-string data of type {type(data)}. Expected a string."
            )
            raise TypeError(
                f"[{self.node_name}] Input 'data' must be a string, but received {type(data)}."
            )

        query = data.lower()
        identified_intent = "unknown_intent"

        logger.info(f"[{self.node_name}] Attempting to classify intent for query: '{data}'")

        try:
            # Iterate through the mapping to find a matching keyword
            for keyword_phrase, intent_name in self._intent_mapping.items():
                if keyword_phrase.lower() in query:
                    identified_intent = intent_name
                    logger.info(
                        f"[{self.node_name}] Identified intent '{identified_intent}' "
                        f"for query: '{data}' (matched on keyword: '{keyword_phrase}')"
                    )
                    break  # Found the first match, return this intent
            else:
                logger.info(
                    f"[{self.node_name}] No specific intent matched for query: '{data}'. "
                    f"Defaulting to '{identified_intent}'."
                )

        except Exception as e:
            logger.exception(
                f"[{self.node_name}] An unexpected error occurred during intent "
                f"classification for query: '{data}'."
            )
            # Re-raise as a RuntimeError to signify an operational failure
            raise RuntimeError(
                f"[{self.node_name}] Failed to classify intent for query '{data}': {e}"
            ) from e

        # The identified intent can also be added to the context for downstream nodes
        # if the orchestration logic expects it there, but the primary output is the return value.
        # context['identified_intent'] = identified_intent

        return identified_intent
