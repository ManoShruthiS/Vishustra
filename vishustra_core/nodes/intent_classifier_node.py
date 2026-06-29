import logging
from typing import Any, Dict, Optional

# Assuming this import path from project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class IntentClassifierNode(BaseNode):
    """
    A Vishustra processing node designed to classify the intent of a given text input.

    This node simulates intent classification based on a set of predefined keyword patterns.
    It expects a string as input data, representing a user query, and returns a dictionary
    containing the classified intent, a simulated confidence score, and the original query.

    The classification logic is configurable through an intent_map provided during
    initialization, allowing for flexible adaptation to various domain-specific intents.
    """

    _DEFAULT_INTENT = "unclear_intent"
    _DEFAULT_CONFIDENCE = 0.5
    _MATCH_CONFIDENCE = 0.9

    def __init__(self, intent_map: Optional[Dict[str, str]] = None):
        """
        Initializes the IntentClassifierNode with an optional, custom intent mapping.

        Args:
            intent_map (Optional[Dict[str, str]]): A dictionary where keys are keywords
                or short phrases (case-insensitive for matching) and values are the
                corresponding intent names (e.g., {"track order": "track_order"}).
                If None, a default internal mapping is used. Custom patterns will
                override or extend default ones if provided.
        """
        self._intent_patterns = self._build_intent_patterns(intent_map)
        logger.debug(f"[{self.node_name}] Initialized with intent patterns: {self._intent_patterns}")

    def _build_intent_patterns(self, custom_map: Optional[Dict[str, str]]) -> Dict[str, str]:
        """
        Constructs the internal intent patterns, merging default patterns with any
        provided custom ones. Keywords are stored in lowercase for case-insensitive matching.
        """
        # In a production system, this would typically involve a trained machine learning
        # model or a more sophisticated natural language understanding (NLU) service.
        # This implementation uses a simple keyword-based approach for simulation purposes.
        default_patterns = {
            "order": "place_order",
            "purchase": "place_order",
            "buy": "place_order",
            "track": "track_order",
            "status": "track_order",
            "where is": "track_order",
            "cancel": "cancel_return",
            "return": "cancel_return",
            "refund": "cancel_return",
            "hello": "greeting",
            "hi": "greeting",
            "hey": "greeting",
            "help": "request_help",
            "support": "request_help",
            "assist": "request_help",
            "thank you": "gratitude",
            "thanks": "gratitude",
            "weather": "get_weather",
            "forecast": "get_weather",
        }
        
        # Merge custom patterns, allowing them to override defaults
        final_patterns = {k.lower(): v for k, v in default_patterns.items()}
        if custom_map:
            final_patterns.update({k.lower(): v for k, v in custom_map.items()})
            
        return final_patterns

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "IntentClassifierNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data, classifying its intent based on predefined patterns.

        The `data` input is expected to be a string representing a user query. The method
        iterates through its known intent patterns and attempts to find a match within
        the (case-normalized) query. The first matching pattern determines the intent.

        Args:
            data (Any): The input data to be classified. Expected to be a string query.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the current processing flow. This node
                                       primarily uses `data` but receives context.

        Returns:
            Dict[str, Any]: A dictionary containing the processing result:
                            - 'original_query': The raw string input that was processed.
                            - 'intent': The classified intent as a string (e.g., "track_order").
                            - 'confidence': A simulated confidence score (float) indicating
                                            the certainty of the classification.
                            - 'error': Optional string, present if a non-critical issue occurred.

        Raises:
            TypeError: If the input 'data' is not a string, as this node's core
                       functionality relies on string manipulation.
        """
        logger.info(f"[{self.node_name}] Starting intent classification for data (truncated): '{str(data)[:100]}'")

        if not isinstance(data, str):
            error_msg = (
                f"[{self.node_name}] Invalid input data type. "
                f"Expected 'str' for intent classification, but received '{type(data).__name__}'."
            )
            logger.error(error_msg)
            # Raising an exception here is crucial as downstream nodes might depend on
            # the structure of the output, and invalid input prevents meaningful processing.
            raise TypeError(error_msg)

        query = data.lower()
        classified_intent = self._DEFAULT_INTENT
        confidence = self._DEFAULT_CONFIDENCE

        # Iterate through patterns to find the first match
        for pattern, intent in self._intent_patterns.items():
            if pattern in query:
                classified_intent = intent
                confidence = self._MATCH_CONFIDENCE # Assign higher confidence for a direct keyword match
                logger.debug(f"[{self.node_name}] Matched pattern '{pattern}' for intent '{intent}' in query: '{data}'")
                break # Stop at the first confident match

        if classified_intent == self._DEFAULT_INTENT:
            logger.info(
                f"[{self.node_name}] No specific intent classified for query: "
                f"'{data}'. Defaulting to '{self._DEFAULT_INTENT}'."
            )
        else:
            logger.info(
                f"[{self.node_name}] Classified intent as '{classified_intent}' "
                f"for query: '{data}' with confidence {confidence:.2f}."
            )

        result = {
            "original_query": data,
            "intent": classified_intent,
            "confidence": confidence,
        }
        
        # In a real-world scenario, context might be updated here, e.g.:
        # context['last_classified_intent'] = classified_intent
        # context['intent_confidence'] = confidence

        return result