import logging
from typing import Any, Dict

# Assuming BaseNode is available at this path as per project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class IntentClassifierNode(BaseNode):
    """
    A Vishustra processing node that classifies the intent of a given text input.
    This node simulates intent classification based on predefined keywords
    and can optionally extend its mapping via the processing context.
    """

    # Predefined intent mapping for simulation purposes.
    # In a real-world scenario, this would involve an actual ML model or NLU service.
    _intent_map: Dict[str, str] = {
        "hello": "greeting.greet",
        "hi": "greeting.greet",
        "hey": "greeting.greet",
        "good morning": "greeting.greet",
        "good afternoon": "greeting.greet",
        "good evening": "greeting.greet",
        "order": "shopping.place_order",
        "buy": "shopping.place_order",
        "purchase": "shopping.place_order",
        "track": "shopping.track_order",
        "status": "shopping.track_order",
        "where is my package": "shopping.track_order",
        "support": "support.get_help",
        "help": "support.get_help",
        "issue": "support.get_help",
        "problem": "support.get_help",
        "cancel": "shopping.cancel_order",
        "revoke": "shopping.cancel_order",
        "change order": "shopping.modify_order",
        "update order": "shopping.modify_order",
        "what can you do": "query.capabilities",
        "features": "query.capabilities",
        "thank you": "greeting.thank",
        "thanks": "greeting.thank",
    }
    _default_intent: str = "fallback.unclear_intent"

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "IntentClassifierNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data (expected to be a string utterance)
        to classify its intent based on internal keyword mappings.
        The `context` dictionary can provide a 'custom_intent_map' to extend
        or override the node's default intent classification rules.

        Args:
            data: The input data, expected to be a string representing a user utterance.
                  Non-string input will raise a TypeError.
            context: A dictionary containing contextual information for processing.
                     It can include a 'custom_intent_map' (Dict[str, str])
                     to dynamically adjust intent classification.

        Returns:
            A string representing the classified intent. If no specific intent is found
            or the input is empty, a default fallback intent is returned.

        Raises:
            TypeError: If the input 'data' is not a string.
        """
        if not isinstance(data, str):
            logger.error(
                f"IntentClassifierNode received non-string data: {type(data).__name__}. Expected string."
            )
            raise TypeError(
                f"IntentClassifierNode expects 'data' to be a string, but received {type(data).__name__}"
            )

        utterance = data.lower().strip()
        classified_intent = self._default_intent
        
        # Prepare the active intent map, allowing context to provide overrides or extensions
        active_intent_map = self._intent_map.copy()
        if "custom_intent_map" in context and isinstance(context["custom_intent_map"], dict):
            # Ensure custom map keywords are also lowercased for consistency
            active_intent_map.update({k.lower(): v for k, v in context["custom_intent_map"].items()})
            logger.debug("IntentClassifierNode is using a custom_intent_map provided in the context.")

        if not utterance:
            logger.info("Received an empty utterance for classification. Assigning default intent.")
            return classified_intent  # Returns _default_intent

        # Simulate intent classification by checking for keywords
        for keyword, intent in active_intent_map.items():
            if keyword in utterance:
                classified_intent = intent
                logger.info(
                    f"Classified intent '{intent}' for utterance starting with "
                    f"'{utterance[:75]}{'...' if len(utterance) > 75 else ''}' "
                    f"based on keyword '{keyword}'."
                )
                break  # Assign the first matching intent found

        if classified_intent == self._default_intent:
            logger.warning(
                f"No specific intent classified for utterance: "
                f"'{utterance[:75]}{'...' if len(utterance) > 75 else ''}'. "
                f"Assigning default intent: '{self._default_intent}'."
            )
            
        return classified_intent