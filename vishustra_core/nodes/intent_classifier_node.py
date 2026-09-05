from vishustra_core.nodes.base_node import BaseNode
from typing import Any, Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class IntentClassifierNode(BaseNode):
    """
    A Vishustra node that classifies the intent of a given text input.

    It uses a set of predefined rules (keywords/phrases mapped to intents)
    to determine the user's intent. Rules can be provided at instantiation
    or dynamically through the context of the process method.
    """

    _DEFAULT_INTENT_RULES: Dict[str, List[str]] = {
        "greeting": ["hello", "hi", "hey", "good morning", "good evening"],
        "farewell": ["bye", "goodbye", "see you", "farewell"],
        "order_status": ["where is my order", "order status", "track my package", "delivery date"],
        "product_info": ["tell me about", "what is", "product details", "specifications of", "info on"],
        "account_management": ["change password", "update profile", "my account"],
        "support_request": ["help me", "support", "i have a problem", "contact agent"],
    }
    _DEFAULT_FALLBACK_INTENT: str = "general_query"

    def __init__(self,
                 initial_intent_rules: Dict[str, List[str]] = None,
                 default_fallback_intent: str = None) -> None:
        """
        Initializes the IntentClassifierNode with optional intent rules.

        Args:
            initial_intent_rules: A dictionary mapping intent names to lists of keywords/phrases.
                                  If None, default rules are used.
            default_fallback_intent: The intent to return if no match is found.
                                     If None, '_DEFAULT_FALLBACK_INTENT' is used.
        """
        self._intent_rules = initial_intent_rules if initial_intent_rules is not None else self._DEFAULT_INTENT_RULES
        self._default_fallback_intent = default_fallback_intent if default_fallback_intent is not None else self._DEFAULT_FALLBACK_INTENT
        logger.info(f"Initialized IntentClassifierNode with {len(self._intent_rules)} default intent rules.")
        logger.debug(f"Default intents: {list(self._intent_rules.keys())}")


    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "IntentClassifier"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Processes the input data (a user query string) to classify its intent.

        The classification rules can be overridden or extended via the 'context' dictionary.
        - `context['intent_rules']`: A dictionary of intent rules to use for this specific call.
                                     If provided, it completely replaces the node's configured rules.
        - `context['default_fallback_intent']`: The intent to return if no match is found.
                                                 If provided, it overrides the node's configured default.

        Args:
            data: The input text string (e.g., user query) to classify.
            context: A dictionary containing additional runtime information or configuration,
                     such as dynamic intent rules.

        Returns:
            A string representing the classified intent. Returns the fallback intent
            if no specific intent is detected.

        Raises:
            ValueError: If the input 'data' is not a string.
            TypeError: If 'intent_rules' provided in context is not a dictionary.
        """
        if not isinstance(data, str):
            logger.error(f"IntentClassifierNode received non-string data: {type(data)}. Expected string.")
            raise ValueError(f"Input data must be a string, but received {type(data).__name__}.")

        current_intent_rules = self._intent_rules
        fallback_intent = self._default_fallback_intent

        if 'intent_rules' in context:
            if not isinstance(context['intent_rules'], dict):
                logger.error(f"Context 'intent_rules' must be a dictionary, but received {type(context['intent_rules']).__name__}.")
                raise TypeError("Context 'intent_rules' must be a dictionary.")
            current_intent_rules = context['intent_rules']
            logger.debug("Using dynamic intent rules from context.")
        else:
            logger.debug("Using node's configured intent rules.")

        if 'default_fallback_intent' in context and isinstance(context['default_fallback_intent'], str):
            fallback_intent = context['default_fallback_intent']
            logger.debug(f"Using dynamic fallback intent '{fallback_intent}' from context.")
        else:
            logger.debug(f"Using node's configured fallback intent '{fallback_intent}'.")

        normalized_query = data.lower().strip()
        logger.debug(f"Attempting to classify intent for query: '{normalized_query}'")

        for intent, keywords in current_intent_rules.items():
            for keyword in keywords:
                if keyword.lower() in normalized_query:
                    logger.info(f"Classified intent as '{intent}' for query: '{normalized_query}' (matched keyword: '{keyword}')")
                    return intent

        logger.info(f"No specific intent found for query: '{normalized_query}'. Falling back to '{fallback_intent}'.")
        return fallback_intent
