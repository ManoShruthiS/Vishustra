import logging
from typing import Any, Dict, List, Optional
from enum import Enum

# Assuming the project structure places BaseNode here
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class StandardIntents(str, Enum):
    """
    Defines a set of standard intents for classification.
    """
    GREETING = "greeting"
    ASK_QUESTION = "ask_question"
    MAKE_PURCHASE = "make_purchase"
    CANCEL_ORDER = "cancel_order"
    CHANGE_SETTINGS = "change_settings"
    FALLBACK = "fallback"
    ERROR = "error" # For cases where processing fails

class IntentClassifierNode(BaseNode):
    """
    A Vishustra processing node designed to classify the intent of a given text input.

    This node simulates intent classification based on a configurable keyword mapping.
    In a production environment, this would typically integrate with an advanced
    Natural Language Understanding (NLU) model or an external AI service.
    """

    def __init__(self,
                 intent_mapping: Optional[Dict[str, List[str]]] = None,
                 default_intent: str = StandardIntents.FALLBACK.value):
        """
        Initializes the IntentClassifierNode with a mapping of intents to keywords.

        Args:
            intent_mapping (Optional[Dict[str, List[str]]]): A dictionary where keys are intent names
                (strings) and values are lists of keywords (strings) associated with that intent.
                Keywords are case-insensitive for matching. If None, a sensible default mapping
                for common intents is utilized.
            default_intent (str): The intent to be returned if the input text does not
                match any defined keywords. Defaults to `StandardIntents.FALLBACK.value`.
        """
        if not isinstance(default_intent, str):
            raise TypeError("default_intent must be a string.")

        self._default_intent: str = default_intent
        self._intent_mapping: Dict[str, List[str]] = {}

        if intent_mapping is None:
            # Provide a robust default mapping for common scenarios
            self._intent_mapping = {
                StandardIntents.GREETING.value: ["hello", "hi", "hey", "good morning", "good evening", "greetings"],
                StandardIntents.ASK_QUESTION.value: ["what", "how", "when", "why", "who", "where", "question", "tell me about", "inquire"],
                StandardIntents.MAKE_PURCHASE.value: ["buy", "purchase", "order", "get me", "add to cart", "checkout"],
                StandardIntents.CANCEL_ORDER.value: ["cancel", "revoke", "stop order", "undo purchase", "return item"],
                StandardIntents.CHANGE_SETTINGS.value: ["settings", "preferences", "configure", "update profile", "my account"],
            }
        else:
            # Validate and normalize custom intent_mapping
            for intent, keywords in intent_mapping.items():
                if not isinstance(intent, str) or not isinstance(keywords, list):
                    raise TypeError(f"Intent mapping keys must be strings and values must be lists of strings. Got {type(intent)} and {type(keywords)}")
                self._intent_mapping[intent] = [str(k).lower() for k in keywords if isinstance(k, str)]
                if not self._intent_mapping[intent]:
                    logger.warning(f"Intent '{intent}' has an empty or invalid keyword list. It may never be matched.")

        logger.info(f"IntentClassifierNode initialized. Default intent: '{self._default_intent}'. "
                    f"Registered intents: {list(self._intent_mapping.keys())}")

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this processing node.
        """
        return "IntentClassifier"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to determine its primary intent.

        The method expects `data` to be a string representing a user query or statement.
        It performs a keyword-based matching against its configured intent map.

        Args:
            data (Any): The input data to be classified, ideally a string.
            context (Dict[str, Any]): A dictionary providing contextual information
                                       for the current processing flow. This context
                                       can be modified by the node for downstream use.

        Returns:
            Dict[str, Any]: A dictionary containing the classification result,
                            typically including 'intent' and 'confidence'.
                            Example: `{'intent': 'greeting', 'confidence': 0.95}`
                            Returns `{'intent': 'error', 'message': '...'}` if input is invalid.

        Raises:
            TypeError: If the input `data` is not a string, indicating an incompatible
                       data type for this node's operation.
        """
        if not isinstance(data, str):
            error_msg = (f"IntentClassifierNode expects string input for classification, "
                         f"but received type: {type(data).__name__}. Data: {data!r}")
            logger.error(error_msg)
            # Raising an error is often preferred for type mismatches in a framework
            # to ensure strict data contracts between nodes.
            raise TypeError(error_msg)

        text_input = data.lower()
        classified_intent = self._default_intent
        confidence = 0.5  # Default low confidence for fallback or weak match

        if not text_input.strip():
            logger.warning("Received empty string for intent classification. Returning default intent.")
            return {"intent": self._default_intent, "confidence": confidence}

        for intent, keywords in self._intent_mapping.items():
            for keyword in keywords:
                if keyword in text_input:
                    classified_intent = intent
                    confidence = 0.95  # Simulate high confidence for a direct keyword match
                    logger.debug(f"Matched keyword '{keyword}' for intent '{intent}' in input: '{data[:50]}...'")
                    break  # Found a match, no need to check other keywords for this intent
            if classified_intent != self._default_intent:
                break  # Found a strong intent, stop checking other intents

        # Optionally, add the raw or processed input to context for traceability
        context['last_classified_text'] = data
        context['classified_intent_result'] = {'intent': classified_intent, 'confidence': confidence}

        logger.info(f"Classified intent for input '{data[:75]}...' as '{classified_intent}' with confidence {confidence:.2f}")
        return {
            "intent": classified_intent,
            "confidence": confidence
        }
