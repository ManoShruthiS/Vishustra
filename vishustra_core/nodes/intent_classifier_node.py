import logging
from typing import Any, Dict, List, Optional

# Assuming this path exists in the Vishustra project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class IntentClassifierNode(BaseNode):
    """
    A Vishustra processing node that simulates intent classification from text input.

    This node takes a string (typically a user query) and attempts to classify
    its underlying intent based on predefined keyword rules. In a production
    environment, this would likely integrate with a more sophisticated machine
    learning model or a comprehensive rule-based engine configured via external sources.
    The simulation here provides a basic example of such functionality.
    """

    _DEFAULT_INTENT_RULES = {
        "book_reservation": ["book", "reserve", "reservation"],
        "cancel_reservation": ["cancel", "change", "alter"],
        "get_weather": ["weather", "forecast", "temperature"],
        "greeting": ["hello", "hi", "hey", "good morning", "good evening"],
        "tell_joke": ["joke", "funny", "laugh"],
        "get_time": ["time", "current time", "what time is it"],
        "get_date": ["date", "today's date", "what day is it"],
    }
    _DEFAULT_CONFIDENCE = 0.6
    _MATCH_CONFIDENCE = 0.95

    def __init__(self, intent_rules: Optional[Dict[str, List[str]]] = None):
        """
        Initializes the IntentClassifierNode with a set of intent classification rules.

        Args:
            intent_rules (Optional[Dict[str, List[str]]]): A dictionary mapping
                                                            intent names to lists of keywords.
                                                            If None, default rules are used.
        """
        self._intent_rules = intent_rules if intent_rules is not None else self._DEFAULT_INTENT_RULES
        logger.info(f"IntentClassifierNode initialized with {len(self._intent_rules)} intent rules.")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "IntentClassifier"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to classify its intent.

        Args:
            data (Any): The input data, expected to be a string representing a user query.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the processing flow. Can be used for dynamic
                                       rule overrides or additional parameters.

        Returns:
            Dict[str, Any]: A dictionary containing the classified intent, a confidence score,
                            and the original query.
                            Example: {"intent": "book_reservation", "confidence": 0.95, "original_query": "I want to book a table."}
                            If no specific intent is matched, it returns a "general_query" intent.

        Raises:
            ValueError: If the input data is not a string.
        """
        if not isinstance(data, str):
            error_msg = f"{self.node_name} received invalid data type. Expected 'str', got '{type(data).__name__}'."
            logger.error(error_msg)
            raise ValueError(error_msg)

        original_query = data
        processed_query = data.lower().strip()
        
        # Allow dynamic override or addition of rules from context for flexibility
        current_intent_rules = self._intent_rules.copy()
        if 'dynamic_intent_rules' in context and isinstance(context['dynamic_intent_rules'], dict):
            logger.debug(f"Applying dynamic intent rules from context for {self.node_name}.")
            current_intent_rules.update(context['dynamic_intent_rules'])
        
        matched_intent: Optional[str] = None
        for intent, keywords in current_intent_rules.items():
            # Check if any keyword associated with the intent is present in the query
            if any(keyword in processed_query for keyword in keywords):
                matched_intent = intent
                break # Take the first matching intent for simplicity and speed

        result: Dict[str, Any]
        if matched_intent:
            result = {
                "intent": matched_intent,
                "confidence": self._MATCH_CONFIDENCE,
                "original_query": original_query,
            }
            logger.info(
                f"Query '{original_query[:50]}{'...' if len(original_query) > 50 else ''}' "
                f"classified as intent '{matched_intent}' with confidence {self._MATCH_CONFIDENCE:.2f}."
            )
        else:
            result = {
                "intent": "general_query",
                "confidence": self._DEFAULT_CONFIDENCE,
                "original_query": original_query,
            }
            logger.info(
                f"No specific intent matched for query '{original_query[:50]}{'...' if len(original_query) > 50 else ''}'. "
                f"Classified as 'general_query' with confidence {self._DEFAULT_CONFIDENCE:.2f}."
            )
        
        return result
