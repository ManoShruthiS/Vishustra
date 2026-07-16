import logging
from typing import Any, Dict, Optional, Union

# Assuming BaseNode is available at this path as per project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class IntentClassifierNode(BaseNode):
    """
    A Vishustra processing node designed to classify the intent of a user query.

    This node simulates intent classification based on predefined keywords and patterns.
    In a production environment, this would typically involve integration with
    a machine learning model, an NLU (Natural Language Understanding) service,
    or a more sophisticated rule-based engine.

    The node expects the input data to be a user query string or a dictionary
    containing the query under a specific key.
    """

    _DEFAULT_INTENT_MAP: Dict[str, Any] = {
        "order_status": ["track my order", "where is my order", "delivery status", "order update", "my packages"],
        "product_info": ["tell me about", "what is", "product details", "specifications of", "features of"],
        "account_management": ["change password", "update profile", "my account settings", "manage subscription"],
        "technical_support": ["error with", "not working", "troubleshoot", "technical issue", "bug report"],
        "greeting": ["hello", "hi there", "hey", "good morning", "good evening"],
        "goodbye": ["bye now", "see you later", "farewell", "goodbye for now"],
        "thank_you": ["thanks a lot", "thank you very much", "appreciate it"],
        "returns": ["return an item", "how to return", "return policy"],
        "shipping_cost": ["shipping cost", "delivery fee", "how much is shipping"]
    }

    def __init__(self, intent_map: Optional[Dict[str, Any]] = None):
        """
        Initializes the IntentClassifierNode.

        Allows for an optional custom intent map to override or extend
        the default classification rules. Keywords are processed to be
        case-insensitive.

        Args:
            intent_map (Optional[Dict[str, Any]]): A dictionary mapping intent names
                                                   (str) to a list of associated keywords (str).
                                                   If None, the node uses a robust default map.
        """
        self._intent_map = intent_map if intent_map is not None else self._DEFAULT_INTENT_MAP
        # Process keywords for case-insensitive matching
        self._processed_intent_map = {
            intent: [kw.lower() for kw in kws]
            for intent, kws in self._intent_map.items()
        }
        logger.debug(
            f"IntentClassifierNode initialized. "
            f"Using custom intent map: {intent_map is not None}. "
            f"Known intents: {list(self._processed_intent_map.keys())}"
        )

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this processing node."""
        return "IntentClassifier"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to classify the user's intent.

        The method extracts a query string from the input `data` and attempts
        to match it against predefined intent patterns. It then returns a
        structured result containing the identified intent and a simulated
        confidence score.

        Args:
            data (Any): The input data. Expected to be either:
                        - A string representing the user query directly.
                        - A dictionary containing a 'query' key whose value is the query string.
            context (Dict[str, Any]): A dictionary providing contextual information
                                      for the current processing flow. This can include
                                      session data, user preferences, or previous node outputs.

        Returns:
            Dict[str, Any]: A dictionary containing:
                            - "query" (str): The original user query.
                            - "intent" (str): The classified intent (e.g., "order_status",
                                              "product_info", "unclear_intent").
                            - "confidence" (float): A simulated confidence score (0.0 to 1.0).

        Raises:
            ValueError: If the input `data` is not in the expected string or dictionary format,
                        or if the dictionary does not contain a valid 'query' key.
        """
        query: Optional[str] = None

        if isinstance(data, str):
            query = data
        elif isinstance(data, dict) and 'query' in data and isinstance(data['query'], str):
            query = data['query']
        else:
            error_msg = (
                f"IntentClassifierNode received invalid data format. "
                f"Expected a string or a dictionary with a 'query' key. Got type: {type(data)}."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        processed_query = query.lower().strip()
        classified_intent = "unclear_intent"  # Default if no specific intent is found
        confidence = 0.2  # Base confidence for unclear intent

        logger.info(f"Attempting to classify intent for query: '{query}'")
        logger.debug(f"Processing context keys: {list(context.keys()) if context else 'None'}")

        found_matches: Dict[str, int] = {} # Counts keyword matches per intent

        for intent, keywords in self._processed_intent_map.items():
            for keyword in keywords:
                if keyword in processed_query:
                    found_matches[intent] = found_matches.get(intent, 0) + 1

        if found_matches:
            # Simple logic: pick the intent with the most keyword matches
            best_intent = max(found_matches, key=found_matches.get) # type: ignore
            max_matches = found_matches[best_intent]

            # Refined confidence based on number of matches and intent prominence
            classified_intent = best_intent
            confidence = min(0.6 + (max_matches * 0.1), 0.99) # Cap confidence

            logger.info(f"Successfully identified intent: '{classified_intent}' for query: '{query}'")
            logger.debug(f"Matching details: {found_matches}")
        else:
            # Fallback to a broader default if no specific keywords match
            # This could be handled by a more general intent model or next node
            classified_intent = "general_query"
            confidence = 0.4
            logger.warning(
                f"No specific intent keywords matched for query: '{query}'. "
                f"Classifying as '{classified_intent}'."
            )

        result = {
            "query": query,
            "intent": classified_intent,
            "confidence": round(confidence, 2)
        }
        logger.debug(f"Intent classification result for '{query}': {result}")
        return result
