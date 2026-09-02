import logging
from typing import Any, Dict, List, Optional

# Assuming BaseNode is available at the specified path as per project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class IntentClassifierNode(BaseNode):
    """
    A Vishustra node that classifies the intent of a given text input.

    This node simulates intent classification by looking for keywords
    in the input text and mapping them to predefined intents.
    It supports configurable intent mappings and robustly handles
    various input types and edge cases.
    """

    def __init__(self, intent_mapping: Optional[Dict[str, List[str]]] = None):
        """
        Initializes the IntentClassifierNode with an optional custom intent mapping.

        Args:
            intent_mapping (Optional[Dict[str, List[str]]]): A dictionary where keys
                are intent names (str) and values are lists of keywords (List[str])
                associated with that intent. If None, a default mapping is used.
        """
        self._default_intent_mapping = {
            "book_travel": ["book", "flight", "hotel", "travel", "reservation"],
            "get_weather": ["weather", "forecast", "temperature", "climate"],
            "play_music": ["play", "music", "song", "playlist", "track"],
            "set_reminder": ["remind", "reminder", "alert"],
            "get_news": ["news", "headlines", "articles"],
            "check_balance": ["balance", "account", "money"],
        }
        self._intent_mapping = intent_mapping if intent_mapping is not None else self._default_intent_mapping
        logger.debug(f"IntentClassifierNode initialized with mapping: {self._intent_mapping}")


    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "IntentClassifier"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input text to classify its intent based on keywords.

        Args:
            data (Any): The input data, expected to be a string representing a user query.
            context (Dict[str, Any]): A dictionary containing contextual information.
                                       This node currently does not utilize `context` for classification
                                       but it is available for future enhancements (e.g., user-specific
                                       intent preferences or session context).

        Returns:
            Dict[str, Any]: A dictionary containing the classified intent, the original text,
                            and any keywords that led to the classification.
                            Example: {"intent": "book_travel", "text": "book me a flight to London", "matched_keywords": ["book", "flight"]}
                            If no specific intent is found, it defaults to "general_query".
                            If the input is empty, it returns "empty_query".

        Raises:
            TypeError: If the input data is not a string, ensuring type consistency in the pipeline.
        """
        if not isinstance(data, str):
            logger.error(f"IntentClassifierNode received invalid input type. Expected 'str', got '{type(data).__name__}'.")
            raise TypeError(f"IntentClassifierNode expects string input, but received {type(data).__name__}.")

        query = data.strip().lower()

        if not query:
            logger.warning("IntentClassifierNode received an empty query. Classifying as 'empty_query'.")
            return {"intent": "empty_query", "text": data, "matched_keywords": []}

        classified_intent = "general_query"
        found_keywords: List[str] = []

        # Iterate through defined intents and their keywords to find a match
        for intent, keywords_for_intent in self._intent_mapping.items():
            for keyword in keywords_for_intent:
                if keyword in query:
                    classified_intent = intent
                    found_keywords.append(keyword)
                    # For simplicity, we assign the first specific intent found.
                    # More advanced logic could rank intents based on multiple keyword matches,
                    # keyword weight, or context from the 'context' dictionary.
                    logger.debug(f"Matched keyword '{keyword}' for intent '{intent}' in query: '{query}'")
                    break  # Stop checking keywords for this intent, move to the next intent if multiple matches desired or break if first match is sufficient
            if classified_intent != "general_query":
                # If a specific intent was found, we can stop searching further.
                # Remove this break if multiple intent classification or a confidence score is needed.
                break

        if classified_intent == "general_query":
            logger.info(f"No specific intent identified for query: '{query}'. Defaulting to 'general_query'.")
        else:
            logger.info(f"Classified intent as '{classified_intent}' for query: '{query}' (matched keywords: {', '.join(found_keywords)}).")

        return {"intent": classified_intent, "text": data, "matched_keywords": found_keywords}