import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class IntentClassifierNode(BaseNode):
    """
    A processing node that classifies the intent of a given text input.

    This node simulates intent classification based on simple keyword matching.
    In a production environment, this would typically integrate with
    a machine learning model (e.g., an NLU service, a fine-tuned transformer).
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "IntentClassifier"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input text to determine its underlying intent.

        The `data` input is expected to be a string representing the user's query.
        The `context` dictionary can optionally contain configuration for the
        classifier, though for this simulated implementation, it's not strictly
        necessary for the core logic.

        Args:
            data (Any): The input data, expected to be a string (user query).
            context (Dict[str, Any]): A dictionary containing context-specific
                                       information, potentially including model
                                       configuration or additional parameters.

        Returns:
            Any: A dictionary containing the original text and the classified intent.
                 Example: {"text": "What's the weather like?", "intent": "get_weather"}

        Raises:
            TypeError: If the input `data` is not a string.
            ValueError: If the classification logic encounters an unexpected issue.
        """
        if not isinstance(data, str):
            logger.error(f"IntentClassifierNode received invalid data type: {type(data)}. Expected str.")
            raise TypeError(f"IntentClassifierNode expects string input, but received {type(data)}.")

        query = data.lower().strip()
        classified_intent = "general_query" # Default intent

        try:
            # Simulate intent classification with simple keyword matching
            if any(keyword in query for keyword in ["book", "reserve", "flight", "ticket", "travel"]):
                classified_intent = "book_flight"
            elif any(keyword in query for keyword in ["weather", "forecast", "temperature", "climate"]):
                classified_intent = "get_weather"
            elif any(keyword in query for keyword in ["news", "headlines", "article"]):
                classified_intent = "get_news"
            elif any(keyword in query for keyword in ["order", "purchase", "buy", "item", "product"]):
                classified_intent = "place_order"
            elif any(keyword in query for keyword in ["help", "support", "assist"]):
                classified_intent = "request_help"
            
            # Additional context-based hints could be used here
            # For example, if context['previous_intent'] was 'book_flight',
            # a query like 'for tomorrow' could be classified as 'confirm_details_flight'

            logger.info(f"Classified intent for '{query[:50]}...' as '{classified_intent}'")
            return {"text": data, "intent": classified_intent}

        except Exception as e:
            logger.exception(f"An error occurred during intent classification for query '{query[:50]}...': {e}")
            raise ValueError(f"Failed to classify intent for query: '{query[:50]}...' due to an internal error.") from e

