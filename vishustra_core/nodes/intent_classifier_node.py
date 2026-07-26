import logging
from typing import Any, Dict, Optional, List

# Assume vishustra_core.nodes.base_node exists as per project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class IntentClassifierNode(BaseNode):
    """
    A Vishustra node that classifies the intent of a given text input.

    This node simulates intent classification by looking for predefined keywords
    or patterns within the input text. It can be configured with custom intent
    mappings via the context to allow dynamic adaptation.
    """

    @property
    def node_name(self) -> str:
        """Returns the programmatic name of the node."""
        return "IntentClassifier"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to determine its intent.

        Args:
            data (Any): The input data, expected to be a string representing a user query
                        or a piece of text to be classified.
            context (Dict[str, Any]): A dictionary containing contextual information
                                     for the node's operation. This can include:
                                     - 'intent_map' (Dict[str, str]): An optional dictionary
                                       mapping keywords (str) to intent names (str). If
                                       provided, this map will override or augment the
                                       node's default classification rules. Keywords are
                                       case-insensitive during matching.

        Returns:
            Dict[str, Any]: A structured dictionary containing the classification results:
                            - 'intent' (str): The classified intent, e.g., 'book_flight', 'get_weather'.
                                              Defaults to 'unknown' if no specific intent is identified.
                            - 'confidence' (float): A simulated confidence score (0.0 to 1.0)
                                                    for the identified intent.
                            - 'original_query' (str): The original input text that was classified.

        Raises:
            ValueError: If the input data is not a string, which is required for text classification.
        """
        if not isinstance(data, str):
            logger.error(
                "IntentClassifierNode received non-string data. Expected a string for classification. "
                f"Received type: {type(data).__name__}"
            )
            raise ValueError(
                f"Invalid data type for IntentClassifierNode. Expected 'str', got '{type(data).__name__}'."
            )

        query: str = data.lower()
        classified_intent: str = "unknown"
        confidence: float = 0.5  # Default low confidence if no strong match

        # Define a robust default intent mapping based on common use cases.
        # Keywords are checked for presence in the lowercased query.
        default_intent_map: Dict[str, str] = {
            "book": "book_flight",
            "reservation": "check_reservation",
            "weather": "get_weather",
            "forecast": "get_weather",
            "news": "get_news",
            "headline": "get_news",
            "order": "place_order",
            "status": "check_order_status",
            "help": "get_help",
            "support": "get_help",
            "cancel": "cancel_service",
            "schedule": "schedule_event",
            "price": "get_price",
        }

        # Allow an 'intent_map' to be provided via context for dynamic configuration.
        # This enables external systems or earlier nodes to define custom classification rules.
        intent_map: Dict[str, str] = context.get("intent_map", default_intent_map)

        logger.debug(f"Processing query: '{data}' with intent map containing {len(intent_map)} entries.")

        # Simulate intent classification using a simple keyword-matching approach.
        # In a production system, this would typically involve a machine learning model.
        # The first matching keyword determines the intent and confidence.
        for keyword_phrase, intent_name in intent_map.items():
            if keyword_phrase.lower() in query:
                classified_intent = intent_name
                # Assign a higher confidence for direct matches, especially from default maps.
                # This simple heuristic allows some distinction in a simulated environment.
                if keyword_phrase in default_intent_map:
                    confidence = 0.95
                else:
                    confidence = 0.85 # Slightly lower for context-provided, as they might be more volatile.
                logger.info(
                    f"Intent '{classified_intent}' identified for query '{data}' "
                    f"via keyword/phrase '{keyword_phrase}'."
                )
                break  # Stop on the first strong match

        if classified_intent == "unknown":
            logger.info(f"No specific intent found for query '{data}'. Classifying as 'unknown'.")
            # For unknown intents, provide a lower, default confidence.
            confidence = 0.3

        result: Dict[str, Any] = {
            "intent": classified_intent,
            "confidence": confidence,
            "original_query": data,
        }

        logger.debug(f"IntentClassifierNode processed query. Result: {result}")
        return result