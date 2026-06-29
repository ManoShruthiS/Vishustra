import logging
from typing import Any, Dict, List

# Assume BaseNode is available at this path as per project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class IntentClassifierNode(BaseNode):
    """
    A Vishustra node designed to classify the intent of a given text input.

    This node simulates intent classification using a configurable keyword-based
    approach. In a production Vishustra environment, this class would typically
    integrate with actual machine learning models (e.g., Transformer-based
    models, simpler text classifiers) or advanced rule engines, potentially
    loading them via the `context` or during initialization.

    The node expects a string as input data and returns a string representing
    the identified intent or "UNKNOWN" if no specific intent is detected.
    """

    def __init__(self, intent_map: Dict[str, List[str]] = None):
        """
        Initializes the IntentClassifierNode with a mapping of intents to keywords.

        Args:
            intent_map: An optional dictionary where keys are intent names (str)
                        and values are lists of keywords/phrases (List[str])
                        that trigger that intent. If None, a default intent map
                        is used. This allows for flexible configuration.
        """
        # Default intent map for demonstration. In a real-world scenario,
        # this would likely be loaded from a configuration file, a database,
        # or dynamically generated.
        self._intent_map: Dict[str, List[str]] = intent_map if intent_map is not None else {
            "GREETING": ["hello", "hi", "hey", "good morning", "good evening"],
            "ORDER_STATUS": ["where is my order", "order status", "track my package", "order #", "my delivery"],
            "PRODUCT_INQUIRY": ["tell me about", "product info", "what is", "features of"],
            "BOOK_FLIGHT": ["book a flight", "flight to", "fly me to", "travel to"],
            "CANCEL_ORDER": ["cancel my order", "cancel order #", "undo order"],
            "HELP": ["help", "support", "assistance", "can you help me"],
            "GOODBYE": ["bye", "goodbye", "see you", "farewell"]
        }
        logger.debug(f"[{self.node_name}] Initialized with intent categories: {list(self._intent_map.keys())}")

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this processing node.
        """
        return "IntentClassifier"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Processes the input data to determine and return its classified intent.

        This method converts the input data to lowercase and checks it against
        predefined keywords in the internal intent map. The first matching
        intent is returned. If no keywords match, "UNKNOWN" is returned.

        Args:
            data: The input data to be classified, expected to be a string
                  (e.g., a user query, a message from an upstream node).
            context: A dictionary containing runtime contextual information.
                     In advanced implementations, this could carry references
                     to pre-loaded ML models, configuration flags, or session
                     data relevant to the classification process.

        Returns:
            A string representing the classified intent (e.g., "GREETING",
            "ORDER_STATUS", "UNKNOWN").

        Raises:
            TypeError: If the input 'data' is not of type string, ensuring
                       data integrity for downstream processing.
        """
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected 'str', received '{type(data).__name__}'.")
            raise TypeError(
                f"[{self.node_name}] Input data must be a string for intent classification. Got: {type(data).__name__}"
            )

        query = data.lower().strip()
        classified_intent = "UNKNOWN"

        # Advanced implementation note: In a production setting, 'context'
        # might be used to pass a pre-trained model instance (e.g.,
        # `model = context.get("intent_model")`), allowing the node to
        # use a more sophisticated classification mechanism.
        # For this simulation, we rely on the internal keyword map.

        for intent, keywords in self._intent_map.items():
            for keyword in keywords:
                if keyword in query:
                    classified_intent = intent
                    break  # Found a match for this intent
            if classified_intent != "UNKNOWN":
                break  # Found an intent, no need to check other intents

        if classified_intent == "UNKNOWN":
            logger.warning(f"[{self.node_name}] Could not classify intent for query: '{data}'")
        else:
            logger.info(f"[{self.node_name}] Classified intent for query '{data}' as '{classified_intent}'")

        return classified_intent