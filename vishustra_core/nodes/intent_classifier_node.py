import logging
from typing import Any, Dict, List

# Assuming BaseNode is available at this path as per instructions
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class IntentClassifierNode(BaseNode):
    """
    A Vishustra node designed to classify the intent of a given text input.

    This node simulates intent classification based on a simple rule-based approach
    using predefined keywords. In a production environment, this would typically
    integrate with advanced NLP models (e.g., fine-tuned transformer models,
    machine learning classifiers, or commercial NLP services) to provide robust
    and accurate intent recognition.

    The `process` method expects the input `data` to be a string representing
    a user query or utterance. It returns a dictionary containing the classified
    intent and a simulated confidence score.
    """

    def __init__(self, rules: Dict[str, List[str]] = None):
        """
        Initializes the IntentClassifierNode with a set of classification rules.

        Args:
            rules (Dict[str, List[str]], optional): A dictionary where keys represent
                intent names (str) and values are lists of keywords (List[str])
                associated with that intent. If `None`, a sensible default set
                of rules will be employed for demonstration.
        """
        self._node_name = "IntentClassifierNode"
        self._default_rules = {
            "book_flight": ["book flight", "fly to", "ticket to", "travel plan", "reservation"],
            "check_status": ["order status", "shipment tracking", "delivery date", "my order"],
            "customer_support": ["help me", "support", "contact us", "problem with", "agent"],
            "product_inquiry": ["what is", "about product", "features of", "specifications"],
            "greeting": ["hello", "hi", "hey", "good morning"],
            "goodbye": ["bye", "goodbye", "see you", "farewell"]
        }
        self.classification_rules = rules if rules is not None else self._default_rules
        logger.info(f"[{self.node_name}] Initialized with {len(self.classification_rules)} intent rules.")

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node.
        """
        return self._node_name

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to classify its underlying intent.

        The method iterates through predefined rules, attempting to match keywords
        within the input text to an intent. The first matching intent is returned
        with a simulated high confidence. If no specific intent is matched,
        it defaults to 'unknown' with a lower confidence.

        Args:
            data (Any): The input data, which must be a string containing the query
                        or utterance to be classified.
            context (Dict[str, Any]): A dictionary containing contextual information
                                      relevant to the current execution flow. While
                                      not directly used for classification logic
                                      in this node, it's available for broader
                                      orchestration or logging purposes.

        Returns:
            Dict[str, Any]: A dictionary containing two keys:
                            - 'intent' (str): The classified intent, e.g., "book_flight", "unknown".
                            - 'confidence' (float): A simulated confidence score between 0.0 and 1.0.

        Raises:
            TypeError: If the input `data` is not of type `str`.
            ValueError: If an unexpected error occurs during the classification process.
        """
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected 'str', "
                f"received '{type(data).__name__}'."
            )
            raise TypeError(
                f"[{self.node_name}] Input 'data' must be a string for intent classification. "
                f"Received type: {type(data).__name__}"
            )

        query = data.lower().strip()
        classified_intent = "unknown"
        confidence = 0.0

        logger.debug(f"[{self.node_name}] Attempting to classify intent for query: '{query}'")

        try:
            for intent, keywords in self.classification_rules.items():
                for keyword in keywords:
                    if keyword in query:
                        classified_intent = intent
                        confidence = 0.95  # Simulate high confidence for a direct keyword match
                        logger.info(
                            f"[{self.node_name}] Classified intent as '{classified_intent}' "
                            f"for query '{query}' (matched keyword: '{keyword}')."
                        )
                        return {"intent": classified_intent, "confidence": confidence}

            # If no specific intent is matched after checking all rules
            if classified_intent == "unknown":
                confidence = 0.5  # Simulate moderate confidence for unclassified queries
                logger.info(
                    f"[{self.node_name}] No specific intent found for query '{query}'. "
                    f"Defaulting to '{classified_intent}'."
                )

            return {"intent": classified_intent, "confidence": confidence}

        except Exception as e:
            logger.critical(
                f"[{self.node_name}] An unexpected error occurred during intent classification "
                f"for query '{query}': {e}",
                exc_info=True
            )
            # Re-raising as a ValueError to indicate a processing failure to upstream nodes
            raise ValueError(f"[{self.node_name}] Classification failed due to an internal error.") from e