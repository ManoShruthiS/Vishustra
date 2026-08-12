import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class IntentClassifierNode(BaseNode):
    """
    A Vishustra processing node designed to classify the intent of a given text input.

    This node simulates intent classification based on a set of predefined keyword rules.
    In a production environment, this would typically integrate with a dedicated
    Natural Language Understanding (NLU) service or a sophisticated machine learning model
    to perform more robust and dynamic intent recognition.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initializes the IntentClassifierNode with optional configuration.

        Args:
            config: An optional dictionary containing configuration parameters.
                    Expected keys:
                    - 'intent_rules': A dictionary where keys are intent names (str)
                                      and values are lists of keywords (list[str])
                                      associated with that intent. These rules
                                      will merge with or override default rules.
        """
        self._node_name = "IntentClassifierNode"
        
        # Default intent rules for demonstration
        self._intent_rules = {
            "greeting": ["hello", "hi", "hey", "good morning", "good evening"],
            "order_status": ["track my order", "where is my package", "order status", "delivery status", "my order"],
            "product_info": ["tell me about", "what is", "product details", "specs for", "information on"],
            "goodbye": ["bye", "see you", "goodbye", "farewell"],
            "support_request": ["help", "support", "contact us", "problem with"],
            "thank_you": ["thank you", "thanks", "appreciate it"],
            # 'unclear' is handled as a fallback if no other rules match
        }

        if config and isinstance(config, dict) and 'intent_rules' in config:
            if isinstance(config['intent_rules'], dict):
                # Merge custom rules provided in config with default rules
                self._intent_rules.update(config['intent_rules'])
                logger.info(f"[{self.node_name}] Initialized with custom intent rules from configuration.")
            else:
                logger.warning(
                    f"[{self.node_name}] 'intent_rules' in config must be a dictionary. "
                    "Falling back to default rules."
                )
        else:
            logger.info(f"[{self.node_name}] Initialized with default intent rules.")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return self._node_name

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classifies the intent of the input text using keyword matching.

        The input `data` is expected to be either a string directly representing
        the user query, or a dictionary containing a 'text' key whose value
        is the string to be classified.

        Args:
            data: The input data, which can be a string (e.g., "Hello, how are you?")
                  or a dictionary like `{'text': 'Where is my order?'}`.
            context: A dictionary providing shared context or state across nodes
                     within the Vishustra pipeline. Not directly used by this
                     node for intent classification, but available.

        Returns:
            A dictionary containing the original text, the classified intent,
            and a confidence score for that classification.
            Example: `{'text': 'Hello there', 'intent': 'greeting', 'confidence': 0.9}`

        Raises:
            ValueError: If the input data is not a string or a dictionary with
                        a valid 'text' key.
        """
        text_to_classify = None
        if isinstance(data, str):
            text_to_classify = data
        elif isinstance(data, dict) and 'text' in data and isinstance(data['text'], str):
            text_to_classify = data['text']
        
        if text_to_classify is None:
            logger.warning(
                f"[{self.node_name}] Invalid input data type. Expected string or dict with 'text' key. Got: {type(data)}."
            )
            raise ValueError(
                f"[{self.node_name}] Input data must be a string or a dictionary with a 'text' key. "
                f"Received type: {type(data)}."
            )

        processed_text = text_to_classify.lower()
        
        classified_intent = "unclear"
        confidence = 0.5  # Default confidence for an 'unclear' classification

        # Perform simple keyword-based intent classification
        for intent, keywords in self._intent_rules.items():
            if not keywords: # Skip intents with no keywords defined if we ever get them
                continue
            for keyword in keywords:
                if keyword.lower() in processed_text:
                    classified_intent = intent
                    confidence = 0.9  # Assign higher confidence for a direct keyword match
                    logger.debug(f"[{self.node_name}] Matched keyword '{keyword}' for intent '{intent}'.")
                    break  # Found a match, no need to check other keywords for this intent
            if classified_intent != "unclear":
                break  # If a specific intent was found, stop checking other intents

        logger.info(
            f"[{self.node_name}] Classified intent for text '{text_to_classify[:70]}{'...' if len(text_to_classify) > 70 else ''}' "
            f"as '{classified_intent}' with confidence {confidence:.2f}."
        )
        
        return {
            "text": text_to_classify,
            "intent": classified_intent,
            "confidence": confidence
        }
