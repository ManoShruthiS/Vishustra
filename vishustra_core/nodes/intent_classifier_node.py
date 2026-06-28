import logging
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class IntentClassifierNode(BaseNode):
    """
    A Vishustra node designed to classify the intent of a given text input.

    This node provides a simulated intent classification mechanism. In a
    production environment, it would typically interface with a specialized
    machine learning model or an external NLP service to perform
    sophisticated intent recognition. For demonstration, it uses a simple
    keyword-based matching logic that can be configured via the context.
    """

    def __init__(self):
        """
        Initializes the IntentClassifierNode.

        Any setup for loading models, configuring external APIs, or
        initializing resources for intent classification would occur here.
        """
        logger.debug("IntentClassifierNode initialized.")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this node."""
        return "IntentClassifier"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to identify its underlying intent.

        The `data` input is expected to be a string containing the text
        to be classified. The `context` dictionary can be used to provide
        configuration parameters, such as an `intent_mapping` dictionary
        to define custom intents and their associated keywords, or a
        `default_intent` string for unclassified inputs.

        Args:
            data: The input text as a string that needs intent classification.
            context: A dictionary containing operational parameters for the node.
                     Expected keys include:
                     - 'intent_mapping' (Dict[str, List[str]]): A mapping from
                       intent names to a list of keywords associated with that intent.
                     - 'default_intent' (str): The intent to return if no other
                       intent is matched. Defaults to 'unknown'.

        Returns:
            A dictionary containing:
            - 'text' (str): The original input text.
            - 'intent' (str): The classified intent, e.g., 'greeting', 'place_order', 'unknown'.
            - 'confidence' (float): A simulated confidence score for the classification,
                                    ranging from 0.0 to 1.0.

        Raises:
            TypeError: If the input `data` is not a string.
            ValueError: If the 'intent_mapping' in `context` is not a dictionary.
        """
        if not isinstance(data, str):
            logger.error(
                f"Invalid input data type for IntentClassifierNode. Expected 'str', "
                f"but received '{type(data).__name__}'."
            )
            raise TypeError(
                f"IntentClassifierNode requires string input, but got {type(data).__name__}."
            )

        text_input = data.strip().lower()
        classified_intent = context.get('default_intent', 'unknown')
        confidence = 0.5  # Default confidence for unclassified or low-match scenarios

        intent_mapping = context.get('intent_mapping')

        if intent_mapping:
            if not isinstance(intent_mapping, dict):
                logger.error(
                    f"Invalid 'intent_mapping' type in context. Expected 'dict', "
                    f"but received '{type(intent_mapping).__name__}'."
                )
                raise ValueError("Context 'intent_mapping' must be a dictionary.")

            matched = False
            for intent, keywords in intent_mapping.items():
                if not isinstance(keywords, (list, tuple)):
                    logger.warning(
                        f"Keywords for intent '{intent}' in context 'intent_mapping' "
                        f"are not a list or tuple. Skipping this intent."
                    )
                    continue

                for keyword in keywords:
                    if isinstance(keyword, str) and keyword.lower() in text_input:
                        classified_intent = intent
                        confidence = 0.95  # High confidence for direct keyword match
                        matched = True
                        break
                if matched:
                    break
            
            if not matched:
                logger.debug(f"No specific intent matched via 'intent_mapping' for text: '{data[:50]}...'")
                # If mapping was provided but no match, use default_intent with slightly higher confidence than raw unknown
                confidence = 0.6 if classified_intent != 'unknown' else 0.5 # If default was explicitly set

        else:
            logger.debug("No 'intent_mapping' found in context. Using internal fallback logic for classification.")
            # Fallback to simple internal logic if no context mapping is provided
            if any(phrase in text_input for phrase in ["hello", "hi", "hey"]):
                classified_intent = "greeting"
                confidence = 0.9
            elif any(phrase in text_input for phrase in ["bye", "goodbye", "see ya"]):
                classified_intent = "farewell"
                confidence = 0.9
            elif any(phrase in text_input for phrase in ["help", "support", "assist"]):
                classified_intent = "request_help"
                confidence = 0.85
            elif any(phrase in text_input for phrase in ["order", "purchase", "buy"]):
                classified_intent = "place_order"
                confidence = 0.8
            else:
                classified_intent = context.get('default_intent', 'unknown')
                confidence = 0.5

        logger.info(
            f"Classified intent for text '{data[:70]}{'...' if len(data) > 70 else ''}' "
            f"as '{classified_intent}' with confidence {confidence:.2f}."
        )

        return {
            "text": data,
            "intent": classified_intent,
            "confidence": confidence
        }