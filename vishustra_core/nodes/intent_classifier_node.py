import logging
from typing import Any, Dict, List, Optional

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class IntentClassifierNode(BaseNode):
    """
    A processing node designed to classify the intent of an input text.
    
    This node simulates intent classification based on predefined keyword rules.
    In a production environment, this would integrate with a proper NLP model
    or service for intent recognition.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the IntentClassifierNode with classification rules and a default intent.

        Args:
            config: A dictionary containing configuration for the classifier.
                    Expected keys:
                    - 'classification_rules': A dict where keys are intent names (str) and
                                              values are lists of keywords (List[str]).
                                              Example: {'greeting': ['hi', 'hello'], 'farewell': ['bye', 'goodbye']}
                    - 'default_intent': The intent (str) to return if no rule matches.
                                        Defaults to 'unknown_intent'.
        """
        self._config = config if config is not None else {}
        self._classification_rules: Dict[str, List[str]] = self._config.get('classification_rules', {})
        self._default_intent: str = self._config.get('default_intent', 'unknown_intent')
        logger.debug(
            f"IntentClassifierNode initialized with rules: {self._classification_rules} "
            f"and default intent: '{self._default_intent}'"
        )

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "IntentClassifier"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Classifies the intent of the input text based on predefined keyword rules.

        Args:
            data: The input text (str) to classify.
            context: A dictionary containing additional context data.
                     This node currently does not utilize the context.

        Returns:
            A string representing the classified intent.

        Raises:
            ValueError: If the input 'data' is not a string.
        """
        if not isinstance(data, str):
            logger.error(
                f"IntentClassifier received non-string data of type '{type(data).__name__}'. "
                "Expected a string for classification."
            )
            raise ValueError("IntentClassifier expects string input for classification.")

        text_lower = data.lower()
        classified_intent = self._default_intent

        logger.debug(f"Attempting to classify intent for text: '{data}'")

        # Simple keyword-based classification
        for intent, keywords in self._classification_rules.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    classified_intent = intent
                    logger.info(
                        f"Classified intent as '{classified_intent}' for text: '{data}' "
                        f"based on keyword '{keyword}'"
                    )
                    return classified_intent  # Return the first match

        logger.info(
            f"No specific intent classified for text: '{data}' based on rules. "
            f"Defaulting to '{classified_intent}'."
        )
        return classified_intent