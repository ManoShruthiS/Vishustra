import logging
from typing import Any, Dict, List

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class IntentClassifierNode(BaseNode):
    """
    A processing node that classifies the intent of a given text input.

    This node simulates intent classification based on predefined keywords
    or patterns. It expects the input 'data' to be a string (the user query)
    and uses the 'context' to optionally provide an intent mapping.

    Configuration via context:
    - 'intent_map' (Dict[str, List[str]]): A dictionary where keys are intent
      names (e.g., "greeting", "purchase") and values are lists of keywords
      or phrases associated with that intent. If not provided, a sensible
      default map will be used.
    - 'default_intent' (str): The intent to return if no explicit match is found.
      Defaults to "unknown".
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "IntentClassifierNode"

    def _get_default_intent_map(self) -> Dict[str, List[str]]:
        """Provides a default mapping for common intents."""
        return {
            "greeting": ["hello", "hi", "hey", "good morning", "good evening"],
            "purchase": ["buy", "order", "purchase", "add to cart", "checkout"],
            "cancellation": ["cancel", "revoke", "stop order", "undo purchase"],
            "support": ["help", "support", "customer service", "technical issue"],
            "farewell": ["bye", "goodbye", "see you", "farewell"],
        }

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classifies the intent of the input text data.

        Args:
            data (Any): The input data, expected to be a string representing
                        the user's query or utterance.
            context (Dict[str, Any]): A dictionary containing contextual information
                                     and potential configuration, e.g., 'intent_map'.

        Returns:
            Dict[str, Any]: A dictionary containing the classified intent and a
                            confidence score, e.g., {"intent": "greeting", "confidence": 0.95}.

        Raises:
            TypeError: If the input `data` is not a string.
            ValueError: If the `intent_map` in context is malformed.
        """
        if not isinstance(data, str):
            logger.error(f"Invalid input data type for {self.node_name}. Expected str, got {type(data)}.")
            raise TypeError(f"IntentClassifierNode expects string input, but received {type(data)}.")

        query = data.lower().strip()
        classified_intent = context.get("default_intent", "unknown")
        confidence = 0.5  # Default confidence for unknown intent

        try:
            intent_map = context.get("intent_map")
            if intent_map is None:
                intent_map = self._get_default_intent_map()
                logger.info(f"No 'intent_map' provided in context for {self.node_name}. Using default map.")
            elif not isinstance(intent_map, dict):
                raise ValueError(f"'intent_map' in context must be a dictionary, got {type(intent_map)}.")

            for intent, keywords in intent_map.items():
                if not isinstance(keywords, list) or not all(isinstance(k, str) for k in keywords):
                    raise ValueError(f"Keywords for intent '{intent}' in 'intent_map' must be a list of strings.")
                
                for keyword in keywords:
                    if keyword.lower() in query:
                        classified_intent = intent
                        confidence = 0.95  # High confidence for a direct keyword match
                        logger.debug(f"Query '{data}' matched intent '{intent}' with keyword '{keyword}'.")
                        return {"intent": classified_intent, "confidence": confidence}

            logger.info(f"No specific intent matched for query: '{data}'. Defaulting to '{classified_intent}'.")

        except ValueError as e:
            logger.error(f"Configuration error in IntentClassifierNode: {e}")
            raise e
        except Exception as e:
            logger.error(f"An unexpected error occurred during intent classification for query '{data}': {e}")
            # Re-raise or wrap in a custom node-specific exception if needed for higher-level handling
            raise RuntimeError(f"Failed to classify intent due to internal error: {e}") from e
            
        return {"intent": classified_intent, "confidence": confidence}

