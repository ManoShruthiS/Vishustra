import logging
from typing import Any, Dict, List, Optional
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class IntentClassifierNode(BaseNode):
    """
    A node responsible for classifying the user's intent based on input text.
    This node can be configured with a set of keywords or leveraged alongside
    an LLM-based classification strategy via the context.
    """

    def __init__(self, default_intent: str = "unknown"):
        self.default_intent = default_intent

    @property
    def node_name(self) -> str:
        """
        Returns the unique identifier for this node type.
        """
        return "intent_classifier"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes the input data to determine the user's intent.
        
        Args:
            data: The input text to classify (expected to be a string).
            context: Execution context which may contain 'intent_map' or model configurations.
            
        Returns:
            A dictionary containing 'intent' and 'confidence'.
            
        Raises:
            ValueError: If the input data is not a string.
        """
        if not isinstance(data, str):
            logger.error(f"Invalid data type received: {type(data)}. Expected str.")
            raise ValueError("IntentClassifierNode requires string input for processing.")

        logger.info("Starting intent classification process.")

        try:
            # Normalize input
            text = data.strip().lower()
            
            # Retrieve intent map from context or use a basic internal fallback
            intent_map: Dict[str, List[str]] = context.get("intent_map", {
                "greeting": ["hello", "hi", "hey", "greetings"],
                "termination": ["exit", "quit", "bye", "stop"],
                "help": ["help", "info", "support", "docs"]
            })

            detected_intent = self.default_intent
            confidence = 0.0

            # Simple keyword matching logic for demonstration/fallback
            # In a production environment, this might call an external NLP service or LLM
            for intent, keywords in intent_map.items():
                if any(keyword in text for keyword in keywords):
                    detected_intent = intent
                    confidence = 0.95  # Heuristic confidence
                    break

            result = {
                "intent": detected_intent,
                "confidence": confidence,
                "original_input": data
            }

            logger.info(f"Classification complete. Detected intent: '{detected_intent}'")
            return result

        except Exception as e:
            logger.exception(f"An error occurred during intent classification: {str(e)}")
            return {
                "intent": self.default_intent,
                "confidence": 0.0,
                "error": str(e)
            }

    def __repr__(self) -> str:
        return f"<IntentClassifierNode(default_intent='{self.default_intent}')>"