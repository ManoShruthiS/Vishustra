import logging
from typing import Any, Dict, List, Optional
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class IntentClassifierNode(BaseNode):
    """
    Analyzes input text to determine the underlying user intent.
    This node serves as a router within the pipeline,
    allowing downstream logic to branch based on classification.
    """

    def __init__(self, 
                 categories: Optional[Dict[str, List[str]]] = None, 
                 default_intent: str = "general_query"):
        """
        Initializes the classifier with optional custom intent mappings.
        
        Args:
            categories: A dictionary mapping intent labels to lists of keywords/phrases.
            default_intent: The fallback intent if no specific category is matched.
        """
        self._default_intent = default_intent
        self._categories = categories or {
            "informational": ["what", "how", "explain", "tell me", "meaning"],
            "transactional": ["buy", "order", "purchase", "subscribe", "pay"],
            "navigation": ["go to", "find", "show", "where is", "open"],
            "support": ["help", "error", "issue", "problem", "fix", "wrong"]
        }

    @property
    def node_name(self) -> str:
        """Returns the unique identifier for this node type."""
        return "IntentClassifierNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to classify its intent.
        
        Args:
            data: The input string to be classified.
            context: The shared orchestration context.
            
        Returns:
            A dictionary containing the original input and the detected intent.
            
        Raises:
            ValueError: If the input data is not a string.
        """
        if not isinstance(data, str):
            logger.error(f"[{self.node_name}] Received invalid data type: {type(data)}. Expected string.")
            raise ValueError(f"{self.node_name} requires a string input for classification.")

        normalized_input = data.lower().strip()
        detected_intent = self._default_intent

        try:
            # Basic keyword-based heuristic classification
            for intent, keywords in self._categories.items():
                if any(keyword in normalized_input for keyword in keywords):
                    detected_intent = intent
                    break

            logger.info(f"[{self.node_name}] Successfully classified intent as: '{detected_intent}'")

            result = {
                "original_input": data,
                "classified_intent": detected_intent,
                "confidence_score": 1.0 if detected_intent != self._default_intent else 0.5
            }

            # Persist classification result to context for downstream logic
            context["classification_metadata"] = result
            
            return result

        except Exception as e:
            logger.error(f"[{self.node_name}] Failed to process intent classification: {str(e)}")
            raise RuntimeError(f"Internal error in {self.node_name}: {e}") from e

    def add_category(self, intent: str, keywords: List[str]) -> None:
        """Dynamically adds or updates an intent category."""
        self._categories[intent] = [k.lower() for k in keywords]
        logger.debug(f"[{self.node_name}] Updated category '{intent}' with {len(keywords)} keywords.")

    def __repr__(self) -> str:
        return f"<{self.node_name}(categories={list(self._categories.keys())})>"

