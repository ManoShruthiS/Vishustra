import logging
from typing import Any, Dict, List, Optional
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class IntentClassifierNode(BaseNode):
    """
    A specialized node within the Vishustra framework designed to categorize 
    incoming natural language data into predefined semantic intents.
    
    This node typically serves as a router in complex LLM chains, 
    allowing subsequent nodes to execute logic based on detected user intent.
    """

    def __init__(self, categories: Optional[List[str]] = None, default_intent: str = "unknown"):
        """
        Initializes the IntentClassifierNode.

        :param categories: A list of supported intent labels.
        :param default_intent: The fallback intent if classification fails or is low confidence.
        """
        self._categories = categories or ["greeting", "query", "command", "feedback", "technical_support"]
        self._default_intent = default_intent

    @property
    def node_name(self) -> str:
        """
        Returns the unique identifier for this node type.
        """
        return "IntentClassifierNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes the input data to determine the user's intent.
        
        Args:
            data: The input string to be classified.
            context: Shared execution context for the pipeline.

        Returns:
            A dictionary containing the identified 'intent', 'confidence', 
            and the original 'input_data'.

        Raises:
            ValueError: If the input data is not a string or is empty.
        """
        logger.debug(f"Starting intent classification for node: {self.node_name}")

        if not isinstance(data, str) or not data.strip():
            logger.error("IntentClassifierNode received invalid input data type. Expected non-empty string.")
            raise ValueError("Input data for IntentClassifierNode must be a non-empty string.")

        try:
            # Note: In a production scenario, this section would interface with an
            # NLU model or an LLM prompt. For this implementation, we simulate
            # classification logic based on semantic markers.
            
            normalized_input = data.lower().strip()
            detected_intent = self._default_intent
            confidence = 0.0

            # Simple heuristic simulation for framework demonstration
            if any(word in normalized_input for word in ["help", "support", "broken", "error"]):
                detected_intent = "technical_support"
                confidence = 0.92
            elif any(word in normalized_input for word in ["hello", "hi", "hey"]):
                detected_intent = "greeting"
                confidence = 0.98
            elif normalized_input.endswith("?"):
                detected_intent = "query"
                confidence = 0.85
            else:
                # Defaulting logic
                detected_intent = self._categories[0] if self._categories else self._default_intent
                confidence = 0.50

            result = {
                "intent": detected_intent,
                "confidence": confidence,
                "original_input": data,
                "metadata": {
                    "processor": self.node_name,
                    "categories_evaluated": self._categories
                }
            }

            # Update context for downstream nodes
            context["last_intent"] = detected_intent
            
            logger.info(f"Successfully classified intent as '{detected_intent}' (conf: {confidence})")
            return result

        except Exception as e:
            logger.exception(f"Critical error during intent classification: {str(e)}")
            raise RuntimeError(f"IntentClassifierNode failed to process data: {e}") from e

    def __repr__(self) -> str:
        return f"<IntentClassifierNode(categories={self._categories})>"