import logging
from typing import Any, Dict, List, Optional
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class IntentClassifierNode(BaseNode):
    """
    Analyzes input text to categorize the user's intent.
    
    This node serves as a traffic controller within the Vishustra orchestration
    pipeline, allowing downstream nodes to branch logic based on the 
    identified intent (e.g., 'query', 'action', 'greeting', or 'feedback').
    """

    def __init__(self, categories: Optional[List[str]] = None):
        """
        Initializes the IntentClassifierNode.
        
        :param categories: Optional list of specific intent labels to prioritize.
        """
        self.categories = categories or ["information_retrieval", "task_execution", "navigation", "general_chat"]
        self._default_intent = "unclassified"

    @property
    def node_name(self) -> str:
        """
        Returns the unique identifier for this node type.
        """
        return "intent_classifier_v1"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to determine the primary intent.
        
        Expects 'data' to be a string representing the user input.
        Returns a dictionary containing the identified intent and classification confidence.
        """
        try:
            if not isinstance(data, str):
                logger.error(f"[{self.node_name}] Input data must be a string. Received: {type(data)}")
                raise TypeError(f"Node '{self.node_name}' requires string input.")

            input_text = data.strip()
            if not input_text:
                logger.warning(f"[{self.node_name}] Received empty string for classification.")
                return self._generate_response(input_text, self._default_intent, 0.0)

            # In a production LLM orchestration framework, this section would typically 
            # involve an embedding-based lookup or a zero-shot LLM classification call.
            # Here we simulate the logic for the node's architectural role.
            detected_intent, confidence = self._perform_classification(input_text)

            logger.info(f"[{self.node_name}] Intent identified: {detected_intent} (conf: {confidence})")
            
            return self._generate_response(input_text, detected_intent, confidence)

        except Exception as e:
            logger.exception(f"[{self.node_name}] Critical failure during intent classification: {str(e)}")
            raise RuntimeError(f"IntentClassifierNode failed to process input: {e}") from e

    def _perform_classification(self, text: str) -> (str, float):
        """
        Internal logic to simulate classification. 
        In an integrated Vishustra deployment, this would interface with a 
        specialized ModelNode or external API.
        """
        normalized_text = text.lower()
        
        # Simple heuristic mapping for demonstration of the node's output structure
        if any(kw in normalized_text for kw in ["find", "search", "who", "what", "where"]):
            return "information_retrieval", 0.92
        elif any(kw in normalized_text for kw in ["do", "run", "execute", "create", "delete"]):
            return "task_execution", 0.88
        elif any(kw in normalized_text for kw in ["go to", "open", "show me"]):
            return "navigation", 0.85
        
        return "general_chat", 0.70

    def _generate_response(self, text: str, intent: str, confidence: float) -> Dict[str, Any]:
        """
        Constructs the standardized output payload for the IntentClassifierNode.
        """
        return {
            "input": text,
            "classification": {
                "primary_intent": intent,
                "confidence_score": confidence,
                "available_labels": self.categories
            },
            "metadata": {
                "node_id": self.node_name,
                "version": "1.0.4"
            }
        }