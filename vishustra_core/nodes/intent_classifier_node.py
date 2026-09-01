import logging
from typing import Any, Dict

# Assuming this path from the project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class IntentClassifierNode(BaseNode):
    """
    A processing node that simulates intent classification for user queries.
    It identifies a primary intent from a predefined set based on keywords or
    delegates to a configured model.
    """

    def __init__(self, model_identifier: str = "dummy-keyword-classifier"):
        """
        Initializes the IntentClassifierNode.

        Args:
            model_identifier (str): A string identifying the classification model to use.
                                    In a production environment, this would dictate
                                    which specific NLP model (e.g., a fine-tuned transformer)
                                    is loaded and utilized. For this simulation, it's a label.
        """
        self._model_identifier = model_identifier
        logger.info(f"IntentClassifierNode initialized with model: {self._model_identifier}")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "IntentClassifier"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data (assumed to be a string query) to classify its intent.

        This method simulates the classification process by analyzing keywords within
        the input query. In a real-world scenario, this would involve calling a
        machine learning model's inference endpoint or local model.

        Args:
            data (Any): The input data, expected to be a string containing the user query.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       relevant for processing, which might include
                                       session data, user preferences, etc.
                                       (Not directly used in this simulation, but important for signature).

        Returns:
            Dict[str, Any]: A dictionary containing the classified intent, a simulated
                            confidence score, the original query, and the model identifier used.
                            Example: {"intent": "order_status", "confidence": 0.95, "query": "track my order"}
                            If no clear intent is found, it defaults to "unknown" with a lower confidence.

        Raises:
            ValueError: If the input data is not a string, as this node specifically
                        expects textual input for classification.
            Exception: For any unexpected errors that occur during the classification logic.
        """
        if not isinstance(data, str):
            logger.error(
                f"IntentClassifierNode received non-string data type: {type(data)}. "
                "Expected a string query for intent classification."
            )
            raise ValueError(f"Input data for '{self.node_name}' must be a string, but got {type(data).__name__}.")

        query: str = data.lower().strip()
        classified_intent: str = "unknown"
        confidence: float = 0.5  # Default confidence for 'unknown' intent

        try:
            # --- Simulated Intent Classification Logic ---
            # This section would typically involve calling an external NLP service
            # or a loaded local ML model for inference.
            if any(phrase in query for phrase in ["order status", "track my order", "where is my package"]):
                classified_intent = "order_status"
                confidence = 0.95
            elif any(phrase in query for phrase in ["make a reservation", "book a table", "reserve a spot", "book an appointment"]):
                classified_intent = "make_reservation"
                confidence = 0.92
            elif any(phrase in query for phrase in ["contact support", "help me", "technical issue", "talk to an agent"]):
                classified_intent = "contact_support"
                confidence = 0.90
            elif any(phrase in query for phrase in ["greeting", "hello", "hi", "hey"]):
                classified_intent = "greeting"
                confidence = 0.85
            elif any(phrase in query for phrase in ["thanks", "thank you", "appreciate"]):
                classified_intent = "gratitude"
                confidence = 0.80
            else:
                logger.debug(f"No specific predefined intent found for query: '{query}'. Defaulting to 'unknown'.")
                # Confidence remains default 0.5 for 'unknown'

        except Exception as e:
            logger.exception(
                f"An unexpected error occurred during intent classification for query: '{query}' "
                f"using model: '{self._model_identifier}'."
            )
            # Re-raise the exception to propagate the error up the orchestration chain
            raise

        result = {
            "intent": classified_intent,
            "confidence": confidence,
            "query": data,  # Preserve the original casing of the query
            "model_identifier_used": self._model_identifier,
        }
        logger.debug(
            f"Query: '{query}' classified as intent: '{classified_intent}' "
            f"with simulated confidence: {confidence:.2f} using model '{self._model_identifier}'."
        )
        return result
