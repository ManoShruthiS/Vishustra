import logging
from typing import Any, Dict

# Assuming vishustra_core.nodes.base_node exists in the project structure
# In a real scenario, this would be `from vishustra.core.nodes.base_node import BaseNode`
# or similar, depending on the exact project layout.
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class IntentClassifierNode(BaseNode):
    """
    A Vishustra processing node that classifies the intent of a given text input.

    This node simulates intent classification based on simple keyword matching.
    In a real-world production setup, this would typically wrap a sophisticated
    machine learning model (e.g., a transformer-based model fine-tuned for intent
    recognition or a rules-based system with advanced NLP techniques).
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "IntentClassifierNode"

    def _simulate_intent_classification(self, text: str) -> Dict[str, Any]:
        """
        Performs a simulated intent classification based on keywords.
        This method serves as a placeholder for a more complex ML model inference.

        Args:
            text: The input text to classify.

        Returns:
            A dictionary containing the classified intent and a simulated confidence score.
        """
        lower_text = text.lower()

        if any(keyword in lower_text for keyword in ["hello", "hi", "hey", "good morning", "greetings"]):
            return {"intent": "greeting", "confidence": 0.98}
        elif any(keyword in lower_text for keyword in ["weather", "forecast", "temperature", "climate"]):
            return {"intent": "weather_query", "confidence": 0.92}
        elif any(keyword in lower_text for keyword in ["book flight", "travel plans", "reservation", "ticket"]):
            return {"intent": "flight_booking", "confidence": 0.89}
        elif any(keyword in lower_text for keyword in ["order food", "pizza", "sushi", "restaurant", "menu"]):
            return {"intent": "food_ordering", "confidence": 0.85}
        elif any(keyword in lower_text for keyword in ["set alarm", "reminder", "timer"]):
            return {"intent": "utility_alarm_timer", "confidence": 0.87}
        elif any(keyword in lower_text for keyword in ["play music", "song", "artist", "album"]):
            return {"intent": "music_playback", "confidence": 0.90}
        elif any(keyword in lower_text for keyword in ["what is", "tell me about", "who is", "information on"]):
            return {"intent": "information_retrieval", "confidence": 0.80}
        else:
            return {"intent": "unclassified", "confidence": 0.50}

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to classify its underlying intent.

        The `data` input is expected to be a string representing a user query or text snippet.
        The `context` dictionary can be used to pass additional information such as
        user language, session history, or configuration parameters for the
        underlying intent classification model (though not used in this simulated version).

        Args:
            data: The input data to be processed. Expected to be a string.
            context: A dictionary containing contextual information.

        Returns:
            A dictionary containing the classified intent and its confidence.
            Example: {"intent": "greeting", "confidence": 0.98}
                     {"intent": "unclassified", "confidence": 0.50}

        Raises:
            ValueError: If the input `data` is not a string, indicating an invalid input type.
            RuntimeError: If an unexpected error occurs during the classification process.
        """
        logger.debug(f"[{self.node_name}] Attempting to classify intent for input data.")

        if not isinstance(data, str):
            error_msg = (
                f"[{self.node_name}] Invalid input data type. "
                f"Expected 'str', received '{type(data).__name__}'."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            # Perform the actual (or simulated) intent classification
            classified_result = self._simulate_intent_classification(data)
            logger.info(
                f"[{self.node_name}] Classified '{data[:70]}{'...' if len(data) > 70 else ''}' "
                f"as intent: '{classified_result['intent']}' with confidence: {classified_result['confidence']:.2f}."
            )
            return classified_result
        except Exception as e:
            # Catch any unexpected errors from the classification logic
            error_msg = (
                f"[{self.node_name}] An unexpected error occurred during intent classification for data: "
                f"'{data[:100]}{'...' if len(data) > 100 else ''}'. Error: {e}"
            )
            logger.exception(error_msg)  # Log exception details
            raise RuntimeError(error_msg) from e
