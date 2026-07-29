import logging
from typing import Any, Dict, Union

# Assuming the specified import path from the project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class IntentClassifierNode(BaseNode):
    """
    A Vishustra processing node responsible for classifying the intent of a given text input.

    This node simulates intent classification based on predefined keywords. It's designed
    to be extensible, allowing for integration with more sophisticated ML models in the future.
    It expects the input data to be either a raw string utterance or a dictionary
    containing the utterance under a 'text' key.
    """

    @property
    def node_name(self) -> str:
        """Returns the unique and descriptive name of this node."""
        return "IntentClassifier"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to classify its underlying intent.

        The method extracts text from the input `data` and uses a keyword-based
        approach to determine the most likely user intent, returning the
        classified intent and a confidence score.

        Args:
            data: The input data to be classified. This can be:
                  - A `str`: The direct utterance to classify.
                  - A `Dict[str, Any]`: Expected to contain a 'text' key
                                        whose value is the utterance string.
            context: A dictionary providing additional runtime context for the processing.
                     This can include session ID, user profile, previous turns, etc.
                     (Currently unused by the core classification logic in this implementation
                     but available for advanced scenarios like context-aware classification).

        Returns:
            A `Dict[str, Any]` containing:
            - 'intent': A string representing the classified intent (e.g., "schedule_meeting").
            - 'confidence': A float representing the confidence score (0.0 to 1.0)
                            for the classified intent.

        Raises:
            ValueError: If the input `data` is not in an expected format (string
                        or a dictionary with a 'text' string key).
            Exception: For any unforeseen errors during the classification process,
                       ensuring robust error propagation.
        """
        text_to_classify: str = ""
        result: Dict[str, Any] = {"intent": "unknown", "confidence": 0.0}

        try:
            if isinstance(data, str):
                text_to_classify = data
            elif isinstance(data, dict) and "text" in data and isinstance(data["text"], str):
                text_to_classify = data["text"]
            else:
                logger.error(
                    f"IntentClassifierNode received invalid data format. "
                    f"Expected str or dict with 'text' key, got: {type(data)}."
                )
                raise ValueError(
                    "Invalid input data format. Expected a string or a dictionary "
                    "with a 'text' key containing the utterance."
                )

            if not text_to_classify.strip():
                logger.warning("Received empty or whitespace-only text for intent classification.")
                result["intent"] = "empty_query"
                result["confidence"] = 0.5
                return result

            # Normalize text for case-insensitive keyword matching
            normalized_text = text_to_classify.lower()

            # --- Simulated Keyword-Based Intent Classification Logic ---
            # In a production system, this would typically involve a trained ML model
            # (e.g., fine-tuned BERT, Llama, or a custom NLU service call).
            # This simulation provides a basic yet functional example.
            intent_mapping = {
                "schedule_meeting": {"keywords": ["schedule", "meeting", "appointment", "calendar"], "confidence": 0.95},
                "place_order": {"keywords": ["order", "buy", "purchase", "item", "checkout"], "confidence": 0.90},
                "customer_support": {"keywords": ["help", "support", "issue", "problem", "assist", "troubleshoot"], "confidence": 0.88},
                "check_status": {"keywords": ["status", "track", "delivery", "where is my"], "confidence": 0.85},
                "general_inquiry": {"keywords": ["what is", "tell me about", "information", "how to"], "confidence": 0.75}
            }

            classified_intent = "unknown_intent"
            confidence = 0.4  # Base confidence for truly unknown intent

            for intent_name, details in intent_mapping.items():
                if any(keyword in normalized_text for keyword in details["keywords"]):
                    classified_intent = intent_name
                    confidence = details["confidence"]
                    break  # Assign the first matching intent found

            result["intent"] = classified_intent
            result["confidence"] = confidence

            logger.debug(
                f"Intent classified for '{text_to_classify[:70]}{'...' if len(text_to_classify) > 70 else ''}': "
                f"Intent='{classified_intent}', Confidence={confidence:.2f}"
            )
            return result

        except ValueError:
            # Re-raise ValueErrors as they indicate invalid input data specific to this node
            raise
        except Exception as e:
            logger.critical(
                f"An unexpected error occurred in IntentClassifierNode while processing data: '{data}'. "
                f"Error: {e}", exc_info=True
            )
            # For critical, unhandled exceptions, re-raise to allow upstream orchestration
            # to handle the failure according to its retry/error handling policy.
            raise
