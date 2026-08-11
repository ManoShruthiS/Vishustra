
import logging
from typing import Any, Dict, Optional

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class IntentClassifierNode(BaseNode):
    """
    A processing node designed to classify the intent of an input text utterance.

    This node simulates intent classification by matching keywords in the input
    against predefined patterns. In a production environment, this would typically
    integrate with a sophisticated machine learning model (e.g., a transformer-based
    classifier or a custom NLU service) to provide robust intent detection
    and confidence scores.
    """

    # For demonstration, a simple keyword-based intent mapping is used.
    # In a real system, this would be loaded from configuration, a model,
    # or an external service.
    _INTENT_PATTERNS: Dict[str, Dict[str, Any]] = {
        "greeting": {"keywords": ["hello", "hi", "hey", "good morning", "good evening"], "confidence": 0.95},
        "farewell": {"keywords": ["bye", "goodbye", "see you later"], "confidence": 0.90},
        "weather_query": {"keywords": ["weather", "forecast", "temperature", "rain", "sunny"], "confidence": 0.88},
        "time_query": {"keywords": ["time", "what time is it", "current time"], "confidence": 0.85},
        "help_request": {"keywords": ["help", "support", "assist me", "trouble"], "confidence": 0.80},
        "create_task": {"keywords": ["create task", "add to do", "new task"], "confidence": 0.92},
        "set_reminder": {"keywords": ["set reminder", "remind me"], "confidence": 0.91},
    }

    def __init__(self):
        """
        Initializes the IntentClassifierNode.

        In a production scenario, this might involve loading a pre-trained
        intent classification model, configurations, or establishing
        connections to external NLU services.
        """
        logger.debug(f"[{self.node_name}] Initializing node.")
        # self.model = load_intent_model() # Placeholder for actual model loading

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "IntentClassifierNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to classify its underlying intent.

        The `data` input is expected to be either a raw string containing the
        user utterance or a dictionary that includes a 'text' key with the utterance.

        Args:
            data: The input utterance, which can be a `str` or a `Dict[str, str]`
                  containing a 'text' key.
            context: A dictionary providing contextual information that might be
                     relevant for intent classification (e.g., user session data,
                     previous turns in a conversation, available tools).

        Returns:
            A dictionary containing the classified 'intent' (str) and a
            'confidence' score (float) indicating the model's certainty.
            If no specific intent is confidently matched, it defaults to
            "unknown" with a low confidence score.

        Raises:
            TypeError: If the input `data` is not a string or a dictionary.
            ValueError: If `data` is a dictionary but does not contain a 'text' key,
                        or if the value associated with 'text' is not a string.
        """
        utterance: Optional[str] = None
        result: Dict[str, Any] = {"intent": "unknown", "confidence": 0.1} # Default fallback

        if isinstance(data, str):
            utterance = data
        elif isinstance(data, dict):
            utterance = data.get("text")
            if utterance is None:
                logger.error(
                    f"[{self.node_name}] Input dictionary 'data' is missing the mandatory 'text' key. Received: {data}"
                )
                raise ValueError(
                    f"[{self.node_name}] Input dictionary 'data' must contain a 'text' key for intent classification."
                )
            if not isinstance(utterance, str):
                logger.error(
                    f"[{self.node_name}] The value associated with 'text' in input data is not a string. Received: {data}"
                )
                raise TypeError(
                    f"[{self.node_name}] The 'text' value in input data must be a string for intent classification."
                )
        else:
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected str or dict, but received {type(data).__name__}. Data: {data}"
            )
            raise TypeError(
                f"[{self.node_name}] Invalid input data type for IntentClassifierNode. Expected str or dict, got {type(data).__name__}."
            )

        processed_utterance = utterance.lower().strip()
        if not processed_utterance:
            logger.warning(f"[{self.node_name}] Received empty or whitespace-only utterance. Defaulting to 'unknown' intent.")
            return result

        logger.debug(f"[{self.node_name}] Classifying intent for utterance: '{processed_utterance}'")

        matched_intent: Optional[str] = None
        highest_confidence: float = 0.0

        for intent_name, config in self._INTENT_PATTERNS.items():
            for keyword in config["keywords"]:
                if keyword in processed_utterance:
                    current_confidence = config["confidence"]
                    if current_confidence > highest_confidence:
                        highest_confidence = current_confidence
                        matched_intent = intent_name
                        logger.debug(
                            f"[{self.node_name}] Potential match for intent '{intent_name}' with keyword '{keyword}' (confidence: {current_confidence:.2f})"
                        )
                    # For a simple keyword model, once a keyword matches, we can prioritize
                    # the first match or the highest confidence if multiple intents share keywords.
                    # Here, we keep searching for the highest confidence match.

        if matched_intent:
            result["intent"] = matched_intent
            result["confidence"] = highest_confidence
            logger.info(
                f"[{self.node_name}] Classified intent: '{matched_intent}' with confidence: {highest_confidence:.2f} for utterance: '{processed_utterance}'"
            )
        else:
            logger.info(
                f"[{self.node_name}] No specific intent matched for utterance: '{processed_utterance}'. Defaulting to 'unknown'."
            )

        return result

