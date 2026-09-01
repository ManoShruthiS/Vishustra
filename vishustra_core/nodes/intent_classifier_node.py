import logging
from typing import Any, Dict, List

# Assuming vishustra_core is the package root and base_node.py is located at vishustra_core/nodes/base_node.py
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class IntentClassifierNode(BaseNode):
    """
    A Vishustra processing node that simulates intent classification for text input.

    This node takes a string (typically a user utterance) and attempts to classify
    its intent based on a set of predefined keyword patterns. It serves as a
    demonstrative and testing component within the framework, simulating an
    intent detection service.
    """

    # Predefined keyword patterns for intent simulation. In a real system,
    # this would involve NLP models, external APIs, or more sophisticated matching.
    _intent_patterns: Dict[str, List[str]] = {
        "BookFlight": ["book flight", "fly to", "flight from", "travel to"],
        "CheckWeather": ["weather in", "how's the weather", "temperature in", "forecast for"],
        "OrderFood": ["order food", "pizza delivery", "hungry for", "delivery from"],
        "PlayMusic": ["play music", "start song", "music by", "listen to"],
        "GetNews": ["latest news", "news about", "tell me news", "current events"],
    }

    def __init__(self, fallback_intent: str = "unknown", min_confidence: float = 0.5):
        """
        Initializes the IntentClassifierNode with configuration options.

        Args:
            fallback_intent: The intent label to assign if no specific intent
                             is confidently detected. Defaults to "unknown".
            min_confidence: A simulated confidence threshold. If the detected
                            intent's confidence is below this, the fallback
                            intent is returned. (In this simulation, confidence
                            is either 1.0 for a match or 0.0).
        """
        self._fallback_intent = fallback_intent
        self._min_confidence = min_confidence
        logger.info(
            f"IntentClassifierNode initialized. Fallback intent: '{fallback_intent}', "
            f"Min confidence threshold: {min_confidence:.2f}"
        )

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "IntentClassifier"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to classify its intent based on predefined patterns.

        Args:
            data: The input utterance as a string that needs intent classification.
            context: A dictionary containing contextual information relevant to the
                     orchestration. This node does not directly use context for
                     classification but is part of the standard node interface.

        Returns:
            A dictionary containing the classified 'intent' and a simulated 'confidence' score.
            Example: `{'intent': 'BookFlight', 'confidence': 1.0}`
                     `{'intent': 'unknown', 'confidence': 0.0}`

        Raises:
            TypeError: If the input `data` is not a string.
            ValueError: If the input `data` is an empty string after stripping whitespace.
        """
        if not isinstance(data, str):
            logger.error(
                f"IntentClassifierNode received invalid data type. Expected 'str', "
                f"but got '{type(data).__name__}'."
            )
            raise TypeError(
                f"IntentClassifierNode expects string input for 'data', "
                f"but received {type(data).__name__}."
            )

        utterance = data.strip().lower()

        if not utterance:
            logger.warning("IntentClassifierNode received an empty or whitespace-only utterance.")
            raise ValueError("IntentClassifierNode cannot classify an empty string utterance.")

        detected_intent: str = self._fallback_intent
        confidence: float = 0.0

        for intent_name, keywords in self._intent_patterns.items():
            for keyword in keywords:
                if keyword in utterance:
                    detected_intent = intent_name
                    confidence = 1.0  # Simulate high confidence for a direct keyword match
                    logger.debug(
                        f"Matched keyword '{keyword}' for intent '{intent_name}' "
                        f"in utterance: '{utterance}'"
                    )
                    break  # Found a match, no need to check other keywords for this intent
            if confidence == 1.0:
                break  # Found a definitive intent, no need to check other intents

        # Apply the confidence threshold. In this simulation, if we found a match,
        # confidence is 1.0. If not, it's 0.0. This check ensures the fallback
        # logic is applied if 0.0 is below _min_confidence.
        if confidence < self._min_confidence:
            detected_intent = self._fallback_intent
            confidence = 0.0
            logger.info(
                f"Detected intent's simulated confidence ({confidence:.2f}) is below "
                f"threshold ({self._min_confidence:.2f}). Falling back to "
                f"'{self._fallback_intent}'."
            )

        result = {"intent": detected_intent, "confidence": confidence}
        logger.info(
            f"Classified utterance '{utterance[:75]}{'...' if len(utterance) > 75 else ''}' "
            f"to: {result}"
        )
        return result
