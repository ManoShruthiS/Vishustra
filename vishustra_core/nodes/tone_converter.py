import logging
from typing import Any, Dict

# Assuming vishustra_core.nodes.base_node provides the BaseNode class
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ToneConverter(BaseNode):
    """
    A Vishustra processing node that converts the tone of a given text.
    It expects a string as input data and a 'target_tone' in the context.
    """

    def __init__(self):
        """
        Initializes the ToneConverter node.
        No special setup required for this simulated version.
        """
        super().__init__()
        logger.debug(f"[{self.node_name}] Node initialized.")

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node.
        """
        return "ToneConverter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, attempting to convert its tone based on the
        'target_tone' specified in the context.

        Args:
            data: The input data, expected to be a string.
            context: A dictionary containing operational context,
                     expected to have 'target_tone' (str).

        Returns:
            The processed data (string with converted tone) or the original
            data if conversion fails or is not applicable.
        """
        if not isinstance(data, str):
            logger.warning(
                f"[{self.node_name}] Received non-string data of type {type(data).__name__}. "
                "Tone conversion requires string input. Returning original data."
            )
            return data

        target_tone = context.get("target_tone")
        if not target_tone or not isinstance(target_tone, str):
            logger.error(
                f"[{self.node_name}] 'target_tone' not found or not a string in context. "
                "Cannot perform tone conversion. Context: %s", context
            )
            return data

        original_text = data
        converted_text = original_text

        try:
            # Simulate tone conversion based on the specified target_tone.
            # In a real-world scenario, this would involve NLP models,
            # LLM calls, or sophisticated rule-based systems.
            tone_map = {
                "formal": self._to_formal,
                "informal": self._to_informal,
                "professional": self._to_professional,
                "empathetic": self._to_empathetic,
                "direct": self._to_direct,
            }
            
            converter_func = tone_map.get(target_tone.lower())

            if converter_func:
                converted_text = converter_func(original_text)
                logger.info(
                    f"[{self.node_name}] Successfully converted text to '{target_tone}' tone."
                )
            else:
                logger.warning(
                    f"[{self.node_name}] Unknown target tone '{target_tone}'. "
                    "Returning original data. Supported tones: " + ", ".join(tone_map.keys())
                )
                return original_text

            return converted_text

        except Exception as e:
            logger.exception(
                f"[{self.node_name}] An unexpected error occurred during tone conversion "
                f"for target '{target_tone}': {e}"
            )
            # In case of an unexpected error, return the original data to ensure pipeline continuity.
            return original_text

    # --- Private Helper Methods for Tone Simulation ---
    # These methods provide a highly simplified simulation of tone conversion.
    # They are illustrative and would be replaced by more robust NLP or LLM logic.

    def _to_formal(self, text: str) -> str:
        """Simulates converting text to a more formal tone."""
        text = text.replace("don't", "do not")
        text = text.replace("can't", "cannot")
        text = text.replace("it's", "it is")
        text = text.replace("i'm", "I am")
        text = text.replace("you're", "you are")
        text = text.replace("guys", "colleagues")
        text = text.replace("hi", "hello")
        text = text.replace("thanks", "thank you")
        return text

    def _to_informal(self, text: str) -> str:
        """Simulates converting text to a more informal tone."""
        text = text.replace("do not", "don't")
        text = text.replace("cannot", "can't")
        text = text.replace("it is", "it's")
        text = text.replace("I am", "I'm")
        text = text.replace("you are", "you're")
        text = text.replace("hello", "hi")
        text = text.replace("thank you", "thanks")
        return text

    def _to_professional(self, text: str) -> str:
        """Simulates converting text to a professional tone."""
        text = self._to_formal(text)  # Build upon formal for simplicity
        text = text.replace("problem", "challenge")
        text = text.replace("issue", "concern")
        text = text.replace("get", "obtain")
        text = text.replace("start", "initiate")
        return text

    def _to_empathetic(self, text: str) -> str:
        """Simulates adding empathetic phrasing to text."""
        if not text.strip().lower().startswith(("i understand", "it sounds like", "i hear you", "i appreciate")):
            # Prepend an empathetic phrase if not already present
            text = "I understand. " + text
        return text

    def _to_direct(self, text: str) -> str:
        """Simulates making text more direct by removing common filler words."""
        text = text.replace("just ", "").replace("actually ", "").replace("you know, ", "").replace("like, ", "")
        # Remove multiple spaces and strip leading/trailing whitespace
        return ' '.join(text.split()).strip()