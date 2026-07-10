import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ToneConverter(BaseNode):
    """
    A Vishustra node designed to convert the tone of input text.

    This node takes a string as input and, based on the 'target_tone' specified
    in the context, applies a simulated transformation to achieve a desired
    textual style. Supported tones include 'formal', 'informal', 'emphatic',
    'sarcastic', and 'neutral'.
    """

    _SUPPORTED_TONES = {
        "formal",
        "informal",
        "neutral",
        "emphatic",
        "sarcastic",
    }

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "ToneConverter"

    def _apply_formal_tone(self, text: str) -> str:
        """Simulates transforming text to a formal tone."""
        # Simple simulation: capitalize sentence beginnings, avoid common contractions
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        processed_sentences = [s.capitalize() for s in sentences]
        transformed_text = ". ".join(processed_sentences)
        transformed_text = transformed_text.replace("don't", "do not").replace("can't", "cannot")
        transformed_text = transformed_text.replace("it's", "it is").replace("i'm", "I am")
        return transformed_text.strip() + ("." if transformed_text and not transformed_text.endswith('.') else "")

    def _apply_informal_tone(self, text: str) -> str:
        """Simulates transforming text to an informal tone."""
        # Simple simulation: use contractions, add a casual opening/closing
        transformed_text = text.replace("do not", "don't").replace("cannot", "can't")
        transformed_text = transformed_text.replace("it is", "it's").replace("I am", "i'm")
        if not transformed_text.strip().endswith(('.', '?', '!')):
            transformed_text += " :)"
        return "Hey there! " + transformed_text

    def _apply_emphatic_tone(self, text: str) -> str:
        """Simulates transforming text to an emphatic tone."""
        # Simple simulation: add exclamation marks, capitalize some words
        words = text.split()
        transformed_words = []
        for i, word in enumerate(words):
            # Arbitrary rule to capitalize some words for emphasis
            if len(word) > 2 and i % 3 == 0:
                transformed_words.append(word.upper())
            else:
                transformed_words.append(word)
        transformed_text = " ".join(transformed_words)
        if not transformed_text.strip().endswith('!'):
            transformed_text += "!!!"
        return transformed_text

    def _apply_sarcastic_tone(self, text: str) -> str:
        """Simulates transforming text to a sarcastic tone."""
        # Simple simulation: interleave case and add a sarcastic tag
        sarcastic_chars = []
        for i, char in enumerate(text):
            sarcastic_chars.append(char.lower() if i % 2 == 0 else char.upper())
        sarcastic_text = "".join(sarcastic_chars)
        return sarcastic_text + " (obviously)"


    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, converting its tone based on the 'target_tone'
        specified in the context.

        Args:
            data (Any): The input data, expected to be a string representing the text
                        whose tone needs to be converted.
            context (Dict[str, Any]): A dictionary containing contextual information.
                                     It must include 'target_tone' (str) which specifies
                                     the desired output tone (e.g., 'formal', 'informal').

        Returns:
            Any: The transformed text string, reflecting the new tone. If the input data
                 is not a string or an error occurs during conversion, the original data
                 might be returned, and an error logged.

        Raises:
            TypeError: If the input 'data' is not a string.
            ValueError: If 'target_tone' in context is provided but specifies an
                        unsupported tone. (Note: currently handles by defaulting to 'neutral'
                        and logging a warning instead of raising ValueError).
        """
        logger.debug(f"[{self.node_name}] Initiating tone conversion process.")

        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected string, "
                f"received {type(data).__name__}. This node only processes strings."
            )
            raise TypeError(f"Input data for {self.node_name} must be a string, but received {type(data).__name__}.")

        # Get target tone from context, default to 'neutral' if not specified
        target_tone = context.get("target_tone", "neutral").lower()

        if target_tone not in self._SUPPORTED_TONES:
            logger.warning(
                f"[{self.node_name}] Unsupported target_tone '{target_tone}' in context. "
                f"Supported tones are: {', '.join(self._SUPPORTED_TONES)}. "
                f"Defaulting to 'neutral' tone for processing."
            )
            target_tone = "neutral"

        transformed_text = data
        try:
            if target_tone == "formal":
                transformed_text = self._apply_formal_tone(data)
            elif target_tone == "informal":
                transformed_text = self._apply_informal_tone(data)
            elif target_tone == "emphatic":
                transformed_text = self._apply_emphatic_tone(data)
            elif target_tone == "sarcastic":
                transformed_text = self._apply_sarcastic_tone(data)
            elif target_tone == "neutral":
                # For neutral, no specific tone transformation is applied.
                # Could apply basic normalization if needed, but for now, it's a pass-through.
                pass
            # No `else` block needed as target_tone is validated and defaulted earlier.

            logger.info(f"[{self.node_name}] Successfully converted text to '{target_tone}' tone.")
            logger.debug(
                f"[{self.node_name}] Original text (first 100 chars): '{data[:100]}...'\n"
                f"[{self.node_name}] Transformed text (first 100 chars): '{transformed_text[:100]}...'"
            )
        except Exception as e:
            logger.error(
                f"[{self.node_name}] An unexpected error occurred during tone conversion to '{target_tone}': {e}",
                exc_info=True
            )
            # In case of an internal error during conversion, return the original data
            # to prevent pipeline breakage and log the failure.
            transformed_text = data

        return transformed_text