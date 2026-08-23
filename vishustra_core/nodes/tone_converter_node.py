import logging
from typing import Any, Dict

# Assuming the base_node exists at this path relative to the project root
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ToneConverterNode(BaseNode):
    """
    A Vishustra processing node designed to simulate the conversion of text tone.

    This node accepts a string as input and, based on the 'target_tone'
    parameter provided in the context, produces a transformed string reflecting
    the desired tone. In a production environment, this would typically
    orchestrate interactions with an underlying language model or a sophisticated
    NLP engine.
    """

    _SUPPORTED_TONES = {"formal", "informal", "professional", "casual", "sarcastic"}
    """Set of tones explicitly supported by this simulation."""

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this processing node."""
        return "ToneConverter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input text to convert its tone according to the context.

        Args:
            data (Any): The input data, which is expected to be a string (text)
                        for tone conversion.
            context (Dict[str, Any]): A dictionary containing operational context.
                                      It MUST include a 'target_tone' key
                                      (e.g., "formal", "informal") as a string.

        Returns:
            Any: The tone-converted string. If the target tone is unsupported,
                 the original data is returned after logging a warning.

        Raises:
            ValueError: If 'data' is not a string, or if 'target_tone' is
                        missing or not a valid string in the context.
        """
        if not isinstance(data, str):
            logger.error(
                f"{self.node_name}: Received invalid data type. Expected 'str', "
                f"but got '{type(data).__name__}'."
            )
            raise ValueError(
                f"Invalid input data type for {self.node_name}. Expected 'str', "
                f"got '{type(data).__name__}'."
            )

        target_tone = context.get("target_tone")
        if not target_tone or not isinstance(target_tone, str):
            logger.error(
                f"{self.node_name}: 'target_tone' is missing or not a valid string "
                f"in context. Received: '{target_tone}' (type: {type(target_tone).__name__})."
            )
            raise ValueError(
                f"Context must contain a valid 'target_tone' (str) for {self.node_name}."
            )

        normalized_tone = target_tone.lower()
        if normalized_tone not in self._SUPPORTED_TONES:
            logger.warning(
                f"{self.node_name}: Unsupported target tone '{target_tone}'. "
                f"Supported tones are: {', '.join(self._SUPPORTED_TONES)}. "
                f"Returning original data without conversion."
            )
            return data

        logger.info(f"{self.node_name}: Initiating text tone conversion to '{normalized_tone}'.")

        # Simulate tone conversion based on the specified tone
        converted_text = self._simulate_tone_conversion(data, normalized_tone)

        return converted_text

    def _simulate_tone_conversion(self, text: str, target_tone: str) -> str:
        """
        Simulates the tone conversion of the given text.

        This method provides a placeholder for actual tone transformation logic.
        In a real-world implementation within Vishustra, this would typically
        involve making a request to an external LLM service, utilizing
        a specialized NLP library, or applying more complex linguistic rules.

        Args:
            text (str): The original text to convert.
            target_tone (str): The desired tone for the output text.

        Returns:
            str: The text with its tone simulated to be converted.
        """
        processed_text = text.strip()

        if target_tone == "formal":
            if processed_text:
                processed_text = processed_text[0].upper() + processed_text[1:]
                if not any(processed_text.endswith(p) for p in ['.', '!', '?']):
                    processed_text += '.'
            processed_text = (
                processed_text.replace("hey", "Greetings")
                .replace("what's up", "How do you do")
                .replace("gonna", "going to")
                .replace("wanna", "want to")
            )
            return processed_text

        elif target_tone == "informal":
            processed_text = processed_text.lower()
            processed_text = (
                processed_text.replace("i am", "i'm")
                .replace("thank you", "thanks")
                .replace("apologies", "sorry")
            )
            if not text.lower().startswith("yo"):
                processed_text = "yo, " + processed_text
            return processed_text

        elif target_tone == "professional":
            if processed_text:
                processed_text = processed_text[0].upper() + processed_text[1:]
            processed_text = (
                processed_text.replace("mate", "colleague")
                .replace("cool", "satisfactory")
                .replace("get together", "meeting")
            )
            return processed_text

        elif target_tone == "casual":
            processed_text = processed_text.lower()
            processed_text = (
                processed_text.replace("dear", "hi")
                .replace("sincerely", "cheers")
                .replace("however", "but")
            )
            if not text.lower().startswith("just wanted to say"):
                processed_text = "just wanted to say, " + processed_text
            return processed_text

        elif target_tone == "sarcastic":
            sarcastic_chars = []
            for i, char in enumerate(processed_text):
                if char.isalpha():
                    sarcastic_chars.append(char.upper() if i % 2 == 0 else char.lower())
                else:
                    sarcastic_chars.append(char)
            processed_text = "".join(sarcastic_chars)
            if not text.lower().startswith("oh, sure"):
                processed_text = "Oh, sure, " + processed_text + ". Obviously."
            return processed_text

        # Fallback in case a supported tone somehow bypasses specific logic
        logger.debug(f"{self.node_name}: No specific simulation logic for '{target_tone}'. Returning original text.")
        return text