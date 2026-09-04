import logging
import re
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ToneConverterNode(BaseNode):
    """
    A Vishustra processing node that converts the tone of input text.

    This node expects a string as input data and a 'target_tone' key in
    the context dictionary to determine the desired output tone.
    Supported tones include 'formal', 'informal', 'enthusiastic', and 'neutral'.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "ToneConverter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input text data by converting its tone based on the
        'target_tone' specified in the context.

        Args:
            data (Any): The input data, expected to be a string (text).
            context (Dict[str, Any]): A dictionary containing contextual information.
                                      Must include a 'target_tone' string key.

        Returns:
            Any: The tone-converted string.

        Raises:
            TypeError: If the input 'data' is not a string.
            ValueError: If 'target_tone' is missing or invalid in the context.
            RuntimeError: If an unexpected error occurs during tone conversion.
        """
        if not isinstance(data, str):
            logger.error(
                "ToneConverter received non-string data. Expected string, got %s.",
                type(data).__name__
            )
            raise TypeError("ToneConverter expects string data for processing.")

        target_tone = context.get("target_tone")
        if not target_tone or not isinstance(target_tone, str):
            logger.error(
                "Context missing or invalid 'target_tone' key for ToneConverter. "
                "Expected a string 'target_tone'."
            )
            raise ValueError(
                "Context must contain a string 'target_tone' for ToneConverter."
            )

        original_text = data
        processed_text = original_text
        normalized_tone = target_tone.lower().strip()

        logger.debug(
            "ToneConverter processing text for target tone: '%s'", normalized_tone
        )

        try:
            if normalized_tone == "formal":
                # Ensure each sentence starts with a capital letter and ends with a period.
                sentences = re.split(r'(?<=[.!?])\s*', original_text.strip())
                formal_sentences = []
                for sentence in sentences:
                    if sentence.strip():
                        cleaned_sentence = sentence.strip()
                        # Capitalize first letter
                        formal_sentence = cleaned_sentence[0].upper() + cleaned_sentence[1:]
                        # Ensure it ends with a period if it's a statement
                        if not re.search(r'[.!?]$', formal_sentence):
                            formal_sentence += '.'
                        formal_sentences.append(formal_sentence)
                processed_text = ' '.join(formal_sentences)

            elif normalized_tone == "informal":
                # Convert to lowercase, remove some punctuation for a casual feel.
                processed_text = original_text.lower()
                # Simple replacement for common formal elements
                processed_text = re.sub(r'\.{3,}', '...', processed_text) # Keep ellipses
                processed_text = re.sub(r'[.!?]$', '', processed_text) # Remove trailing punctuation
                processed_text = re.sub(r'\b(?:very|quite|rather)\b', '', processed_text, flags=re.IGNORECASE) # Remove intensifiers
                if not processed_text.endswith("!") and not processed_text.endswith("?"):
                     processed_text = processed_text.strip() # Remove trailing space after removing punctuation

            elif normalized_tone == "enthusiastic":
                # Add exclamation marks and inject a positive sentiment.
                processed_text = original_text.strip()
                if not processed_text.endswith("!"):
                    processed_text += "!!!"
                if not processed_text.lower().startswith("wow") and not processed_text.lower().startswith("great"):
                     processed_text = "Wow! " + processed_text

            elif normalized_tone == "neutral":
                # Clean up leading/trailing spaces and ensure initial capitalization
                processed_text = original_text.strip()
                if processed_text:
                    processed_text = processed_text[0].upper() + processed_text[1:]
                # Ensure only one trailing punctuation if any.
                processed_text = re.sub(r'[.!?]{2,}$', '.', processed_text)
                if not re.search(r'[.!?]$', processed_text) and processed_text:
                    processed_text += '.'

            else:
                logger.warning(
                    "Unsupported target tone '%s' provided for ToneConverter. "
                    "Returning original data.",
                    target_tone
                )
                processed_text = original_text

        except Exception as e:
            logger.error(
                "An unexpected error occurred during tone conversion for tone '%s': %s",
                target_tone, e, exc_info=True
            )
            raise RuntimeError(
                f"Failed to convert tone for '{target_tone}' due to an internal error."
            ) from e

        logger.debug(
            "ToneConverter finished processing. Original: '%s', Processed: '%s'",
            original_text, processed_text
        )
        return processed_text
