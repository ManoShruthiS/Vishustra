import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class ToneConverterNode(BaseNode):
    """
    A processing node designed to convert the tone of text data based on a specified
    target tone provided in the execution context. This node simulates various tone
    transformations, ensuring flexibility in LLM output presentation.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "ToneConverter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data (expected to be a string) and converts its
        tone based on the 'target_tone' specified in the context.

        Supported tones (simulated): "formal", "informal", "professional", "concise".
        If an unsupported tone is provided, the original data is returned with a warning.

        Args:
            data: The input text data to be converted. Must be a string.
            context: A dictionary containing execution context parameters.
                     Expected to include 'target_tone' (str).

        Returns:
            The text data with its tone converted, or the original data if conversion
            is not possible or the tone is unsupported.

        Raises:
            TypeError: If the input 'data' is not a string.
            ValueError: If 'target_tone' is missing or not a string in the context.
            RuntimeError: For unexpected errors during the tone conversion process.
        """
        if not isinstance(data, str):
            logger.error(
                "ToneConverterNode received non-string data. Expected 'str', got '%s'.",
                type(data).__name__
            )
            raise TypeError(
                f"ToneConverterNode expects string data, but received {type(data).__name__}"
            )

        target_tone = context.get("target_tone")
        if not target_tone or not isinstance(target_tone, str):
            logger.error(
                "ToneConverterNode requires 'target_tone' (string) in the context "
                "to perform tone conversion."
            )
            raise ValueError(
                "Missing or invalid 'target_tone' in context for ToneConverterNode. "
                "Please provide a string value for 'target_tone'."
            )

        original_text = data
        transformed_text = original_text  # Initialize with original text

        logger.debug(
            "Attempting to convert text tone to '%s' using ToneConverterNode.",
            target_tone
        )

        try:
            lower_target_tone = target_tone.lower().strip()

            if lower_target_tone == "formal":
                # Simulate formal tone: capitalize sentences, replace contractions, ensure punctuation
                transformed_text = original_text.replace("don't", "do not").replace("it's", "it is")
                if not transformed_text.strip().endswith(('.', '!', '?')):
                    transformed_text += '.'
                transformed_text = transformed_text.strip()
                # Basic sentence capitalization. More robust solutions would use NLP.
                if transformed_text:
                    transformed_text = transformed_text[0].upper() + transformed_text[1:]
                logger.debug("Text converted to formal tone.")

            elif lower_target_tone == "informal":
                # Simulate informal tone: lowercase, replace punctuation with exclamation/ellipsis
                transformed_text = original_text.lower().replace('.', '').replace('?', '')
                if not transformed_text.strip().endswith('!'):
                    transformed_text += '!'
                logger.debug("Text converted to informal tone.")

            elif lower_target_tone == "professional":
                # Simulate professional tone: clear, concise, avoid slang
                transformed_text = original_text.replace("gonna", "going to").replace("wanna", "want to")
                if not transformed_text.strip().endswith(('.', '!', '?')):
                    transformed_text += '.'
                transformed_text = transformed_text.strip()
                if transformed_text:
                    transformed_text = transformed_text[0].upper() + transformed_text[1:]
                logger.debug("Text converted to professional tone.")

            elif lower_target_tone == "concise":
                # Simulate concise tone: very basic truncation
                words = original_text.split()
                if len(words) > 8:
                    transformed_text = ' '.join(words[:8]) + '...'
                else:
                    transformed_text = original_text
                logger.debug("Text converted to concise tone.")

            else:
                logger.warning(
                    "Unsupported target_tone '%s' provided for ToneConverterNode. "
                    "Returning original data without transformation.",
                    target_tone
                )
                # If the tone is not recognized, return the original data.

        except Exception as e:
            logger.error(
                "An unexpected error occurred during tone conversion with target_tone '%s': %s",
                target_tone, e, exc_info=True
            )
            raise RuntimeError(
                f"Failed to convert tone due to an internal processing error: {e}"
            ) from e

        return transformed_text