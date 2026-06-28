import logging
from typing import Any, Dict, Literal

# Import the base class from the Vishustra core nodes
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

# Define allowed tones for robustness and type hinting
ToneType = Literal['professional', 'friendly', 'formal', 'casual', 'sarcastic', 'neutral']

class ToneConverterNode(BaseNode):
    """
    A Vishustra processing node responsible for converting the tone of input text.

    The node can be initialized with a default target tone, which can then be
    overridden by specific requests in the `context` dictionary during processing.
    """

    def __init__(self, target_tone: ToneType = 'neutral'):
        """
        Initializes the ToneConverterNode with a default target tone.

        Args:
            target_tone (ToneType): The default tone to convert the text to.
                                    Must be one of 'professional', 'friendly', 'formal',
                                    'casual', 'sarcastic', or 'neutral'.
        """
        if not isinstance(target_tone, str) or target_tone not in ToneType.__args__:
            logger.warning(
                f"Invalid target_tone '{target_tone}' provided during initialization. "
                "Defaulting to 'neutral'."
            )
            self._target_tone: ToneType = 'neutral'
        else:
            self._target_tone = target_tone
        logger.debug(f"ToneConverterNode initialized with default target_tone: '{self._target_tone}'")

    @property
    def node_name(self) -> str:
        """Returns the name of the node, indicating its default tone configuration."""
        return f"ToneConverter:{self._target_tone.capitalize()}"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data (expected to be a string) and converts its tone.

        The tone can be specified in the `context` dictionary under the key
        'tone_converter_target_tone'. If not provided or invalid, the node's
        default `target_tone` is used.

        Args:
            data (Any): The input data, expected to be a string.
            context (Dict[str, Any]): A dictionary containing additional information
                                      or configuration for the processing step.
                                      Can include 'tone_converter_target_tone'
                                      to override the default.

        Returns:
            Any: The tone-converted string.

        Raises:
            TypeError: If the input `data` is not a string.
            RuntimeError: If an unexpected error occurs during tone conversion.
        """
        if not isinstance(data, str):
            logger.error(
                f"ToneConverterNode received non-string data. "
                f"Expected str, but got {type(data).__name__}."
            )
            raise TypeError("ToneConverterNode expects string data for tone conversion.")

        original_text: str = data
        effective_target_tone: ToneType = self._target_tone

        # Attempt to override the target tone from context
        context_tone = context.get('tone_converter_target_tone')
        if context_tone and isinstance(context_tone, str) and context_tone in ToneType.__args__:
            effective_target_tone = context_tone
            logger.debug(
                f"Context overridden tone: '{context_tone}'. "
                f"Node's default tone: '{self._target_tone}' is being overridden."
            )
        elif context_tone is not None:
            logger.warning(
                f"Invalid or unsupported 'tone_converter_target_tone' '{context_tone}' found in context. "
                f"Falling back to node's default tone: '{self._target_tone}'."
            )

        converted_text: str = original_text

        try:
            logger.info(
                f"Attempting to convert text tone to '{effective_target_tone}' for "
                f"data (first 50 chars: '{original_text[:50].strip()}...')."
            )

            # Simulate tone conversion. In a production environment, this would
            # involve an actual LLM call or sophisticated NLP libraries.
            # This simulation uses simple string replacements and prefixes.
            if effective_target_tone == 'professional':
                converted_text = (
                    f"[PROFESSIONAL] {original_text.replace('hey', 'Dear').replace('guys', 'colleagues')}"
                    f". Please find the details below. Regards."
                )
            elif effective_target_tone == 'friendly':
                converted_text = (
                    f"[FRIENDLY] Hi there! {original_text.replace('Regards', 'Best regards').replace('Sincerely', 'Cheers')}"
                    f". Hope you have a great day! :)"
                )
            elif effective_target_tone == 'formal':
                converted_text = (
                    f"[FORMAL] To Whom It May Concern: {original_text.replace('hey', 'Dear Sir/Madam')}"
                    f". We require your immediate attention to this matter. Sincerely."
                )
            elif effective_target_tone == 'casual':
                converted_text = (
                    f"[CASUAL] Yo! {original_text.replace('Regards', 'Later').replace('Sincerely', 'Talk soon')}"
                    f". Catch you on the flip side!"
                )
            elif effective_target_tone == 'sarcastic':
                converted_text = (
                    f"[SARCASTIC] Oh, how utterly fascinating. {original_text}"
                    f" (Because we all know how much you truly meant that.)"
                )
            elif effective_target_tone == 'neutral':
                converted_text = f"[NEUTRAL] {original_text.strip()}"
            else:
                # Should not be reached if effective_target_tone is validated
                logger.warning(
                    f"Unhandled effective_target_tone '{effective_target_tone}'. "
                    "Returning original text without conversion."
                )
                converted_text = original_text

            logger.debug(
                f"Tone conversion complete. Original (first 50): '{original_text[:50].strip()}', "
                f"Converted (first 50): '{converted_text[:50].strip()}'"
            )
            return converted_text

        except Exception as e:
            logger.exception(
                f"An unexpected error occurred during tone conversion for data "
                f"(first 50 chars: '{original_text[:50].strip()}...')."
            )
            raise RuntimeError(
                f"Failed to process data through ToneConverterNode due to an internal error: {e}"
            ) from e
