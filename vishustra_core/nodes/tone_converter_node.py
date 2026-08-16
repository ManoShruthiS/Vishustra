import logging
from typing import Any, Dict

# Assuming vishustra_core.nodes.base_node is available in the Python path
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ToneConverterNode(BaseNode):
    """
    A Vishustra node designed to simulate the conversion of an input text's tone.

    This node accepts a string as input data and expects a 'target_tone' key
    within the provided context dictionary. It then applies a set of predefined
    transformations to simulate the desired tonal adjustment. This node is
    primarily for demonstration and integration purposes within the Vishustra
    framework.

    Supported simulated tones and their effects:
    - 'formal': Attempts to capitalize sentence beginnings and introduce formal phrasing.
    - 'casual': Transforms text to lowercase beginnings and adds informal expressions.
    - 'enthusiastic': Incorporates exclamation marks and positive, upbeat language.
    - 'concise': Simulates reduction of filler words and potential sentence shortening.
    - Other tones: Default to a neutral processing, ensuring basic sentence structure.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the unique name of this node.
        """
        return "ToneConverter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, applying a simulated tone conversion based on
        the 'target_tone' specified in the context.

        Args:
            data: The input text to be tone-converted. Expected to be a string.
            context: A dictionary containing runtime context variables.
                     It *must* include a 'target_tone' (str) indicating the
                     desired tone for conversion.

        Returns:
            The processed text string with its tone adjusted (simulated).

        Raises:
            ValueError: If 'data' is not a string, or if 'target_tone' is
                        missing, not a string, or empty in the context.
        """
        if not isinstance(data, str):
            logger.error(
                "ToneConverterNode received invalid data type. Expected string, got '%s'.",
                type(data).__name__
            )
            raise ValueError(f"ToneConverterNode requires string data, received {type(data).__name__}.")

        target_tone = context.get("target_tone")
        if not isinstance(target_tone, str) or not target_tone.strip():
            logger.error(
                "ToneConverterNode 'target_tone' is missing or invalid in context. "
                "Expected non-empty string, got '%s'.", target_tone
            )
            raise ValueError(
                "Context must contain a valid 'target_tone' (non-empty string) for ToneConverterNode."
            )

        original_text = data.strip()
        converted_text = original_text
        tone_key = target_tone.lower().strip()

        logger.info("Initiating tone conversion of text (length %d) to '%s'.", len(original_text), tone_key)

        # --- Simulated Tone Conversion Logic ---
        if tone_key == "formal":
            sentences = [s.strip() for s in original_text.split('.') if s.strip()]
            processed_sentences = []
            for s in sentences:
                if s:
                    processed_sentences.append(s[0].upper() + s[1:])
            converted_text = ". ".join(processed_sentences)
            if converted_text and not converted_text.endswith('.'):
                converted_text += '.'
            converted_text = f"Regarding the matter at hand, it is imperative to note: {converted_text}"
            logger.debug("Applied 'formal' tone conversion.")

        elif tone_key == "casual":
            converted_text = original_text.lower().replace("regarding", "about").replace("furthermore", "also")
            converted_text = f"Hey there, just wanted to share: {converted_text} No biggie."
            if converted_text.endswith('.'):
                converted_text = converted_text[:-1] + '!'
            logger.debug("Applied 'casual' tone conversion.")

        elif tone_key == "enthusiastic":
            converted_text = original_text.replace(".", "!!!").replace("!", "!!!").replace("?", "!!!")
            converted_text = f"Absolutely fantastic news! {converted_text} This is truly amazing!!!"
            logger.debug("Applied 'enthusiastic' tone conversion.")

        elif tone_key == "concise":
            words = original_text.split()
            # Simulate removal of common filler/linking words
            filler_words = {"the", "a", "an", "is", "was", "be", "to", "and", "but", "so", "however", "therefore", "in order to"}
            filtered_words = [word for word in words if word.lower() not in filler_words]
            converted_text = " ".join(filtered_words)
            # Basic sentence shortening if original was long
            if len(converted_text) > 120 and '.' in converted_text:
                converted_text = converted_text.split('.')[0] + '.'
            converted_text = f"In brief: {converted_text}"
            logger.debug("Applied 'concise' tone conversion.")

        else:
            logger.warning(
                "Unsupported or unrecognized target tone '%s'. Applying default neutral adjustment.",
                target_tone
            )
            # Default adjustment: ensure basic sentence casing and punctuation.
            sentences = [s.strip() for s in original_text.split('.') if s.strip()]
            processed_sentences = []
            for s in sentences:
                if s:
                    processed_sentences.append(s[0].upper() + s[1:])
            converted_text = ". ".join(processed_sentences)
            if converted_text and not converted_text.endswith('.'):
                converted_text += '.'
            converted_text = f"Default adjusted text: {converted_text}"

        logger.info(
            "Successfully completed tone conversion to '%s'. Resulting text length: %d.",
            tone_key, len(converted_text)
        )
        return converted_text