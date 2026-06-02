import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ToneConverter(BaseNode):
    """
    A Vishustra processing node designed to adjust the linguistic tone of input text.

    This node simulates the conversion of textual content into a specified target tone,
    such as formal, informal, professional, or sarcastic. It expects input `data` to be a string
    and requires a 'target_tone' string within the `context` dictionary to guide the transformation.

    In a production environment, this node would typically integrate with advanced Natural Language
    Processing (NLP) models or LLMs to perform nuanced tone adjustments. For this demonstration,
    the conversion is simulated through predefined text manipulations.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "ToneConverter"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Processes the input `data` (text) to convert its tone based on the
        'target_tone' specified in the provided `context`.

        Args:
            data (Any): The input data, which must be a string representing the text
                        to be converted.
            context (Dict[str, Any]): A dictionary containing contextual information
                                      for the processing. It *must* include a
                                      'target_tone' key with a non-empty string value
                                      specifying the desired tone.

        Returns:
            str: The processed text with the simulated new tone applied. If an
                 unsupported tone is requested, the original text is returned
                 after logging a warning.

        Raises:
            TypeError: If the input `data` is not a string.
            ValueError: If 'target_tone' is missing from the context or is not
                        a valid non-empty string.
        """
        if not isinstance(data, str):
            logger.error("ToneConverter: Invalid input data type. Expected 'str', but received '%s'.", type(data).__name__)
            raise TypeError(f"ToneConverter expects string data, but got '{type(data).__name__}'.")

        target_tone = context.get("target_tone")
        if not isinstance(target_tone, str) or not target_tone.strip():
            logger.error("ToneConverter: Missing or invalid 'target_tone' in context. Expected a non-empty string, received: '%s'.", target_tone)
            raise ValueError("Context for ToneConverter must contain a non-empty string for 'target_tone'.")

        # Normalize the target tone for case-insensitive comparison and lookup
        normalized_tone = target_tone.lower().strip()

        # Simulated transformations for various tones.
        # This dictionary would be replaced by actual NLP/LLM calls in a real system.
        simulated_transformations = {
            "formal": "In a formal capacity, we communicate: {}",
            "informal": "Hey, just a quick heads-up: {}",
            "professional": "From a professional standpoint, it is advised: {}",
            "casual": "Just wanted to let you know, casually: {}",
            "humorous": "Here's a light-hearted perspective: {} (chuckle)",
            "sarcastic": "Oh, absolutely, that's just utterly brilliant: {} (insert obvious eye-roll)",
            "encouraging": "You're doing great! Remember: {}",
            "academic": "Academically speaking, the premise suggests: {}",
            "technical": "Per the technical specification, the procedure dictates: {}",
            "cautionary": "Please exercise caution: {}",
        }

        if normalized_tone not in simulated_transformations:
            logger.warning("ToneConverter: Unsupported target tone '%s' requested. Returning original text without modification.", target_tone)
            # In some scenarios, an error might be raised for unsupported tones.
            # For robustness, we default to returning the original data.
            return data

        transformed_text = simulated_transformations[normalized_tone].format(data)
        logger.info("ToneConverter: Successfully applied '%s' tone. Original text snippet: '%s...'",
                    target_tone, data[:70].replace('\n', ' '))
        return transformed_text