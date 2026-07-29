import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode # Import BaseNode as per requirements

# Setup logger for this module
logger = logging.getLogger(__name__)

class ToneConverter(BaseNode):
    """
    A Vishustra node that converts the tone of an input text.

    This node is designed to simulate a tone transformation on textual data.
    It allows specifying a default target tone during initialization, which
    can subsequently be overridden by a 'target_tone' entry within the
    `context` dictionary during the `process` method call.

    In a production environment, the tone conversion logic within the `process`
    method would typically integrate with advanced NLP models or external
    APIs to perform sophisticated linguistic transformations. For this
    demonstration, a simplified approach of appending a tone-modifying
    phrase is used.
    """

    # Internal mapping of supported tones to a descriptive phrase simulating
    # the outcome of a tone conversion.
    _TONE_TRANSFORMATIONS: Dict[str, str] = {
        "professional": "Ensuring optimal clarity and adherence to best practices, this content has been refined to reflect a professional standard.",
        "casual": "Just chilling here, making sure this message sounds super laid-back and easygoing!",
        "formal": "With due diligence and adherence to established protocols, the preceding discourse has been meticulously adjusted to embody a distinctly formal register.",
        "humorous": "Ha! Bet you didn't see *that* coming, did you? We've sprinkled some giggles into this text for maximum chuckle-factor!",
        "optimistic": "Approaching this with boundless enthusiasm, the message has been imbued with a forward-looking and positive outlook!",
        "pessimistic": "While striving for accuracy, one cannot ignore the inherent challenges and potential pitfalls, thus the message reflects a cautious and perhaps somber perspective.",
    }

    def __init__(self, default_target_tone: str = "professional"):
        """
        Initializes the ToneConverter node with a default target tone.

        Args:
            default_target_tone (str): The default tone to convert to if
                                       'target_tone' is not provided in the
                                       processing context. This value must
                                       be one of the supported tones defined
                                       within the node.

        Raises:
            ValueError: If the provided `default_target_tone` is not supported.
        """
        if default_target_tone not in self._TONE_TRANSFORMATIONS:
            raise ValueError(
                f"Unsupported default_target_tone: '{default_target_tone}'. "
                f"Supported tones are: {list(self._TONE_TRANSFORMATIONS.keys())}"
            )
        self._default_target_tone = default_target_tone
        logger.debug(f"ToneConverter initialized with default_target_tone: '{self._default_target_tone}'")

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node.
        """
        return "ToneConverter"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Processes the input data by converting its tone to a specified target.

        The method expects the input `data` to be a string. The desired
        `target_tone` can be passed via the `context` dictionary under the key
        'target_tone'. If 'target_tone' is not present in `context`, the
        node's `default_target_tone` (set during initialization) will be used.

        Args:
            data (Any): The input data to be processed. Expected to be a string
                        containing the text whose tone needs conversion.
            context (Dict[str, Any]): A dictionary containing runtime parameters.
                                      May include 'target_tone' (str) to override
                                      the default.

        Returns:
            str: The processed text, with a simulated tone adjustment applied.

        Raises:
            TypeError: If the input `data` is not a string.
            ValueError: If the specified `target_tone` (either from `context`
                        or the default) is not one of the supported tones.
        """
        if not isinstance(data, str):
            logger.error(
                f"Invalid input data type for ToneConverter. Expected 'str', "
                f"but received '{type(data).__name__}'. Data: '{str(data)[:100]}...'"
            )
            raise TypeError(
                f"ToneConverter expects string data, but received {type(data).__name__}. "
                "Ensure previous nodes provide string output."
            )

        # Determine the target tone, preferring context over instance default
        target_tone = context.get('target_tone', self._default_target_tone)
        logger.debug(f"ToneConverter processing with effective target_tone: '{target_tone}'")

        if target_tone not in self._TONE_TRANSFORMATIONS:
            logger.error(
                f"Unsupported target_tone '{target_tone}' specified. "
                f"Supported tones are: {list(self._TONE_TRANSFORMATIONS.keys())}"
            )
            raise ValueError(
                f"Unsupported target_tone: '{target_tone}'. "
                f"Supported tones are: {list(self._TONE_TRANSFORMATIONS.keys())}"
            )

        original_text = data.strip()
        tone_modifier_phrase = self._TONE_TRANSFORMATIONS[target_tone]

        # Simulate tone conversion by appending a modifier phrase.
        # In a real-world Vishustra pipeline, this logic would invoke
        # sophisticated NLP services or models.
        processed_text = f"{original_text} [Tone Adjustment: {tone_modifier_phrase}]"

        logger.info(
            f"ToneConverter successfully adjusted text tone to '{target_tone}'. "
            f"Original (first 50 chars): '{original_text[:50]}...', "
            f"Processed (first 50 chars): '{processed_text[:50]}...'"
        )
        return processed_text