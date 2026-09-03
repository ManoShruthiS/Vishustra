import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ToneConverterNode(BaseNode):
    """
    A processing node designed to simulate the conversion of input text to a specified tone.
    This node expects a string `data` and a `target_tone` within the context dictionary.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this node.
        """
        return "ToneConverter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, attempting to convert its tone based on the
        'target_tone' specified in the context.

        Args:
            data (Any): The input data, expected to be a string representing text.
            context (Dict[str, Any]): A dictionary containing contextual information,
                                      expected to have a 'target_tone' key with a string value.

        Returns:
            Any: The processed text with the simulated tone applied. If conversion
                 is not possible or `target_tone` is missing/unsupported, the
                 original data or a modified version with a warning is returned.

        Raises:
            TypeError: If the input `data` is not a string or if `target_tone`
                       in context is not a string.
        """
        if not isinstance(data, str):
            logger.error(
                f"ToneConverterNode received non-string data. Expected a string for tone conversion, "
                f"but got type: {type(data).__name__}. Data: {data!r}"
            )
            raise TypeError(
                f"Invalid input data type for ToneConverterNode: Expected string, got {type(data).__name__}"
            )

        target_tone = context.get("target_tone")

        if target_tone is None:
            logger.warning(
                "ToneConverterNode context is missing the 'target_tone' key. "
                "Returning original data without modification."
            )
            return data

        if not isinstance(target_tone, str):
            logger.error(
                f"ToneConverterNode 'target_tone' in context must be a string. "
                f"Got type: {type(target_tone).__name__}. Context: {context}"
            )
            raise TypeError(
                f"Invalid 'target_tone' type in context: Expected string, got {type(target_tone).__name__}"
            )

        processed_text = data
        lower_target_tone = target_tone.lower()

        # Simulate tone conversion based on common tones
        if lower_target_tone == "formal":
            processed_text = f"FORMAL: {data}"
            logger.debug(f"ToneConverterNode converted text to formal tone. Original: '{data[:50]}...'")
        elif lower_target_tone == "casual":
            processed_text = f"CASUAL: {data}"
            logger.debug(f"ToneConverterNode converted text to casual tone. Original: '{data[:50]}...'")
        elif lower_target_tone == "humorous":
            processed_text = f"{data} (LOL)"
            logger.debug(f"ToneConverterNode converted text to humorous tone. Original: '{data[:50]}...'")
        elif lower_target_tone == "professional":
            processed_text = f"[Professional Note]: {data}"
            logger.debug(f"ToneConverterNode converted text to professional tone. Original: '{data[:50]}...'")
        else:
            logger.warning(
                f"ToneConverterNode encountered an unsupported 'target_tone': '{target_tone}'. "
                f"Returning original data without specific tone modification."
            )
            # For unsupported tones, we could optionally append a disclaimer or leave as is
            # For this simulation, we'll return original data for unknown tones
            pass # processed_text remains `data`

        return processed_text