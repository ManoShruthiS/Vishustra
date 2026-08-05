import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ToneConverterNode(BaseNode):
    """
    A Vishustra processing node that simulates converting the tone of a given text.

    This node expects the input 'data' to be a string and the 'context'
    to contain a 'target_tone' string, which dictates the desired
    output tone. For simulation purposes, it prefixes the input text
    based on the specified tone.

    Supported tones include: 'formal', 'informal', 'sarcastic', 'friendly', 'professional'.
    If an unsupported tone is provided, the original data is returned with a warning.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "ToneConverter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input text to simulate converting its tone based on the
        'target_tone' specified in the context.

        Args:
            data (Any): The input data, which is expected to be a string
                        (the text whose tone needs to be converted).
            context (Dict[str, Any]): A dictionary containing processing context.
                                      It MUST include a 'target_tone' key
                                      whose value is a string.

        Returns:
            Any: The tone-converted text as a string. If the 'target_tone'
                 is unsupported or an error occurs during simulated conversion,
                 the original data or a generically modified version is returned.

        Raises:
            TypeError: If the input 'data' is not a string.
            ValueError: If 'target_tone' is missing from the 'context' or is not a string.
        """
        logger.debug(f"[{self.node_name}] Initiating processing for input data.")

        if not isinstance(data, str):
            error_msg = (
                f"[{self.node_name}] Invalid input data type. Expected 'str', "
                f"but received '{type(data).__name__}'. Data: {data!r}"
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

        if 'target_tone' not in context or not isinstance(context['target_tone'], str):
            error_msg = (
                f"[{self.node_name}] Missing or invalid 'target_tone' in context. "
                f"Context must contain a string value for 'target_tone'. Context: {context}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        target_tone = context['target_tone'].lower()
        processed_text = data

        # In a real LLM scenario, this would involve a call to an LLM
        # with a prompt structured for tone conversion.
        # For this simulation, we use simple string prefixing.
        tone_modifiers = {
            "formal": "The following statement conveys a formal message: ",
            "informal": "Yo, check this out: ",
            "sarcastic": "Oh, how *utterly* fascinating, truly: ",
            "friendly": "Hey there! Just wanted to share: ",
            "professional": "From a strictly professional standpoint, one might say: ",
        }

        if target_tone in tone_modifiers:
            processed_text = tone_modifiers[target_tone] + data
            logger.info(
                f"[{self.node_name}] Successfully simulated tone conversion to "
                f"'{target_tone}' for data snippet: '{data[:75]}...'"
            )
        else:
            logger.warning(
                f"[{self.node_name}] Unsupported 'target_tone' '{target_tone}'. "
                f"Returning original data. Supported tones are: {', '.join(tone_modifiers.keys())}"
            )
            # When an unsupported tone is requested, we opt to return the original text
            # to prevent pipeline failure and allow downstream nodes to potentially handle.
            processed_text = data

        logger.debug(f"[{self.node_name}] Finished processing. Returning modified text.")
        return processed_text