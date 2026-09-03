import logging
from typing import Any, Dict

# Assuming this path exists in the Vishustra project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class ToneConverterNode(BaseNode):
    """
    A Vishustra processing node designed to simulate the conversion of text tone.

    This node takes an input string and a 'target_tone' specified in the context,
    then attempts to return a string reflecting the desired tone.
    It includes robust validation for input data and context parameters.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "ToneConverter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, transforming its tone based on the 'target_tone'
        provided in the context dictionary.

        Args:
            data (Any): The input content to be tone-converted. Expected to be a string.
            context (Dict[str, Any]): A dictionary containing processing parameters.
                                       Must include the 'target_tone' (str) key,
                                       specifying the desired output tone.

        Returns:
            Any: A new string with the simulated tone applied.

        Raises:
            TypeError: If the input `data` is not a string, or if 'target_tone'
                       in context is not a string.
            KeyError: If the 'target_tone' key is missing from the `context`.
            ValueError: If the specified 'target_tone' is not recognized or supported.
        """
        logger.info("ToneConverterNode: Initiating text tone conversion process.")

        if not isinstance(data, str):
            logger.error(f"ToneConverterNode: Invalid input data type. Expected 'str', but received '{type(data).__name__}'.")
            raise TypeError(f"Input data for ToneConverterNode must be a string, but received '{type(data).__name__}'.")

        try:
            target_tone_raw = context.get('target_tone')
            if target_tone_raw is None:
                raise KeyError("'target_tone' key is missing")
            if not isinstance(target_tone_raw, str):
                logger.error(f"ToneConverterNode: 'target_tone' in context must be a string, but received '{type(target_tone_raw).__name__}'.")
                raise TypeError(f"Context parameter 'target_tone' must be a string, but received '{type(target_tone_raw).__name__}'.")
            target_tone = target_tone_raw.lower()
        except KeyError as e:
            logger.error(f"ToneConverterNode: Required context key missing: {e}. 'target_tone' is essential for this node.")
            raise KeyError(f"Context missing required key for ToneConverterNode: {e}. Please provide a 'target_tone'.")
        except TypeError as e:
            raise e # Re-raise TypeErrors specific to context params
        except Exception as e:
            logger.exception(f"ToneConverterNode: An unexpected error occurred while retrieving 'target_tone' from context.")
            raise RuntimeError(f"Failed to process 'target_tone' from context: {e}")

        converted_text = str(data)  # Ensure we're working with a mutable copy if needed, though str is immutable
        original_text_snippet = data[:70] + "..." if len(data) > 70 else data

        # Simulate tone conversion based on target_tone
        if target_tone == 'formal':
            converted_text = converted_text.replace("don't", "do not").replace("can't", "cannot")
            converted_text = "It is requested that " + converted_text.replace("hi", "greetings").replace("hello", "greetings")
            logger.debug(f"ToneConverterNode: Applying 'formal' tone transformation to text starting with: '{original_text_snippet}'.")
        elif target_tone == 'casual':
            converted_text = converted_text.replace("do not", "don't").replace("cannot", "can't")
            converted_text = "Hey there! " + converted_text.replace("greetings", "hi").replace("it is requested that", "please")
            logger.debug(f"ToneConverterNode: Applying 'casual' tone transformation to text starting with: '{original_text_snippet}'.")
        elif target_tone == 'sarcastic':
            converted_text = f"Oh, how truly fascinating. ({converted_text}) I'm *certain* this will prove to be an invaluable contribution."
            logger.debug(f"ToneConverterNode: Applying 'sarcastic' tone transformation to text starting with: '{original_text_snippet}'.")
        elif target_tone == 'professional':
            converted_text = "Regarding the matter: " + converted_text.replace("hi", "hello").replace("hey", "hello").replace("greetings", "hello")
            converted_text = converted_text.replace("just ", "").replace("maybe", "potentially").replace("I think", "It is my understanding that")
            logger.debug(f"ToneConverterNode: Applying 'professional' tone transformation to text starting with: '{original_text_snippet}'.")
        else:
            logger.warning(f"ToneConverterNode: Unrecognized 'target_tone' specified: '{target_tone}'. Supported tones include 'formal', 'casual', 'sarcastic', 'professional'.")
            raise ValueError(f"Unsupported 'target_tone': '{target_tone}'. Please choose from 'formal', 'casual', 'sarcastic', 'professional'.")

        logger.info(f"ToneConverterNode: Successfully converted tone for text (original snippet: '{original_text_snippet}') to '{target_tone}'.")
        return converted_text