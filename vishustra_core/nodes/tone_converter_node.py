import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ToneConverterNode(BaseNode):
    """
    A processing node designed to simulate the conversion of text tone.

    This node processes input text data, attempting to adjust its tone
    based on a 'target_tone' specified within the execution context.
    It supports a set of predefined tone transformations.
    """

    @property
    def node_name(self) -> str:
        """Returns the programmatic name of this node."""
        return "ToneConverter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by simulating a tone conversion based on context.

        This method expects the 'data' to be a string representing the text
        to be converted. The 'context' dictionary must contain a 'target_tone'
        key, whose value is a string indicating the desired output tone
        (e.g., 'formal', 'casual', 'humorous').

        Args:
            data: The input text to be processed, expected as a string.
            context: A dictionary containing operational parameters,
                     expected to include 'target_tone' (str).

        Returns:
            The processed text as a string, with the simulated tone applied.

        Raises:
            TypeError: If the input 'data' is not a string, or if 'target_tone'
                       in the context is not a string.
            ValueError: If 'target_tone' is missing from the context.
        """
        if not isinstance(data, str):
            error_msg = (
                f"{self.node_name} requires string input for 'data', "
                f"but received type {type(data).__name__}."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

        try:
            target_tone = context.get('target_tone')
            if target_tone is None:
                error_msg = f"Context missing required 'target_tone' key for {self.node_name}."
                logger.error(error_msg)
                raise ValueError(error_msg)
            if not isinstance(target_tone, str):
                error_msg = (
                    f"'target_tone' in context must be a string, "
                    f"but received type {type(target_tone).__name__}."
                )
                logger.error(error_msg)
                raise TypeError(error_msg)

            processed_text = str(data)  # Defensive conversion to string
            normalized_tone = target_tone.lower()

            if normalized_tone == 'formal':
                processed_text = f"Regarding the matter at hand: {processed_text}"
                logger.info(f"Applied 'formal' tone conversion to text using {self.node_name}.")
            elif normalized_tone == 'casual':
                processed_text = f"Hey, just wanted to say: {processed_text}"
                logger.info(f"Applied 'casual' tone conversion to text using {self.node_name}.")
            elif normalized_tone == 'humorous':
                processed_text = f"{processed_text} (just kidding, mostly!)"
                logger.info(f"Applied 'humorous' tone conversion to text using {self.node_name}.")
            else:
                logger.warning(
                    f"Unsupported target tone '{target_tone}' encountered in {self.node_name}. "
                    "Returning original text without modification."
                )

            logger.debug(
                f"{self.node_name} successfully processed data. "
                f"Original length: {len(data)}, Output length: {len(processed_text)}."
            )
            return processed_text

        except (TypeError, ValueError) as e:
            # Re-raise specific TypeErrors/ValueErrors already handled and logged.
            logger.error(f"Parameter validation error in {self.node_name}: {e}")
            raise
        except Exception as e:
            # Catch any unexpected errors during processing
            logger.exception(f"An unexpected error occurred during {self.node_name} processing: {e}")
            raise
