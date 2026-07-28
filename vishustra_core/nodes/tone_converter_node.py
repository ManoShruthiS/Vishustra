import logging
from typing import Any, Dict
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ToneConverterNode(BaseNode):
    """
    A Vishustra processing node designed to convert the tone of input text
    based on a specified target tone provided in the context.

    This node simulates the functionality of an underlying language model
    to achieve stylistic text transformations.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "ToneConverter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data (text) to convert its tone to the target tone
        specified in the context.

        Expected `data`: A string representing the text content to be converted.
        Expected `context`: A dictionary that *must* contain a 'target_tone' key
                           whose value is a string (e.g., 'formal', 'casual',
                           'sarcastic', 'professional', 'friendly').

        Raises:
            ValueError: If `data` is not a string, or if 'target_tone' is
                        missing from the context or is not a string.
            Exception: For any unforeseen errors that occur during the
                       conversion process.
        """
        logger.info(f"[{self.node_name}] Initiating tone conversion process.")

        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected 'str', "
                f"received '{type(data).__name__}'."
            )
            raise ValueError(
                f"Input data for '{self.node_name}' must be a string. "
                f"Received type: {type(data).__name__}."
            )

        try:
            target_tone = context.get('target_tone')
            if not isinstance(target_tone, str):
                logger.error(
                    f"[{self.node_name}] 'target_tone' is missing or invalid in context. "
                    f"Expected 'str', received '{type(target_tone).__name__ if target_tone is not None else 'None'}'."
                )
                raise ValueError(
                    f"Context for '{self.node_name}' must contain a 'target_tone' key "
                    f"with a string value. Received: {target_tone}."
                )

            logger.debug(f"[{self.node_name}] Attempting to convert text to '{target_tone}' tone.")

            # Simulate the tone conversion. In a production environment, this
            # would typically involve an API call to a sophisticated language model.
            converted_text = self._simulate_tone_conversion(data, target_tone.lower())

            logger.info(f"[{self.node_name}] Tone conversion completed successfully to '{target_tone}' tone.")
            return converted_text
        except ValueError as ve:
            logger.error(f"[{self.node_name}] Configuration or input validation error: {ve}")
            raise
        except Exception as e:
            logger.exception(f"[{self.node_name}] An unexpected error occurred during processing.")
            raise

    def _simulate_tone_conversion(self, text: str, tone: str) -> str:
        """
        Internal method to simulate the application of a specific tone to the text.
        This provides a tangible demonstration of functionality without requiring
        an actual LLM integration.
        """
        # Basic examples for demonstration. A real implementation would be
        # significantly more nuanced via an LLM.
        if tone == 'formal':
            return f"Indeed, it is imperative to acknowledge that {text.lower()}."
        elif tone == 'casual':
            return f"Hey, just wanted to share that {text.lower()}."
        elif tone == 'sarcastic':
            return f"Oh, absolutely, because {text.lower()}... obviously."
        elif tone == 'professional':
            return f"Regarding the matter of {text}, our analysis indicates the following:"
        elif tone == 'friendly':
            return f"Hi there! Just a friendly note about {text}!"
        else:
            logger.warning(
                f"[{self.node_name}] Unsupported target tone '{tone}' specified. "
                f"The original text will be returned without modification."
            )
            return text