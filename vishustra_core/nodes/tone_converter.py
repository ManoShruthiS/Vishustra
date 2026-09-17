
import logging
from typing import Any, Dict

# Assuming vishustra_core.nodes.base_node exists in the project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ToneConverter(BaseNode):
    """
    A processing node designed to simulate converting the tone of input text.

    This node expects the input 'data' to be a string. The 'context' dictionary
    must contain a 'target_tone' key specifying the desired output tone
    (e.g., "formal", "informal", "professional", "friendly", "sarcastic").

    If the 'target_tone' is not recognized, the original data is returned
    with a logged warning.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "ToneConverter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by attempting to simulate a tone conversion.

        Args:
            data: The input text, expected to be a string.
            context: A dictionary containing node-specific configuration.
                     Must contain a 'target_tone' key (str) specifying the
                     desired tone for conversion.

        Returns:
            The text with its simulated tone adjusted as per the 'target_tone',
            or the original data if the conversion cannot be performed (e.g.,
            unrecognized tone or invalid input type).

        Raises:
            TypeError: If the input 'data' is not a string.
            ValueError: If 'target_tone' is missing from the context or is not a string.
        """
        if not isinstance(data, str):
            error_msg = (
                f"Input data for '{self.node_name}' must be a string. "
                f"Received type: {type(data).__name__}."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

        if "target_tone" not in context or not isinstance(context["target_tone"], str):
            error_msg = (
                f"Context for '{self.node_name}' requires a string value for the 'target_tone' key. "
                f"Current context keys: {list(context.keys()) if context else 'None'}."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        target_tone = context["target_tone"].lower()
        processed_data = data

        # A dictionary mapping target tones to simple text transformation templates.
        # In a production system, this would typically involve an LLM call
        # or a sophisticated NLP library for actual tone conversion.
        tone_templates = {
            "formal": "Regarding the matter at hand, we must state: {text}. Please note this communication.",
            "informal": "Hey there! Just wanted to quickly share: {text}. Hope that helps!",
            "professional": "This serves to professionally communicate that: {text}. Further action will be taken as required.",
            "friendly": "Hi! Hope you're doing great. Just a friendly heads-up: {text} :)",
            "sarcastic": "Oh, how utterly fascinating! It's truly astonishing that: {text}. Simply brilliant.",
            "polite": "Could you kindly address this: \"{text}\". Thank you very much for your attention and understanding.",
            "respectful": "With due respect, I would like to bring this to your attention: \"{text}\". I would be grateful for your kind consideration.",
            "kind": "So sorry to bother you! Just gently noting that {text}. Thank you so much :)",
        }

        if target_tone in tone_templates:
            template = tone_templates[target_tone]
            try:
                processed_data = template.format(text=data)
                logger.info(f"Successfully simulated tone conversion to '{target_tone}' for data snippet: '{data[:75]}...'")
            except Exception as e:
                logger.error(
                    f"Failed to apply tone template for '{target_tone}' on data '{data[:75]}...'. "
                    f"Error: {e}. Returning original data."
                )
        else:
            logger.warning(
                f"Unrecognized target tone '{target_tone}' for '{self.node_name}'. "
                "Returning original data without modification."
            )

        return processed_data
