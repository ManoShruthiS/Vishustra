
import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ToneConverterNode(BaseNode):
    """
    A processing node that simulates converting the tone of an input text
    based on a specified 'target_tone' in the context.

    This node is designed to demonstrate dynamic text manipulation within
    the Vishustra framework, allowing for flexible content adaptation.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "ToneConverter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Converts the tone of the input data (expected to be a string)
        to the 'target_tone' specified in the context.

        Args:
            data (Any): The input data, expected to be a string representing text.
            context (Dict[str, Any]): A dictionary containing execution context.
                                      Must include 'target_tone' (str).

        Returns:
            Any: The tone-converted string.

        Raises:
            ValueError: If 'data' is not a string, or if 'target_tone' is missing
                        from the context, or if the 'target_tone' is unsupported.
        """
        if not isinstance(data, str):
            logger.error(
                "ToneConverterNode received invalid input data type. Expected string, got %s.",
                type(data).__name__
            )
            raise ValueError(f"Invalid input data type for ToneConverterNode: expected string, got {type(data).__name__}.")

        if "target_tone" not in context:
            logger.error("Context for ToneConverterNode is missing 'target_tone' parameter.")
            raise ValueError("Missing 'target_tone' in context for ToneConverterNode.")

        original_text = data.strip()
        target_tone = context["target_tone"].lower()
        converted_text = original_text

        try:
            # --- Simulated Tone Conversion Logic ---
            # In a real-world scenario, this would involve calling a sophisticated
            # NLP model or an external LLM for actual tone transformation.
            # Here, we use simple string manipulations to simulate the effect.
            if target_tone == "formal":
                if not original_text:
                    converted_text = ""
                else:
                    converted_text = original_text[0].upper() + original_text[1:]
                    if not converted_text.endswith(('.', '?', '!')):
                        converted_text += '.'
                logger.info("Text converted to formal tone using simulated logic.")
            elif target_tone == "informal":
                if not original_text:
                    converted_text = ""
                else:
                    converted_text = original_text.lower().replace('.', '').replace('!', '').replace('?', '')
                    # Add some informal flair
                    if converted_text and not converted_text.endswith('!'):
                        converted_text += '!!!'
                logger.info("Text converted to informal tone using simulated logic.")
            elif target_tone == "professional":
                if not original_text:
                    converted_text = ""
                else:
                    converted_text = original_text.replace("hey", "Dear").replace("hi", "Hello").replace("guys", "team")
                    converted_text = converted_text[0].upper() + converted_text[1:]
                    if not converted_text.endswith(('.', '?', '!')):
                        converted_text += '.'
                logger.info("Text converted to professional tone using simulated logic.")
            elif target_tone == "neutral":
                # For neutral, we might just strip extra whitespace or perform minor cleanup
                converted_text = original_text
                logger.info("Text maintained neutral tone (no significant change simulated).")
            else:
                logger.warning(
                    "Unsupported target_tone '%s' encountered in ToneConverterNode. Returning original data.",
                    target_tone
                )
                # Depending on system requirements, one might raise an error here
                # raise ValueError(f"Unsupported target tone: {target_tone}")
                converted_text = original_text # Default to returning original if tone is unrecognized
            # --- End Simulated Tone Conversion Logic ---

        except Exception as e:
            logger.error(
                "An unexpected error occurred during tone conversion for target_tone '%s': %s",
                target_tone, e, exc_info=True
            )
            # Re-raise or return original data based on error handling policy
            raise RuntimeError(f"Failed to process tone conversion for '{target_tone}': {e}") from e

        return converted_text
