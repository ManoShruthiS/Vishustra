import logging
from typing import Any, Dict

# Assuming vishustra_core is available in the Python path
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ToneConverterNode(BaseNode):
    """
    A processing node designed to simulate converting the tone of input text.

    This node accepts a string as input data and relies on a 'target_tone'
    specified in the context dictionary to apply a simulated tone transformation.
    It's intended to serve as a placeholder or a basic utility within a larger
    orchestration where more sophisticated LLM-driven tone conversion might
    eventually be integrated.
    """

    @property
    def node_name(self) -> str:
        """Returns the programmatic name of the node."""
        return "ToneConverter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to simulate a change in textual tone.

        Expected `data`: A string representing the text content to be
                         tone-converted.
        Expected `context`: A dictionary that *must* contain the key
                            'target_tone' (str). This value dictates the
                            desired tone for the output text.
                            Example: `{"target_tone": "formal"}`

        Currently supported simulated tones:
        - "formal": Prepends "Indeed, " to the input text.
        - "informal": Appends ", lol" to the input text.
        - "concise": Truncates the text to the first 50 characters,
                     appending "..." if the original text was longer.

        The method handles various error conditions gracefully, such as
        incorrect data types or missing context parameters, by logging
        warnings and returning the original, unprocessed data.

        Returns the tone-converted text string (`str`) if successful,
        or the original `data` (`Any`) if conversion could not be performed
        due to errors or unsupported tones.
        """
        if not isinstance(data, str):
            logger.warning(
                f"[{self.node_name}] Invalid input data type. Expected 'str', "
                f"but received '{type(data).__name__}'. Returning original data."
            )
            return data

        target_tone = context.get("target_tone")
        if not isinstance(target_tone, str) or not target_tone.strip():
            logger.warning(
                f"[{self.node_name}] 'target_tone' not found or is an invalid "
                f"type/empty in context. Expected a non-empty string. "
                f"Returning original data."
            )
            return data

        processed_data: str = data
        # Normalize target_tone for case-insensitive matching
        lower_tone = target_tone.lower().strip()

        try:
            if lower_tone == "formal":
                processed_data = f"Indeed, {data}"
                logger.debug(f"[{self.node_name}] Applied 'formal' tone transformation.")
            elif lower_tone == "informal":
                processed_data = f"{data}, lol"
                logger.debug(f"[{self.node_name}] Applied 'informal' tone transformation.")
            elif lower_tone == "concise":
                if len(data) > 50:
                    # Truncate and add ellipsis, stripping potential leading/trailing spaces from truncated part
                    processed_data = data[:50].strip() + "..."
                    logger.debug(f"[{self.node_name}] Applied 'concise' tone transformation (text truncated).")
                else:
                    logger.debug(
                        f"[{self.node_name}] Text already concise or short enough "
                        f"for 'concise' tone; no truncation applied."
                    )
            else:
                logger.info(
                    f"[{self.node_name}] Unsupported target tone specified: '{target_tone}'. "
                    f"Returning original data without modification."
                )
                return data
        except Exception as e:
            # Catch any unexpected errors during the string manipulation
            logger.error(
                f"[{self.node_name}] An unexpected error occurred during tone "
                f"conversion for tone '{target_tone}': {e}. Returning original data."
            )
            return data

        return processed_data