import logging
from typing import Any, Dict

# Assuming vishustra_core is a package accessible in the project
# For local development/testing, this might need adjustment, e.g., from .base_node import BaseNode
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ToneConverterNode(BaseNode):
    """
    A processing node that simulates the conversion of text data to a specified tone.
    This node expects a string as input data and a 'target_tone' in the context.
    """

    # A simple mapping for simulated tone transformations.
    # In a real-world scenario, this would involve LLM calls or sophisticated NLP.
    _tone_transformations = {
        "professional": lambda text: f"Regarding the matter at hand, it is imperative to convey: {text}.",
        "casual": lambda text: f"Hey there! Just wanted to share: {text}!",
        "formal": lambda text: f"It is with considerable deference that we present the following: {text}.",
        "humorous": lambda text: f"Get this, you won't believe it: {text} (just kidding... mostly!)",
        "sarcastic": lambda text: f"Oh, how absolutely unexpected that {text} (said no one ever).",
        "neutral": lambda text: text, # Default, no change
    }

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "ToneConverter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by converting its tone based on the 'target_tone'
        specified in the context.

        Args:
            data: The input data, expected to be a string (text).
            context: A dictionary containing operational parameters.
                     Expects 'target_tone' (str) to specify the desired output tone.

        Returns:
            A string with the converted tone, or the original data if conversion fails.

        Raises:
            TypeError: If the input 'data' is not a string.
            ValueError: If 'target_tone' is missing or unsupported in the context,
                        and no suitable fallback can be applied.
        """
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected string, got {type(data).__name__}."
            )
            raise TypeError(
                f"[{self.node_name}] Input 'data' must be a string, but received {type(data).__name__}."
            )

        target_tone = context.get("target_tone")
        if target_tone is None:
            logger.warning(
                f"[{self.node_name}] 'target_tone' not specified in context. "
                "Defaulting to 'neutral' tone."
            )
            target_tone = "neutral"
        else:
            target_tone = str(target_tone).lower() # Ensure it's a string and lowercased

        transformer_func = self._tone_transformations.get(target_tone)

        if transformer_func is None:
            logger.warning(
                f"[{self.node_name}] Unsupported target tone '{target_tone}' requested. "
                "Falling back to 'neutral' tone."
            )
            transformer_func = self._tone_transformations["neutral"]

        try:
            converted_data = transformer_func(data)
            logger.info(
                f"[{self.node_name}] Successfully converted text to '{target_tone}' tone."
            )
            return converted_data
        except Exception as e:
            logger.exception(
                f"[{self.node_name}] An unexpected error occurred during tone conversion to '{target_tone}': {e}"
            )
            # Depending on framework policy, might re-raise, return original, or a specific error object.
            # For robustness, returning original data as a fallback is often acceptable for non-critical failures.
            raise RuntimeError(
                f"[{self.node_name}] Failed to convert tone to '{target_tone}': {e}"
            ) from e
