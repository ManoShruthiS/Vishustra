import logging
from typing import Any, Dict

# Assuming vishustra_core is available in the Python path for import
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ToneConverterNode(BaseNode):
    """
    A Vishustra processing node designed to convert the tone of input text.
    The conversion logic is simulated based on a 'target_tone' specified in the context.
    This node acts as a placeholder for more advanced LLM-driven tone transformation
    in a full Vishustra orchestration.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "ToneConverter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input text data to simulate a tone conversion based on the
        'target_tone' provided in the context.

        Expected `data`:
            A string representing the text content to be converted.

        Expected `context`:
            A dictionary that *must* contain:
            - 'target_tone' (str): The desired tone for the output text
                                   (e.g., "formal", "informal", "humorous", "concise", "academic").

        Returns:
            str: The text with the simulated tone applied.

        Raises:
            TypeError: If the input `data` is not a string.
            ValueError: If 'target_tone' is missing from the `context`, is empty, or is not a string.
        """
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected `str`, "
                f"but received `{type(data).__name__}`. Data: {data!r}"
            )
            raise TypeError(f"Input data for {self.node_name} must be a string.")

        # Validate and retrieve 'target_tone' from context
        target_tone_raw = context.get("target_tone")
        if not isinstance(target_tone_raw, str) or not target_tone_raw.strip():
            logger.error(
                f"[{self.node_name}] 'target_tone' key is missing, empty, or not a string "
                f"in the context. Received: {target_tone_raw!r}. Context: {context}"
            )
            raise ValueError(
                f"Context for {self.node_name} must contain a non-empty string "
                f"for the 'target_tone' parameter."
            )
        
        target_tone: str = target_tone_raw.strip().lower()
        original_text: str = data.strip()
        converted_text: str = original_text # Initialize with original text as a fallback

        # Simulate tone conversion based on recognized tones
        if target_tone == "formal":
            converted_text = f"Regarding the matter at hand, it is observed that: {original_text}."
            logger.info(f"[{self.node_name}] Converted text to 'formal' tone.")
        elif target_tone == "informal":
            converted_text = f"Hey there, so, {original_text} – ya know?"
            logger.info(f"[{self.node_name}] Converted text to 'informal' tone.")
        elif target_tone == "humorous":
            converted_text = f"{original_text} (just kidding, mostly!)"
            logger.info(f"[{self.node_name}] Converted text to 'humorous' tone.")
        elif target_tone == "concise":
            words = original_text.split()
            # Simulate conciseness by taking first 15 words or less
            converted_text = " ".join(words[:15])
            if len(words) > 15:
                converted_text += "..."
            logger.info(f"[{self.node_name}] Converted text to 'concise' tone.")
        elif target_tone == "academic":
            converted_text = f"It is hypothesized that, through empirical observation, {original_text}."
            logger.info(f"[{self.node_name}] Converted text to 'academic' tone.")
        else:
            logger.warning(
                f"[{self.node_name}] Unrecognized or unsupported 'target_tone': '{target_tone}'. "
                "Returning original data without modification."
            )
            # No change to converted_text if tone is not supported

        return converted_text