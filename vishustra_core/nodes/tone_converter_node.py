import logging
from typing import Any, Dict

# Assuming vishustra_core.nodes.base_node exists as per project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ToneConverterNode(BaseNode):
    """
    A Vishustra processing node that simulates converting the tone of input text.

    This node expects a string as input data and a 'target_tone' string
    within the context dictionary. It appends a tone tag to the input text
    to simulate the tone conversion, providing a foundational step for
    LLM-based tone adjustments.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "ToneConverter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to simulate tone conversion.

        This method expects `data` to be a string and looks for a `target_tone`
        string in the `context` dictionary. It simulates the tone conversion
        by appending a standardized tone tag to the input text.

        Args:
            data: The input data, expected to be a string containing text
                  that needs its tone converted.
            context: A dictionary containing contextual information relevant
                     to the processing. This must include a 'target_tone' key
                     whose value is a string specifying the desired tone (e.g., "formal", "casual").

        Returns:
            A string representing the transformed data with the simulated tone tag.
            If the input `data` is not a string, or if `target_tone` is missing
            or invalid in the `context`, the original `data` is returned,
            and an appropriate warning or error is logged.
        """
        logger.debug(f"[{self.node_name}] Starting process. Data type: {type(data)}, Context keys: {list(context.keys()) if context else 'None'}")

        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected 'str', but received '{type(data).__name__}'. "
                "Returning original data without modification."
            )
            return data

        target_tone = context.get("target_tone")

        if not isinstance(target_tone, str) or not target_tone.strip():
            logger.warning(
                f"[{self.node_name}] 'target_tone' not found or is not a valid non-empty string in the context. "
                "Returning original data without tone conversion."
            )
            return data

        # Simulate tone conversion by appending a tag.
        # In a real scenario, this would involve an LLM call or a sophisticated NLP module.
        cleaned_data = data.strip()
        cleaned_target_tone = target_tone.strip()
        converted_data = f"{cleaned_data} [TONE: {cleaned_target_tone.capitalize()}]"

        logger.info(
            f"[{self.node_name}] Successfully simulated tone conversion to '{cleaned_target_tone}'. "
            f"Original text length: {len(data)}, Transformed text length: {len(converted_data)}"
        )
        return converted_data