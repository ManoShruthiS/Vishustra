import logging
from typing import Any, Dict

# Assuming this path structure based on the project context
# The base_node is expected to be in a subdirectory `nodes` within `vishustra_core`
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ToneConverterNode(BaseNode):
    """
    A Vishustra processing node that simulates converting the tone of an input text.

    This node takes a string as input and, based on the configured target tone,
    simulates a tone conversion. For demonstration, it prepends an indicator
    to the text. In a real scenario, this would involve NLP models or rules.

    Configuration Parameters (via constructor):
        - target_tone (str): The desired tone to convert the input text to
                             (e.g., "formal", "informal", "empathetic", "concise").
    """

    # Define a set of supported tones for validation
    _supported_tones = {"formal", "informal", "empathetic", "concise", "neutral", "professional"}

    def __init__(self, target_tone: str):
        """
        Initializes the ToneConverterNode with a specific target tone.

        Args:
            target_tone (str): The tone to convert the input text to.
                               Must be one of the supported tones.
        Raises:
            ValueError: If the target_tone is not recognized or invalid.
        """
        if not isinstance(target_tone, str) or target_tone.lower() not in self._supported_tones:
            valid_tones = ", ".join(sorted(list(self._supported_tones)))
            logger.error(
                f"Node initialization failed: Invalid or unsupported target_tone '{target_tone}'. "
                f"Supported tones are: {valid_tones}"
            )
            raise ValueError(
                f"Invalid or unsupported target_tone '{target_tone}'. "
                f"Supported tones are: {valid_tones}"
            )
        self._target_tone = target_tone.lower()
        logger.debug(f"ToneConverterNode initialized successfully with target_tone: '{self._target_tone}'")

    @property
    def node_name(self) -> str:
        """
        Returns the dynamic name of the node, including its configured target tone.
        This helps distinguish different instances of ToneConverterNode in a pipeline.
        """
        return f"ToneConverter:{self._target_tone.capitalize()}"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, attempting to convert its tone.

        If the input `data` is a string, it simulates converting its tone
        to the `_target_tone` specified during initialization. If `data`
        is not a string, a TypeError is raised as this node expects text.

        Args:
            data (Any): The input data to be processed, *expected to be a string*.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the current processing flow. This can be
                                       used for pipeline-specific data or shared state.

        Returns:
            Any: The processed data, which is a string with a tone indicator.

        Raises:
            TypeError: If the input `data` is not a string, as this node is designed
                       to operate exclusively on text data.
            RuntimeError: If an unexpected internal error occurs during the tone
                          conversion process, indicating a critical failure.
        """
        if not isinstance(data, str):
            error_msg = (
                f"{self.node_name}: Invalid input data type. Expected a string for tone conversion, "
                f"but received type '{type(data).__name__}'."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

        try:
            # Simulate tone conversion by prepending an indicator to the text.
            # In a production system, this section would integrate with advanced
            # NLP models (e.g., fine-tuned LLMs, style transfer models) or
            # a sophisticated rule-based engine to perform actual tone transformation.
            converted_text = f"[{self._target_tone.upper()} TONE] {data}"
            logger.info(
                f"{self.node_name}: Successfully simulated tone conversion to '{self._target_tone}'. "
                f"Original data (first 50 chars): '{data[:50]}...' -> Converted data (first 50 chars): '{converted_text[:50]}...'"
            )
            return converted_text
        except Exception as e:
            # Catching general Exception for any unforeseen runtime issues during the core logic.
            # Using logger.exception() automatically includes stack trace information.
            logger.exception(
                f"{self.node_name}: An unexpected error occurred during tone conversion of data "
                f"(first 50 chars: '{data[:50]}...')."
            )
            # Re-raise as a RuntimeError to signify a critical failure in the node's processing,
            # allowing upstream pipeline error handling to catch it.
            raise RuntimeError(
                f"Failed to process data in {self.node_name} due to an internal error: {e}"
            ) from e