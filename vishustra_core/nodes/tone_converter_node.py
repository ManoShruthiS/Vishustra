import logging
from typing import Any, Dict

# Assuming this path exists in the Vishustra project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ToneConverterNode(BaseNode):
    """
    A processing node that simulates converting the tone of input text.

    This node is designed to transform textual data by adjusting its
    perceived tone (e.g., from informal to formal, or neutral to humorous).
    The actual tone conversion logic is simulated; in a production system,
    this would typically integrate with an LLM or a sophisticated NLP service.

    The target tone can be specified during initialization or overridden
    via the 'target_tone' key in the context dictionary during processing.
    """

    def __init__(self, default_target_tone: str = "neutral"):
        """
        Initializes the ToneConverterNode with a default tone.

        Args:
            default_target_tone: The default tone to convert to if no
                                 'target_tone' is provided in the process context.
                                 Examples include "formal", "casual", "humorous", "serious".
        Raises:
            ValueError: If the `default_target_tone` is not a valid non-empty string.
        """
        if not isinstance(default_target_tone, str) or not default_target_tone.strip():
            logger.error(
                f"[{self.__class__.__name__}] Invalid default_target_tone provided: '{default_target_tone}'. "
                "Must be a non-empty string."
            )
            raise ValueError("Default target tone must be a non-empty string.")
        self._default_target_tone = default_target_tone.lower().strip()
        logger.info(
            f"[{self.node_name}] Initialized with default target tone: "
            f"'{self._default_target_tone}'."
        )

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "ToneConverter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input text data, simulating a tone conversion.

        The input `data` is expected to be a string. The node attempts to
        determine the desired target tone from the `context` dictionary
        (using the key 'target_tone'). If not found, it falls back to the
        `default_target_tone` set during initialization.

        Args:
            data: The input text (str) to be converted.
            context: A dictionary containing additional runtime information.
                     A 'target_tone' (str) key can be used here to dynamically
                     override the default target tone for this specific call.

        Returns:
            The simulated tone-converted text (str). This simulation
            prepends a tone indicator to the original text.

        Raises:
            ValueError: If the input `data` is not a string, or if the
                        determined target tone (from context or default) is
                        not a valid non-empty string.
        """
        logger.debug(f"[{self.node_name}] Initiating processing. Context keys: {list(context.keys())}")

        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected 'str', "
                f"received '{type(data).__name__}'."
            )
            raise ValueError(
                f"[{self.node_name}] Input 'data' must be a string for tone conversion."
            )

        if not data.strip():
            logger.warning(
                f"[{self.node_name}] Received empty or whitespace-only input data. "
                "Returning data as is."
            )
            return data

        # Determine the target tone, preferring context over default
        target_tone = context.get('target_tone', self._default_target_tone)

        if not isinstance(target_tone, str) or not target_tone.strip():
            logger.error(
                f"[{self.node_name}] Invalid target tone determined. "
                f"From context: '{context.get('target_tone')}', "
                f"Default: '{self._default_target_tone}'. "
                "Target tone must be a non-empty string."
            )
            raise ValueError(
                f"[{self.node_name}] Target tone must be a non-empty string."
            )
        target_tone = target_tone.lower().strip()

        logger.info(f"[{self.node_name}] Attempting to convert text to '{target_tone}' tone.")

        # Simulate the tone conversion.
        # In a real scenario, this would involve a complex NLP model or
        # an LLM API call, possibly rephrasing the entire text.
        # For this simulation, we prepend a clear indicator.
        converted_text = f"[[Tone: {target_tone.capitalize()}]] {data}"

        logger.debug(
            f"[{self.node_name}] Successfully simulated text conversion to "
            f"'{target_tone}' tone."
        )
        return converted_text