import logging
from typing import Any, Dict

# Assuming this path is where BaseNode resides in the Vishustra project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class ToneConverterNode(BaseNode):
    """
    A Vishustra processing node responsible for simulating tone conversion of text data.
    It expects a string as input data and a 'target_tone' string in the context
    to perform a simulated transformation.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this node."""
        return "ToneConverter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to simulate tone conversion based on the
        'target_tone' provided in the context.

        Args:
            data: The input text to be converted, expected to be a string.
            context: A dictionary containing additional processing parameters,
                     expected to include 'target_tone' as a string.

        Returns:
            The simulated tone-converted string, or the original data if
            conversion parameters are invalid.

        Raises:
            TypeError: If the input data is not a string.
        """
        if not isinstance(data, str):
            logger.error(
                "[%s] Invalid input data type. Expected 'str', but received '%s'.",
                self.node_name,
                type(data).__name__,
            )
            raise TypeError(
                f"Input data for {self.node_name} must be a string, got {type(data).__name__}."
            )

        target_tone = context.get("target_tone")

        if not isinstance(target_tone, str) or not target_tone.strip():
            logger.warning(
                "[%s] No valid 'target_tone' specified in the context. "
                "Returning original data without simulated tone conversion.",
                self.node_name,
            )
            # In a real LLM scenario, a default neutral tone might be applied or
            # the request might be rejected. For this simulation, we return original.
            return data

        # Simulate tone conversion by prepending the target tone.
        # In a live LLM orchestration, this would involve a sophisticated prompt
        # to an underlying language model configured for tone adjustment.
        processed_data = f"[{target_tone.strip().capitalize()} Tone] {data}"

        logger.debug(
            "[%s] Successfully applied simulated '%s' tone conversion. "
            "Original data snippet: '%s...', Processed data snippet: '%s...'",
            self.node_name,
            target_tone.strip(),
            data[:50].replace('\n', ' '), # Log first 50 chars, replace newlines for readability
            processed_data[:50].replace('\n', ' ')
        )

        return processed_data