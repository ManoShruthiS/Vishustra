import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class ToneConverterNode(BaseNode):
    """
    A Vishustra processing node that simulates converting the tone of text data.
    It expects a string 'data' and a 'target_tone' in the 'context'.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "ToneConverter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, simulating a tone conversion based on the
        'target_tone' specified in the context.

        Args:
            data: The input text data, expected to be a string.
            context: A dictionary containing operational context, expected to
                     include 'target_tone' (e.g., "formal", "informal", "professional").

        Returns:
            The simulated tone-converted text.

        Raises:
            TypeError: If the input 'data' is not a string.
            ValueError: If 'target_tone' is missing or invalid in the context
                        (though currently, it defaults if missing).
        """
        if not isinstance(data, str):
            logger.error(
                "[%s] Invalid data type. Expected string, got %s.",
                self.node_name,
                type(data).__name__,
            )
            raise TypeError(
                f"[{self.node_name}] Input data must be a string, but received {type(data).__name__}."
            )

        target_tone = context.get("target_tone")

        if not target_tone:
            # Default to a neutral tone if not specified
            target_tone = "neutral"
            logger.warning(
                "[%s] 'target_tone' not found in context. Defaulting to '%s'.",
                self.node_name,
                target_tone,
            )
        elif not isinstance(target_tone, str):
            logger.error(
                "[%s] Invalid 'target_tone' type in context. Expected string, got %s.",
                self.node_name,
                type(target_tone).__name__,
            )
            raise ValueError(
                f"[{self.node_name}] 'target_tone' in context must be a string, but received {type(target_tone).__name__}."
            )

        # Simulate tone conversion. In a real-world scenario, this would involve
        # calling an external LLM, an NLP model, or a sophisticated rule-based system.
        # For this simulation, we append a descriptive tag.
        processed_data = f"{data.strip()} [converted to {target_tone} tone]"

        logger.info(
            "[%s] Successfully converted text to '%s' tone.", self.node_name, target_tone
        )
        logger.debug(
            "[%s] Original: '%s' -> Processed: '%s'",
            self.node_name,
            data,
            processed_data,
        )

        return processed_data
