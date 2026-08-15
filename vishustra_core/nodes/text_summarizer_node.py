import logging
from typing import Any, Dict, Optional

# Assuming this path is correct as per project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class TextSummarizerNode(BaseNode):
    """
    A Vishustra processing node designed for abstractive text summarization.

    This node takes a string of text as input and produces a concise summary.
    In a production environment, this would integrate with an underlying
    Large Language Model (LLM) or a specialized text summarization service.
    For demonstration, it simulates summarization via truncation and a
    placeholder message.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the TextSummarizerNode with configurable parameters.

        Args:
            config: An optional dictionary for node configuration.
                    Expected keys might include 'max_summary_length' (int),
                    'min_summary_length' (int), or 'model_name' (str)
                    for an actual LLM integration.
        """
        self._config = config or {}
        # Default simulation parameters; would be passed to an LLM in a real scenario
        self._max_summary_length = self._config.get('max_summary_length', 250)
        self._min_summary_length = self._config.get('min_summary_length', 50)
        logger.debug(f"TextSummarizerNode initialized with config: {self._config}")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "TextSummarizer"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by simulating text summarization.

        It expects the 'data' parameter to be a string containing the text
        to be summarized. The 'context' dictionary can provide additional
        runtime parameters, though not explicitly used for this simulation.

        Args:
            data: The input text as a string to be summarized.
            context: A dictionary containing contextual information relevant
                     to the current processing flow.

        Returns:
            A string representing the simulated summary of the input text.

        Raises:
            TypeError: If the input 'data' is not of type string.
            ValueError: If the input 'data' is an empty or whitespace-only string.
        """
        logger.info(f"Node '{self.node_name}' starting processing.")
        logger.debug(f"Received data type: {type(data)}, context keys: {list(context.keys()) if context else 'N/A'}")

        if not isinstance(data, str):
            logger.error(f"Invalid input type for '{self.node_name}'. Expected string, but received {type(data).__name__}.")
            raise TypeError(
                f"Input data for '{self.node_name}' must be a string, "
                f"but received type {type(data).__name__}."
            )

        if not data.strip():
            logger.warning(f"Empty or whitespace-only string provided to '{self.node_name}'. Returning an empty summary.")
            raise ValueError(f"Input data for '{self.node_name}' cannot be empty or whitespace-only.")

        # --- Simulation of Summarization Logic ---
        # In a real-world implementation, this section would involve:
        # 1. Calling an external LLM API (e.g., OpenAI, Anthropic).
        # 2. Using an internal model for summarization.
        # 3. Handling API keys, rate limits, and model-specific parameters.

        original_text_length = len(data)
        # Determine the maximum length for the simulated summary,
        # ensuring it doesn't exceed the original text length.
        effective_max_length = min(original_text_length, self._max_summary_length)

        simulated_summary: str
        if effective_max_length < self._min_summary_length:
            # If the text is too short to warrant a complex summary,
            # return it as is or with minor trimming.
            simulated_summary = data.strip()
            logger.debug(
                f"Original text ({original_text_length} chars) is shorter "
                f"than min_summary_length ({self._min_summary_length}). "
                f"Returning full text as simulated summary."
            )
        else:
            # Simulate truncation with a smart cut-off to avoid breaking words
            truncated_segment = data[:effective_max_length].strip()
            if len(truncated_segment) < original_text_length and ' ' in truncated_segment:
                # Find the last space to avoid cutting off mid-word
                last_space_idx = truncated_segment.rfind(' ')
                if last_space_idx != -1:
                    truncated_segment = truncated_segment[:last_space_idx]

            # Add a placeholder to indicate this is a simulated output
            simulated_summary = f"(Simulated Summary) {truncated_segment}..." \
                                if len(truncated_segment) < original_text_length else \
                                f"(Simulated Summary) {truncated_segment}"
            logger.debug(
                f"Simulated summary generated. Original length: {original_text_length}, "
                f"Simulated summary length: {len(simulated_summary)}."
            )
        # --- End of Simulation Logic ---

        logger.info(f"Node '{self.node_name}' finished processing. Summary length: {len(simulated_summary)} characters.")
        return simulated_summary

