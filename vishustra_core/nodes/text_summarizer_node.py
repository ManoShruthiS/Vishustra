import logging
from typing import Any, Dict

# Assuming vishustra_core.nodes.base_node exists in the project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class TextSummarizerNode(BaseNode):
    """
    A Vishustra processing node that simulates text summarization.

    This node takes a string as input and returns a truncated version
    based on 'summarization_max_length' or 'summarization_ratio' provided
    in the context, simulating a content condensation process.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "TextSummarizerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by simulating text summarization.

        Expects `data` to be a string.
        Context can contain:
        - 'summarization_max_length': int, maximum characters for the summary.
        - 'summarization_ratio': float, ratio of original text length to keep.
                                 Takes precedence if both are provided.

        If neither is provided, a default max_length of 250 characters is used.

        Args:
            data: The text content (str) to be summarized.
            context: A dictionary containing operational parameters,
                     e.g., 'summarization_max_length' or 'summarization_ratio'.

        Returns:
            A string representing the summarized text.

        Raises:
            TypeError: If `data` is not a string.
            ValueError: If summarization parameters are invalid (e.g., negative length or invalid ratio).
        """
        logger.info(f"[{self.node_name}] Starting text summarization process.")

        if not isinstance(data, str):
            logger.error(f"[{self.node_name}] Invalid input data type: Expected string, got {type(data).__name__}.")
            raise TypeError(f"Input data for {self.node_name} must be a string, got {type(data).__name__}.")

        if not data.strip():
            logger.warning(f"[{self.node_name}] Input data is an empty or whitespace-only string. Returning as-is.")
            return ""

        original_length = len(data)
        max_length = 250  # Default max characters if no context parameters are given

        # Prioritize 'summarization_ratio' if present
        if 'summarization_ratio' in context:
            try:
                ratio = float(context['summarization_ratio'])
                if not (0.0 < ratio <= 1.0):
                    logger.error(f"[{self.node_name}] 'summarization_ratio' must be between 0.0 (exclusive) and 1.0 (inclusive), got {ratio}.")
                    raise ValueError(f"'summarization_ratio' must be between 0.0 (exclusive) and 1.0 (inclusive), got {ratio}.")
                max_length = int(original_length * ratio)
                logger.debug(f"[{self.node_name}] Using summarization_ratio: {ratio}. Calculated max_length: {max_length}")
            except (ValueError, TypeError) as e:
                logger.error(f"[{self.node_name}] Invalid 'summarization_ratio' in context: {e}. Attempting to use 'summarization_max_length'.")
                # Fallback: if ratio is invalid, try max_length if available
                if 'summarization_max_length' in context:
                    try:
                        max_length = int(context['summarization_max_length'])
                        logger.debug(f"[{self.node_name}] Falling back to summarization_max_length: {max_length}")
                    except (ValueError, TypeError) as e:
                        logger.warning(f"[{self.node_name}] Invalid 'summarization_max_length' in context: {e}. Using default max_length: {max_length}")
                else:
                    logger.warning(f"[{self.node_name}] No valid 'summarization_max_length' found, using default max_length: {max_length}")

        elif 'summarization_max_length' in context:
            try:
                max_length = int(context['summarization_max_length'])
                if max_length < 0:
                    logger.error(f"[{self.node_name}] 'summarization_max_length' cannot be negative, got {max_length}.")
                    raise ValueError(f"'summarization_max_length' cannot be negative, got {max_length}.")
                logger.debug(f"[{self.node_name}] Using summarization_max_length: {max_length}")
            except (ValueError, TypeError) as e:
                logger.error(f"[{self.node_name}] Invalid 'summarization_max_length' in context: {e}. Using default max_length: {max_length}")
        else:
            logger.debug(f"[{self.node_name}] No summarization parameters found in context. Using default max_length: {max_length}")

        if max_length <= 0:
            logger.warning(f"[{self.node_name}] Calculated or provided max_length is {max_length}. Returning empty string for summarization.")
            return ""

        # Ensure effective_max_length doesn't exceed original length by design, unless a very large max_length was given
        effective_max_length = min(max_length, original_length)

        if original_length <= effective_max_length:
            logger.debug(f"[{self.node_name}] Original text length ({original_length}) is less than or equal to effective max_length ({effective_max_length}). Returning original text.")
            return data

        # Simulate summarization via truncation, adding an ellipsis if truncated
        summarized_text = data
        if original_length > effective_max_length:
            if effective_max_length >= 3:
                # Truncate to make space for "..."
                summarized_text = data[:effective_max_length - 3] + "..."
            else:
                # If effective_max_length is too small for "...", just truncate
                summarized_text = data[:effective_max_length]

        logger.info(f"[{self.node_name}] Summarization complete. Original length: {original_length}, Summarized length: {len(summarized_text)}")
        return summarized_text