from abc import ABC, abstractmethod
import logging
from typing import Any, Dict

# Assuming BaseNode is available at this path within the Vishustra framework.
# The base class definition is provided in the project context.
# For local execution without the full framework, one might temporarily define BaseNode here.
# For production, this import path is critical.
try:
    from vishustra_core.nodes.base_node import BaseNode
except ImportError:
    # Fallback for isolated testing if vishustra_core is not installed,
    # assuming the structure provided in the prompt's context.
    logging.warning(
        "Could not import BaseNode from vishustra_core.nodes.base_node. "
        "Using a mock BaseNode definition. Ensure vishustra_core is installed for production."
    )
    class BaseNode(ABC): # type: ignore
        @abstractmethod
        def process(self, data: Any, context: Dict[str, Any]) -> Any:
            pass
        @property
        @abstractmethod
        def node_name(self) -> str:
            pass


logger = logging.getLogger(__name__)

class TextSummarizerNode(BaseNode):
    """
    A processing node designed to generate a concise summary of input text.
    This node expects string data and produces a truncated version,
    useful for initial content previews or short descriptions.
    """

    DEFAULT_SUMMARY_LENGTH = 200
    ELLIPSIS_SUFFIX = "..."

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this processing node."""
        return "TextSummarizer"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Summarizes the input text by truncating it to a specified length.

        The `context` dictionary can optionally include 'summary_length'
        (int) to override the default summary length.

        Args:
            data (Any): The input data expected to be a string.
            context (Dict[str, Any]): A dictionary containing additional
                                      runtime information or configuration.
                                      Can include 'summary_length'.

        Returns:
            Any: The summarized string, or the original data if summarization
                 is not applicable or an empty string if length is 0.

        Raises:
            TypeError: If the input `data` is not a string.
            ValueError: If the input `data` is an empty string.
        """
        logger.info(f"[{self.node_name}] Starting text summarization.")

        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input type. Expected string, "
                f"but received {type(data).__name__}."
            )
            raise TypeError(
                f"[{self.node_name}] 'data' must be a string, got {type(data).__name__}."
            )

        if not data:
            logger.warning(f"[{self.node_name}] Received empty string data for summarization.")
            raise ValueError(f"[{self.node_name}] 'data' cannot be an empty string.")

        summary_length = self.DEFAULT_SUMMARY_LENGTH
        requested_length = context.get('summary_length')

        if requested_length is not None:
            if isinstance(requested_length, int):
                # Ensure summary_length is not negative, use 0 if negative or fallback to default if 0
                summary_length = max(0, requested_length)
                if summary_length == 0:
                    logger.debug(
                        f"[{self.node_name}] 'summary_length' set to 0 in context. "
                        "Returning an empty string."
                    )
                elif requested_length < 0:
                     logger.warning(
                        f"[{self.node_name}] Negative 'summary_length' ({requested_length}) "
                        f"provided in context. Clamping to 0."
                    )
            else:
                logger.warning(
                    f"[{self.node_name}] 'summary_length' in context is not an integer "
                    f"({type(requested_length).__name__}). Using default length "
                    f"{self.DEFAULT_SUMMARY_LENGTH}."
                )
                summary_length = self.DEFAULT_SUMMARY_LENGTH

        if summary_length == 0:
            logger.info(f"[{self.node_name}] Summarization completed: returned empty string.")
            return ""

        if len(data) <= summary_length:
            logger.debug(
                f"[{self.node_name}] Original text length ({len(data)}) is "
                f"less than or equal to desired summary length ({summary_length}). "
                "Returning full text."
            )
            logger.info(f"[{self.node_name}] Summarization completed.")
            return data
        else:
            summarized_text = data[:summary_length] + self.ELLIPSIS_SUFFIX
            logger.debug(
                f"[{self.node_name}] Text summarized to {summary_length} characters. "
                f"Original length: {len(data)}, Summarized length: {len(summarized_text)}."
            )
            logger.info(f"[{self.node_name}] Summarization completed.")
            return summarized_text

