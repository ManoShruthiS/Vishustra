import logging
from typing import Any, Dict, Optional
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class TextSummarizerNode(BaseNode):
    """
    A specialized node within the Vishustra framework designed to condense 
    large text inputs into succinct summaries while preserving core semantic meaning.
    
    This node expects a string input and utilizes configuration parameters 
    passed through the context to determine summarization constraints.
    """

    def __init__(self, default_max_length: int = 200):
        """
        Initializes the TextSummarizerNode with default parameters.
        
        Args:
            default_max_length (int): The default character limit for the summary if not provided in context.
        """
        self.default_max_length = default_max_length

    @property
    def node_name(self) -> str:
        """
        Returns the unique identifier for this node type.
        """
        return "TextSummarizerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Processes the input string to generate a condensed version.

        In a production LLM orchestration workflow, this node would typically interface 
        with an LLM provider or a local transformer model. This implementation 
        simulates the logic of content transformation and validation.

        Args:
            data (Any): The raw text to be summarized.
            context (Dict[str, Any]): Execution context containing metadata, 
                                      session state, and configuration overrides 
                                      such as 'max_length'.

        Returns:
            str: The processed summary.

        Raises:
            TypeError: If the input data is not a string.
            ValueError: If the input data is empty.
            RuntimeError: If an internal processing error occurs.
        """
        try:
            # Type validation
            if not isinstance(data, str):
                logger.error(f"[{self.node_name}] Input data must be a string, received {type(data).__name__}")
                raise TypeError(f"TextSummarizerNode requires string input, got {type(data).__name__}")

            # Content validation
            clean_data = data.strip()
            if not clean_data:
                logger.warning(f"[{self.node_name}] Received empty string for summarization.")
                return ""

            # Extract configuration from context or use defaults
            max_length = context.get("max_length", self.default_max_length)
            logger.debug(f"[{self.node_name}] Processing string of length {len(clean_data)} with limit {max_length}")

            # Simulation of text transformation logic
            # In actual implementation: result = llm_client.summarize(clean_data, max_length=max_length)
            if len(clean_data) <= max_length:
                summary = clean_data
            else:
                # Basic heuristic-based truncation for simulation purposes
                summary = clean_data[:max_length].rsplit(' ', 1)[0] + "..."

            logger.info(f"[{self.node_name}] Successfully generated summary of length {len(summary)}")
            return summary

        except (TypeError, ValueError) as ve:
            # Re-raise known validation errors
            raise ve
        except Exception as e:
            logger.exception(f"[{self.node_name}] Unexpected error during processing: {str(e)}")
            raise RuntimeError(f"Summarization failed due to an internal error: {e}") from e