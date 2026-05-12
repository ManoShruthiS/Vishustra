import logging
from typing import Any, Dict
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class TextSummarizerNode(BaseNode):
    """
    A modular node within the Vishustra framework designed to condense text.
    
    This node processes raw text input and produces a summarized version based
    on configuration parameters provided in the execution context.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the unique identifier for this node.
        """
        return "TextSummarizerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input string to generate a summarized output.

        Args:
            data (Any): The text content to be summarized. Must be a string.
            context (Dict[str, Any]): Orchestration context containing operational 
                                      parameters such as 'max_length' or 'strict_mode'.

        Returns:
            str: The summarized text content.

        Raises:
            TypeError: If the input data is not a string.
            ValueError: If the summarization logic encounters invalid context parameters.
            RuntimeError: If an unexpected error occurs during data transformation.
        """
        if not isinstance(data, str):
            error_msg = f"[{self.node_name}] Expected string input, received {type(data).__name__}."
            logger.error(error_msg)
            raise TypeError(error_msg)

        if not data.strip():
            logger.warning(f"[{self.node_name}] Received empty or whitespace-only input string.")
            return ""

        try:
            # Retrieve configuration from context with sensible defaults
            max_summary_length = context.get("max_length", 300)
            preserve_formatting = context.get("preserve_formatting", False)

            logger.info(f"[{self.node_name}] Summarizing payload (Input Length: {len(data)})")

            # Simulation of an extractive summarization transformation.
            # In a production LLM pipeline, this logic would typically interface with
            # a Transformer-based model or an external API provider.
            sentences = [s.strip() for s in data.split('.') if s.strip()]
            
            if len(sentences) <= 3:
                summary = " ".join(sentences)
            else:
                # Heuristic: Capture the primary context and the concluding result
                summary_parts = sentences[:2] + [sentences[-1]]
                summary = ". ".join(summary_parts) + "."

            # Final truncation and sanitization
            if len(summary) > max_summary_length:
                summary = summary[:max_summary_length].rsplit(' ', 1)[0] + "..."

            if not preserve_formatting:
                summary = summary.replace("\n", " ").strip()

            logger.debug(f"[{self.node_name}] Successfully generated summary of length {len(summary)}.")
            return summary

        except Exception as e:
            logger.exception(f"[{self.node_name}] Transformation failed due to an internal error.")
            raise RuntimeError(f"Node '{self.node_name}' failed to process data: {str(e)}") from e

```python
# Example usage (not part of the production node file)
# node = TextSummarizerNode()
# result = node.process("Long text here...", {"max_length": 100})
