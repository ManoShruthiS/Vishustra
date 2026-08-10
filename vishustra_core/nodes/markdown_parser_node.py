import logging
import re
from typing import Any, Dict

# Assuming BaseNode is available at this path as per project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class MarkdownParserNode(BaseNode):
    """
    A Vishustra processing node that parses Markdown formatted text into a simplified
    HTML string. This node provides basic Markdown-to-HTML conversion capabilities,
    handling common elements such as headings, bold, italic, and links.

    It's designed for simple markdown structures and simulates a lightweight parsing
    process, rather than implementing a full-fledged Markdown specification.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "MarkdownParserNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Parses the input data, expecting a Markdown-formatted string, and converts
        it into a simplified HTML string.

        Args:
            data (Any): The input data, expected to be a string containing Markdown.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the processing. Not directly used for parsing
                                       logic in this node, but available for future
                                       enhancements (e.g., configuration for parsing rules).

        Returns:
            Any: A string containing the simplified HTML representation of the input Markdown.

        Raises:
            TypeError: If the input data is not a string.
            ValueError: If an unexpected issue occurs during the simulated parsing process.
        """
        logger.debug(f"[{self.node_name}] Starting process for data type: {type(data)}")

        if not isinstance(data, str):
            error_msg = (
                f"[{self.node_name}] Invalid input data type. Expected 'str', "
                f"got '{type(data).__name__}'."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

        markdown_text = data
        html_output = []
        paragraph_buffer = []

        try:
            # Split the markdown text into lines for block-level processing
            lines = markdown_text.split('\n')

            for line in lines:
                line = line.strip()

                # Detect and process block-level elements
                header_match = re.match(r'^(#+)\s*(.*)$', line)

                if header_match:
                    # Flush any accumulated paragraph text before a new block element
                    if paragraph_buffer:
                        processed_para = self._process_inline_markdown(' '.join(paragraph_buffer))
                        html_output.append(f"<p>{processed_para}</p>")
                        paragraph_buffer = []
                    
                    level = min(len(header_match.group(1)), 6) # Max H6
                    content = header_match.group(2).strip()
                    html_output.append(f"<h{level}>{content}</h{level}>")
                    continue
                
                if not line:
                    # An empty line often signifies the end of a paragraph
                    if paragraph_buffer:
                        processed_para = self._process_inline_markdown(' '.join(paragraph_buffer))
                        html_output.append(f"<p>{processed_para}</p>")
                        paragraph_buffer = []
                    continue # Skip adding empty lines to buffer

                # Accumulate lines for a paragraph
                paragraph_buffer.append(line)

            # After iterating through all lines, flush any remaining paragraph content
            if paragraph_buffer:
                processed_para = self._process_inline_markdown(' '.join(paragraph_buffer))
                html_output.append(f"<p>{processed_para}</p>")

            result = '\n'.join(html_output)
            logger.debug(f"[{self.node_name}] Successfully parsed Markdown content into HTML.")
            return result

        except Exception as e:
            error_msg = f"[{self.node_name}] An unexpected error occurred during Markdown parsing: {e}"
            logger.exception(error_msg) # Log full traceback for debugging
            raise ValueError(error_msg) from e

    def _process_inline_markdown(self, text: str) -> str:
        """
        Helper method to process inline Markdown elements within a given text block,
        such as bold, italic, and links.
        """
        # Process links: [link text](url) -> <a href="url">link text</a>
        text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', text)
        
        # Process bold: **text** -> <strong>text</strong>
        # Using non-greedy match (.*?) to prevent over-matching across multiple bold segments
        text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
        
        # Process italic: *text* -> <em>text</em>
        # This regex should be applied after bold to avoid conflicts.
        text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)
        
        return text