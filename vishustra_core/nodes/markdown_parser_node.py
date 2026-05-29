import logging
import re
from typing import Any, Dict

# Assuming the BaseNode class is located at this path as per project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class MarkdownParserNode(BaseNode):
    """
    A Vishustra node designed to parse Markdown text and transform it into 
    a simplified HTML string representation. This node supports basic Markdown
    syntax elements like headers, bold text, italic text, and simple links,
    structuring output into HTML paragraphs and block elements.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "MarkdownParser"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Processes the input data, interpreting Markdown syntax and converting it
        into a simplified HTML string. This method handles block-level elements
        like headers and paragraphs, as well as inline formatting such as bold,
        italic, and basic links.

        Args:
            data: The input data, expected to be a string containing Markdown formatted text.
            context: A dictionary containing contextual information. This node does not
                     currently utilize the context for its primary parsing logic, but it
                     is included as per the BaseNode interface.

        Returns:
            A string representing the HTML output of the parsed Markdown.

        Raises:
            TypeError: If the input 'data' is not a string.
        """
        if not isinstance(data, str):
            logger.error(
                f"MarkdownParserNode received invalid input type. Expected string, "
                f"but got {type(data).__name__}. This node requires string input for parsing."
            )
            raise TypeError("MarkdownParserNode expects 'data' to be a string.")

        if not data.strip():
            logger.info("MarkdownParserNode received empty or whitespace-only input data. Returning an empty string.")
            return ""

        logger.debug("MarkdownParserNode: Starting markdown parsing for input data.")

        # Split input into lines for easier processing of block-level elements
        lines = data.split('\n')
        processed_blocks = [] # Stores final HTML blocks (e.g., <p>...</p>, <h1>...</h1>)
        current_paragraph_lines = [] # Collects lines that belong to the same paragraph for later wrapping

        def _flush_paragraph():
            """Helper to convert collected paragraph lines into an HTML paragraph block."""
            if current_paragraph_lines:
                # Join lines, treating multiple spaces as a single space for paragraph content
                paragraph_content = " ".join(current_paragraph_lines).strip()
                if paragraph_content: # Only add if there's actual content after stripping
                    processed_blocks.append(f"<p>{paragraph_content}</p>")
                current_paragraph_lines.clear()

        for line in lines:
            trimmed_line = line.strip()

            if not trimmed_line:
                # An empty line signals the end of a current paragraph.
                _flush_paragraph()
                continue

            # Check for Headers (e.g., # H1, ## H2)
            header_match = re.match(r"^(#+)\s*(.*)$", trimmed_line)
            if header_match:
                _flush_paragraph() # Headers always break the current paragraph
                level = len(header_match.group(1))
                content = header_match.group(2).strip()
                if 1 <= level <= 6: # Standard HTML header levels
                    processed_blocks.append(f"<h{level}>{content}</h{level}>")
                else:
                    # If header level is out of standard range, treat it as a regular paragraph
                    logger.warning(
                        f"MarkdownParserNode: Detected header level '{level}' out of standard HTML bounds (1-6). "
                        f"Treating content '{content}' as a paragraph."
                    )
                    # Inline processing on content before wrapping in <p>
                    processed_inline_content = self._process_inline_markdown(content)
                    processed_blocks.append(f"<p>{processed_inline_content}</p>")
                continue
            
            # For all other lines, apply inline processing and add to current paragraph
            line_content_processed_inline = self._process_inline_markdown(trimmed_line)
            current_paragraph_lines.append(line_content_processed_inline)

        # Flush any remaining paragraph content after the loop finishes
        _flush_paragraph()

        # Join all processed HTML blocks with newlines for readability in the output
        final_html = "\n".join(processed_blocks)
        logger.debug("MarkdownParserNode: Markdown parsing completed successfully.")
        return final_html

    def _process_inline_markdown(self, text: str) -> str:
        """
        Helper method to apply inline Markdown transformations to a given string.
        """
        # Bold: **text**
        # Using a non-greedy match (.*?) to ensure it matches the smallest possible string
        text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", text)
        
        # Italic: *text*
        # Ensure this is after bold to avoid conflicts if `**text*` was possible
        text = re.sub(r"\*(.*?)\*", r"<em>\1</em>", text)
        
        # Simple Links: [link text](url)
        text = re.sub(r"\[(.*?)\]\((.*?)\)", r'<a href="\2">\1</a>', text)

        return text
