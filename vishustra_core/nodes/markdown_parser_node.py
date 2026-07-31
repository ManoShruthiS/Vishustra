import logging
import re
from typing import Any, Dict

# Assuming vishustra_core is accessible in the environment.
# In a real project, this would be an actual module path.
try:
    from vishustra_core.nodes.base_node import BaseNode
except ImportError:
    # This fallback allows local development and testing even if the full
    # vishustra_core package structure isn't yet in place or installed.
    # In a deployed environment, this would signify a critical dependency issue.
    logging.warning(
        "Could not import BaseNode from vishustra_core.nodes.base_node. "
        "Using a mock BaseNode for development purposes. Ensure 'vishustra_core' "
        "is correctly installed and configured in your production environment."
    )
    from abc import ABC, abstractmethod
    class BaseNode(ABC):
        @abstractmethod
        def process(self, data: Any, context: Dict[str, Any]) -> Any:
            pass
        @property
        @abstractmethod
        def node_name(self) -> str:
            pass


logger = logging.getLogger(__name__)


class MarkdownParserNode(BaseNode):
    """
    A Vishustra processing node that simulates parsing Markdown input
    and converting it to a basic HTML string.

    This node provides a simplified conversion handling common Markdown elements
    like headings, strong/emphasis text, and inline code. It aims to structure
    the output into valid HTML paragraphs and line breaks.

    For production environments requiring full CommonMark compliance or advanced
    features, this node should be extended to integrate with a robust Markdown
    parsing library (e.g., `markdown-it-py`, `mistune`, `Python-Markdown`).
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "MarkdownParser"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Parses the input data, expected to be a Markdown string,
        and converts it into a basic HTML string.

        The `context` dictionary is currently not utilized by this node but
        is provided for consistency with `BaseNode` and future extensibility.

        Args:
            data: The input data, expected to be a string containing Markdown.
            context: A dictionary containing contextual information for processing.

        Returns:
            A string representing the HTML rendition of the input Markdown.

        Raises:
            TypeError: If the input `data` is not a string.
            ValueError: If an unexpected error occurs during the parsing process.
        """
        if not isinstance(data, str):
            error_msg = f"Invalid input type for MarkdownParserNode. Expected 'str', got '{type(data).__name__}'."
            logger.error(error_msg)
            raise TypeError(error_msg)

        if not data.strip():
            logger.info("Received empty or whitespace-only markdown string. Returning an empty string.")
            return ""

        processed_html = data
        try:
            # Step 1: Handle block-level elements (Headings)
            # Converts ATX style headings (e.g., # Heading) to HTML <h1>-<h6> tags.
            def _replace_heading(match):
                level = len(match.group(1))
                text = match.group(2).strip()
                if level > 6:
                    level = 6  # HTML headings cap at h6
                return f"<h{level}>{text}</h{level}>\n" # Add a newline to separate blocks naturally

            processed_html = re.sub(r'^(#{1,6})\s+(.*)$', _replace_heading, processed_html, flags=re.MULTILINE)

            # Step 2: Handle inline code blocks (e.g., `code`)
            # Must be processed before strong/emphasis to prevent conflicts.
            processed_html = re.sub(r'`([^`]+)`', r'<code>\1</code>', processed_html)

            # Step 3: Handle inline strong/bold text (e.g., **text** or __text__)
            # Non-greedy match for content between delimiters that are not the delimiter itself.
            processed_html = re.sub(
                r'\*\*([^*]+)\*\*|__([^_]+)__',
                lambda m: f"<strong>{m.group(1) or m.group(2)}</strong>",
                processed_html
            )
            
            # Step 4: Handle inline emphasis/italic text (e.g., *text* or _text_)
            # Ensures it doesn't accidentally match parts of strong text (e.g., `**foo*bar**`)
            processed_html = re.sub(
                r'\*([^*]+)\*|_([^_]+)_',
                lambda m: f"<em>{m.group(1) or m.group(2)}</em>",
                processed_html
            )

            # Step 5: Paragraph and Line break handling
            # Split the text into potential paragraphs by looking for double newlines.
            # This is a simplification; a full parser would build an AST.
            paragraphs = processed_html.split('\n\n')
            final_html_parts = []

            for para_segment in paragraphs:
                para_segment = para_segment.strip()
                if not para_segment:
                    continue # Skip empty segments

                # Very basic check: if a segment already starts with a common block-level HTML tag,
                # assume it's already structured and append as is.
                if re.match(r'^\s*<(h[1-6]|ul|ol|p|div|pre|blockquote)>', para_segment, re.IGNORECASE):
                    final_html_parts.append(para_segment)
                else:
                    # For segments not recognized as existing blocks, replace single newlines with <br/>
                    # and wrap the entire segment in <p> tags.
                    processed_para = re.sub(r'\n', '<br/>', para_segment)
                    final_html_parts.append(f"<p>{processed_para}</p>")
            
            # Join the processed paragraphs/blocks with a newline for better readability in the output HTML.
            result_html = "\n".join(final_html_parts)

            logger.debug(
                f"Successfully parsed markdown. Original (first 100 chars): '{data[:100]}...', "
                f"Processed (first 100 chars): '{result_html[:100]}...'"
            )
            return result_html

        except Exception as e:
            error_msg = f"An unexpected error occurred during Markdown parsing: {e}"
            logger.exception(error_msg) # Log the full traceback for detailed debugging
            raise ValueError(error_msg) from e

