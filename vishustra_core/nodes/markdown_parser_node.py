import logging
import re
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class MarkdownParserNode(BaseNode):
    """
    A Vishustra node that parses Markdown text into a simplified HTML-like string.

    This node demonstrates text transformation by converting common Markdown
    elements such as headers, bold/italic text, links, and lists into
    their corresponding HTML tag representations. It handles basic block-level
    and inline Markdown syntax.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "MarkdownParser"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Parses Markdown input data into a simplified HTML-like string.

        This method takes a string containing Markdown content and transforms
        it into an HTML-like string by replacing common Markdown syntax
        with corresponding HTML tags (e.g., # Header -> <h1>Header</h1>,
        **bold** -> <b>bold</b>).

        Args:
            data: The input data, expected to be a string containing Markdown.
            context: A dictionary containing contextual information. Not directly
                     used by this node but available for future extensions or
                     to pass information through the orchestration pipeline.

        Returns:
            A string representing the HTML-like parsed content.

        Raises:
            TypeError: If the input data is not a string.
        """
        if not isinstance(data, str):
            logger.error(
                f"Invalid input type for MarkdownParserNode. Expected str, got {type(data).__name__}."
            )
            raise TypeError(
                f"MarkdownParserNode expects string data, but received {type(data).__name__}"
            )

        markdown_text: str = data.strip()
        if not markdown_text:
            logger.warning(
                "Received empty or whitespace-only string for MarkdownParserNode. Returning empty string."
            )
            return ""

        parsed_elements = []
        lines = markdown_text.split('\n')
        
        # This implementation simplifies block parsing: each non-empty line
        # is processed independently for block-level elements. A more advanced
        # parser would group consecutive non-block lines into a single paragraph.
        # For demonstration within a single node, this line-by-line approach suffices.
        for line in lines:
            stripped_line = line.strip()

            if not stripped_line:
                # Skip empty lines in the output; they implicitly act as paragraph breaks.
                continue

            # Headers (H1-H6)
            header_match = re.match(r"^(#+)\s*(.*)", stripped_line)
            if header_match:
                level = len(header_match.group(1))
                content = header_match.group(2).strip()
                # Cap header level at h6
                parsed_elements.append(f"<h{min(level, 6)}>{content}</h{min(level, 6)}>")
                continue

            # Unordered lists (simple, one item per line starting with - or *)
            if stripped_line.startswith(('- ', '* ')):
                content = stripped_line[2:].strip()
                # For simplicity, individual <li> elements are added. A <ul> tag
                # would typically wrap a group of <li>s, but this demonstrates
                # item transformation. Inline markdown is also processed within list items.
                parsed_elements.append(f"<li>{self._process_inline_markdown(content)}</li>")
                continue

            # Default to paragraph, processing inline elements within the line
            processed_line = self._process_inline_markdown(stripped_line)
            if processed_line:
                # Each non-block line becomes its own paragraph.
                parsed_elements.append(f"<p>{processed_line}</p>")

        # Join the processed elements with newlines for readability in the output string.
        return "\n".join(filter(None, parsed_elements))

    def _process_inline_markdown(self, text: str) -> str:
        """
        Helper method to process inline Markdown elements within a given string.
        Applies transformations for links, bold, and italic text.
        """
        # Process links first to prevent inner formatting from breaking the URL or text part.
        # Links: [text](url) -> <a href="url">text</a>
        text = re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\2">\1</a>', text)

        # Bold: **text** or __text__ -> <b>text</b>
        # Using non-greedy `*?` to match the smallest possible string between delimiters.
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
        text = re.sub(r'__(.*?)__', r'<b>\1</b>', text)

        # Italic: *text* or _text_ -> <i>text</i>
        # Using negative lookarounds `(?<!\*)` and `(?!\*)` to prevent matching
        # the double asterisks/underscores used for bold.
        text = re.sub(r'(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)', r'<i>\1</i>', text)
        text = re.sub(r'(?<!_)\_(?!_)(.*?)(?<!_)\_(?!_)', r'<i>\1</i>', text)

        return text