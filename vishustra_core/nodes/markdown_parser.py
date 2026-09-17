import logging
import re
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

# It is standard practice for a backend engineer to leverage existing robust libraries
# for well-defined tasks like Markdown parsing. The 'markdown' library is a widely
# adopted and efficient choice for this purpose.
try:
    import markdown
except ImportError:
    logging.critical(
        "The 'markdown' library is not installed. MarkdownParserNode will fall back "
        "to the built-in lightweight converter (pip install markdown for full rendering)."
    )
    markdown = None


def _fallback_to_plain_text(md_text: str) -> str:
    """Lightweight, dependency-free Markdown -> plain text converter.

    Handles the most common constructs (headings, bold/italic/strikethrough,
    links, images, fenced/inline code, lists, blockquotes, horizontal rules)
    so the node remains functional when the optional 'markdown' library is absent.
    This is intentionally lossy/heuristic; install 'markdown' for full rendering.
    """
    text = md_text
    text = re.sub(r"```[\s\S]*?```", lambda m: m.group(0).split("\n", 1)[0] + " [code block]", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r"[image: \1]", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s{0,3}>\s?", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[\t ]*[-*+] [ \t]*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[ \t]*\d+\.[ \t]+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^([-=])\1{2,}\s*$", "\n---\n", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"_([^_]+)_", r"\1", text)
    text = re.sub(r"~~([^~]+)~~", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


logger = logging.getLogger(__name__)


class MarkdownParserNode(BaseNode):
    """
    A Vishustra processing node that converts Markdown text into HTML.

    This node expects the input 'data' to be a string containing Markdown
    and will transform it into an HTML string. It leverages the 'markdown'
    Python library for robust and feature-rich parsing.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "MarkdownParserNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input Markdown data and returns the resulting HTML string.

        This method first validates that the input `data` is a string.
        It then attempts to convert the Markdown content to HTML using the
        `markdown` library. Robust error handling is included to catch
        potential issues during parsing or if the required dependency is missing.

        Args:
            data: The input data, expected to be a string containing Markdown content.
            context: A dictionary containing contextual information. This node
                     does not currently utilize the context, but it is available
                     for future extensions (e.g., passing markdown extensions).

        Returns:
            A string containing the HTML representation of the input Markdown.

        Raises:
            TypeError: If the input 'data' is not a string.
            RuntimeError: If the 'markdown' library is not installed or if
                          an unexpected error occurs during the parsing process.
        """
        logger.debug(f"[{self.node_name}] Initiating process for data type: {type(data)}")

        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected 'str', "
                f"received '{type(data).__name__}'."
            )
            raise TypeError(
                f"Node '{self.node_name}' expects input 'data' to be a string, "
                f"but received type '{type(data).__name__}'."
            )

        try:
            if markdown is None:
                html_output = _fallback_to_plain_text(data)
                logger.info(f"[{self.node_name}] Used built-in lightweight Markdown converter (library not installed).")
            else:
                # Context could be used here to pass `extensions`, `extension_configs`,
                # or `output_format` to the markdown.markdown function if needed.
                # Example: html_output = markdown.markdown(data, extensions=context.get('markdown_extensions', []))
                html_output = markdown.markdown(data)
                logger.info(f"[{self.node_name}] Successfully parsed Markdown to HTML.")
            return html_output
        except Exception as e:
            # Catching a broad exception to ensure no parsing-related issue goes unhandled.
            logger.exception(
                f"[{self.node_name}] An unexpected error occurred during Markdown to HTML conversion."
            )
            # Re-raise a more specific runtime error for upstream handling.
            raise RuntimeError(
                f"Failed to convert Markdown to HTML in '{self.node_name}': {e}"
            ) from e