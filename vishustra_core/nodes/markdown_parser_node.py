import logging
import re
from typing import Any, Dict

# Assuming vishustra_core.nodes.base_node is available in the Python path
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class MarkdownParserNode(BaseNode):
    """
    A Vishustra processing node that parses Markdown content.

    This node takes a string of Markdown, performs a simulated parsing
    to extract plain text and basic metadata, and returns a structured
    dictionary representation. This simulation does not use a full-fledged
    Markdown parser library but rather demonstrates common extraction patterns.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "MarkdownParser"

    def _strip_markdown_syntax(self, markdown_text: str) -> str:
        """
        A very basic simulation of stripping common Markdown syntax to get plain text.
        This method uses regular expressions to remove common Markdown elements.
        It is not a full-fledged Markdown parser and might not handle all edge cases
        or complex nested structures perfectly.
        """
        # Remove headers (e.g., # Header, ## Subheader)
        text = re.sub(r'#+\s*(.*)', r'\1', markdown_text)
        # Remove bold/italic markers (e.g., **bold**, _italic_)
        text = re.sub(r'(\*\*|__)(.*?)\1', r'\2', text)  # Bold
        text = re.sub(r'(\*|_)(.*?)\1', r'\2', text)    # Italic
        # Remove links (e.g., [text](url) -> text)
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
        # Remove images (e.g., ![alt text](url) -> alt text)
        text = re.sub(r'!\[(.*?)\]\(.*?\)', r'\1', text)
        # Remove code blocks (multiline)
        text = re.sub(r'`{3}[\s\S]*?`{3}', '', text, flags=re.DOTALL)
        # Remove inline code (e.g., `code`)
        text = re.sub(r'`(.*?)`', r'\1', text)
        # Remove blockquotes (e.g., > Quote)
        text = re.sub(r'^\s*>\s*', '', text, flags=re.MULTILINE)
        # Remove list markers (unordered: -, *, +; ordered: 1., 2.)
        text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
        # Replace multiple newlines with single ones, then strip leading/trailing whitespace
        text = re.sub(r'\n{2,}', '\n', text)
        text = text.strip()
        return text

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, expecting Markdown content as a string.

        This method simulates parsing Markdown by extracting a plain text
        representation and calculating basic structural metadata, such as
        the number of headers, links, and code blocks.

        Args:
            data (Any): The input data, expected to be a string containing Markdown.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       relevant to the current processing pipeline.
                                       This node does not directly use context for parsing.

        Returns:
            Any: A dictionary containing:
                 - `original_markdown`: The input Markdown string.
                 - `plain_text`: A simulated plain text version of the Markdown.
                 - `metadata`: A dictionary with extracted metrics like character count,
                               word count, number of headers, links, images, code blocks,
                               and list items.
                 - `parsed_structure_simulated`: A boolean flag indicating completion.

        Raises:
            TypeError: If the input `data` is not a string.
        """
        if not isinstance(data, str):
            logger.error(
                "[%s] Invalid input data type. Expected 'str' for Markdown content, got '%s'.",
                self.node_name,
                type(data).__name__,
            )
            raise TypeError(
                f"{self.node_name} expects input 'data' to be a string (Markdown content), "
                f"but received {type(data).__name__}."
            )

        logger.info("[%s] Starting Markdown content parsing process.", self.node_name)

        original_markdown = data
        plain_text = self._strip_markdown_syntax(original_markdown)

        # Simulate extracting basic metadata from the original markdown
        num_headers = len(re.findall(r'^\s*#+\s.*$', original_markdown, re.MULTILINE))
        num_links = len(re.findall(r'\[.*?\]\(.*?\)', original_markdown))
        num_images = len(re.findall(r'!\[.*?\]\(.*?\)', original_markdown))
        num_code_blocks = len(re.findall(r'`{3}[\s\S]*?`{3}', original_markdown, re.DOTALL))
        num_list_items = len(re.findall(r'^\s*[-*+]\s|^\s*\d+\.\s', original_markdown, re.MULTILINE))

        result = {
            "original_markdown": original_markdown,
            "plain_text": plain_text,
            "metadata": {
                "num_characters": len(original_markdown),
                "num_words": len(plain_text.split()) if plain_text else 0,
                "num_headers": num_headers,
                "num_links": num_links,
                "num_images": num_images,
                "num_code_blocks": num_code_blocks,
                "num_list_items": num_list_items,
            },
            "parsed_structure_simulated": True  # Flag to indicate processing occurred
        }

        logger.info(
            "[%s] Markdown parsing completed. Extracted %d characters and %d words (plain text).",
            self.node_name,
            result["metadata"]["num_characters"],
            result["metadata"]["num_words"]
        )
        
        return result