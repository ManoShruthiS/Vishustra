import logging
import re
from typing import Any, Dict, List, Union
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class MarkdownParserNode(BaseNode):
    """
    A specialized node designed to parse Markdown strings into structured data.
    It extracts key components such as headers, links, and code blocks to 
    facilitate downstream LLM processing or indexing.
    """

    @property
    def node_name(self) -> str:
        """Returns the unique identifier for this node type."""
        return "MarkdownParserNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transforms raw Markdown text into a structured dictionary format.

        Args:
            data (Any): The input data, expected to be a Markdown-formatted string.
            context (Dict[str, Any]): Metadata and state information for the current orchestration flow.

        Returns:
            Dict[str, Any]: A dictionary containing headers, links, and the original content length.

        Raises:
            TypeError: If the input data is not a string.
            RuntimeError: If parsing logic encounters an unrecoverable error.
        """
        if not isinstance(data, str):
            logger.error("MarkdownParserNode received non-string input. Type: %s", type(data))
            raise TypeError(f"Input data must be a string, received {type(data).__name__}")

        try:
            logger.debug("Beginning markdown extraction for input of length %d", len(data))
            
            headers = self._extract_headers(data)
            links = self._extract_links(data)
            code_blocks = self._extract_code_blocks(data)

            result = {
                "metadata": {
                    "node": self.node_name,
                    "content_length": len(data),
                    "header_count": len(headers),
                    "link_count": len(links),
                },
                "structured_data": {
                    "headers": headers,
                    "links": links,
                    "code_blocks": code_blocks
                },
                "raw_content": data
            }

            logger.info("Successfully parsed markdown content.")
            return result

        except Exception as e:
            logger.exception("An error occurred during markdown parsing.")
            raise RuntimeError(f"MarkdownParserNode failed to process data: {str(e)}") from e

    def _extract_headers(self, text: str) -> List[Dict[str, Union[int, str]]]:
        """Identifies and extracts markdown headers (h1-h6)."""
        headers = []
        # Matches # Header, ## Header, etc.
        header_pattern = re.compile(r'^(#{1,6})\s+(.*)$', re.MULTILINE)
        for match in header_pattern.finditer(text):
            headers.append({
                "level": len(match.group(1)),
                "text": match.group(2).strip()
            })
        return headers

    def _extract_links(self, text: str) -> List[Dict[str, str]]:
        """Identifies and extracts markdown links [label](url)."""
        links = []
        # Matches [Label](URL)
        link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
        for match in link_pattern.finditer(text):
            links.append({
                "label": match.group(1),
                "url": match.group(2)
            })
        return links

    def _extract_code_blocks(self, text: str) -> List[Dict[str, str]]:
        """Identifies and extracts fenced code blocks."""
        blocks = []
        # Matches ```lang ... ```
        code_pattern = re.compile(r'```(\w+)?\n([\s\S]*?)\n```', re.MULTILINE)
        for match in code_pattern.finditer(text):
            blocks.append({
                "language": match.group(1) or "plain_text",
                "content": match.group(2).strip()
            })
        return blocks