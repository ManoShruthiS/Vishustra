import logging
import re
from typing import Any, Dict, List, Union
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class MarkdownParserNode(BaseNode):
    """
    A specialized node within the Vishustra framework designed to parse 
    Markdown strings into structured dictionary objects.
    
    This node extracts headers, code blocks, links, and provides a 
    sanitized version of the text for downstream consumption by LLMs 
    or vector databases.
    """

    @property
    def node_name(self) -> str:
        """Returns the unique identifier for this node type."""
        return "MarkdownParserNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parses raw Markdown data and returns a structured representation.
        
        Args:
            data: The input, expected to be a Markdown-formatted string.
            context: The execution context containing pipeline metadata.

        Returns:
            A dictionary containing the original content, extracted elements, 
            and metadata.

        Raises:
            TypeError: If the input data is not a string.
            RuntimeError: If an unexpected error occurs during transformation.
        """
        if not isinstance(data, str):
            logger.error(f"MarkdownParserNode received invalid data type: {type(data)}")
            raise TypeError(f"MarkdownParserNode requires a string input, got {type(data)}.")

        try:
            logger.info("Initializing markdown parsing sequence.")
            
            # Encapsulate parsed results in a structured schema
            structured_output = {
                "raw_content": data,
                "node_metadata": {
                    "source_node": self.node_name,
                    "length": len(data)
                },
                "parsed_elements": {
                    "headers": self._extract_headers(data),
                    "code_blocks": self._extract_code_blocks(data),
                    "links": self._extract_links(data),
                    "plain_text": self._strip_markdown_syntax(data)
                }
            }
            
            logger.debug(
                f"Parsing complete. Found {len(structured_output['parsed_elements']['headers'])} headers "
                f"and {len(structured_output['parsed_elements']['code_blocks'])} code blocks."
            )
            
            return structured_output

        except Exception as e:
            logger.exception("Critial failure during Markdown parsing logic.")
            raise RuntimeError(f"Failed to process Markdown node: {str(e)}") from e

    def _extract_headers(self, text: str) -> List[Dict[str, Union[int, str]]]:
        """Identifies and categorizes Markdown headers (H1-H6)."""
        headers = []
        # Match lines starting with #
        pattern = r'^(#{1,6})\s+(.*)$'
        for match in re.finditer(pattern, text, re.MULTILINE):
            headers.append({
                "level": len(match.group(1)),
                "content": match.group(2).strip()
            })
        return headers

    def _extract_code_blocks(self, text: str) -> List[Dict[str, str]]:
        """Extracts fenced code blocks and identifies the language tag."""
        blocks = []
        # Match ```lang ... ``` blocks
        pattern = r'```(\w*)\n([\s\S]*?)\n```'
        for match in re.finditer(pattern, text):
            blocks.append({
                "language": match.group(1) if match.group(1) else "unspecified",
                "code": match.group(2).strip()
            })
        return blocks

    def _extract_links(self, text: str) -> List[Dict[str, str]]:
        """Extracts all Markdown hyperlinks."""
        links = []
        # Match [text](url)
        pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        for match in re.finditer(pattern, text):
            links.append({
                "anchor_text": match.group(1),
                "url": match.group(2)
            })
        return links

    def _strip_markdown_syntax(self, text: str) -> str:
        """
        Provides a heuristic-based cleanup to extract human-readable text 
        without common Markdown decorators.
        """
        # Remove headers
        content = re.sub(r'#+\s+', '', text)
        # Convert links to plain text anchor
        content = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', content)
        # Remove code blocks entirely for the plain text view
        content = re.sub(r'```[\s\S]*?```', '', content)
        # Remove bold/italic markers
        content = re.sub(r'(\*\*|__|\*|_)', '', content)
        
        return content.strip()