import logging
import re
from typing import Any, Dict, List

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class MarkdownParserNode(BaseNode):
    """
    A Vishustra processing node that parses Markdown text into a structured list of elements.
    It identifies and extracts block-level elements like headers and paragraphs, and
    also extracts inline elements such as links from within the text.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "markdown_parser"

    def process(self, data: Any, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Processes the input data, expecting a Markdown string, and returns a list of
        dictionaries representing the parsed Markdown elements.

        This node performs a light-weight parsing to extract key structural and
        informational elements.

        Supported elements and their structure in the output list:
        - Headers:   `{'type': 'header', 'level': int, 'text': str}`
                     e.g., `{'type': 'header', 'level': 1, 'text': 'Main Title'}`
        - Links:     `{'type': 'link', 'text': str, 'url': str}`
                     e.g., `{'type': 'link', 'text': 'Google', 'url': 'https://google.com'}`
                     (Links are extracted from paragraphs and added as distinct elements).
        - Paragraphs: `{'type': 'paragraph', 'text': str}`
                     e.g., `{'type': 'paragraph', 'text': 'This is some text.'}`

        Args:
            data: The input Markdown string to be parsed.
            context: A dictionary containing contextual information for the node's operation.
                     (Not directly used by this specific node, but available for extensions).

        Returns:
            A `List` of `Dict`s, where each dictionary represents a parsed Markdown element.

        Raises:
            TypeError: If the input `data` is not a string, indicating an invalid input type.
        """
        logger.info("MarkdownParserNode received data for processing.")

        if not isinstance(data, str):
            logger.error(
                f"Invalid input type for MarkdownParserNode. Expected 'str', "
                f"but received '{type(data).__name__}'. Aborting process."
            )
            raise TypeError(
                f"MarkdownParserNode expects 'str' input, but received '{type(data).__name__}'."
            )

        parsed_elements: List[Dict[str, Any]] = []
        lines = data.split('\n')

        # Regex patterns for common Markdown elements
        header_pattern = re.compile(r"^(#{1,6})\s+(.*)$") # Matches H1-H6 headers
        link_pattern = re.compile(r"\[([^\]]+?)\]\((.+?)\)") # Matches [text](url) links

        for line_num, line in enumerate(lines, start=1):
            stripped_line = line.strip()

            if not stripped_line:
                # Skip entirely empty or whitespace-only lines
                continue

            # Attempt to parse headers
            header_match = header_pattern.match(stripped_line)
            if header_match:
                level = len(header_match.group(1))
                text = header_match.group(2).strip()
                parsed_elements.append({'type': 'header', 'level': level, 'text': text})
                logger.debug(f"L{line_num}: Parsed header (level {level}): '{text}'")
            else:
                # If not a header, treat the line as a potential paragraph.
                # The raw text of the line is kept for the paragraph.
                parsed_elements.append({'type': 'paragraph', 'text': stripped_line})
                logger.debug(f"L{line_num}: Parsed paragraph: '{stripped_line}'")

                # Additionally, extract any links found within this paragraph line
                for link_match in link_pattern.finditer(stripped_line):
                    link_text = link_match.group(1)
                    link_url = link_match.group(2)
                    parsed_elements.append({'type': 'link', 'text': link_text, 'url': link_url})
                    logger.debug(f"L{line_num}: Extracted link: text='{link_text}', url='{link_url}'")

        logger.info(f"MarkdownParserNode completed processing. Total {len(parsed_elements)} elements parsed.")
        return parsed_elements
