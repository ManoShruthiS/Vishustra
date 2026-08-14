import logging
import re
from typing import Any, Dict

# Assuming BaseNode is available at this path as per Vishustra project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class MarkdownParserNode(BaseNode):
    """
    A Vishustra processing node designed to parse Markdown content.

    This node accepts a string containing Markdown, simulates its parsing,
    and extracts structured information such as headers and bold text.
    It provides a transformed version of the text and a summary of identified elements.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "MarkdownParserNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, expecting Markdown content as a string.

        This method attempts to identify and extract common Markdown elements like
        headers and bold text. It returns a dictionary containing the original content,
        a subtly transformed text, and structured lists of identified elements.

        Args:
            data (Any): The input data, which must be a string containing Markdown.
            context (Dict[str, Any]): A dictionary for contextual information,
                                       not directly used in this basic parsing simulation.

        Returns:
            Any: A dictionary containing:
                 - 'original_content': The input Markdown string.
                 - 'parsed_text': A string where simple Markdown headers are converted
                                  to basic HTML-like tags (e.g., # Header -> <h1>Header</h1>).
                 - 'headers': A list of dictionaries, each representing a header
                              with its level and text.
                 - 'bold_texts': A list of strings, each being content identified as bold.
                 - 'summary': A string summarizing the parsing outcome.

        Raises:
            TypeError: If the input `data` is not a string.
        """
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input type. Expected str, "
                f"got {type(data).__name__}."
            )
            raise TypeError(
                f"[{self.node_name}] Input 'data' must be a string. "
                f"Received {type(data).__name__}."
            )

        if not data.strip():
            logger.warning(f"[{self.node_name}] Received empty or whitespace-only markdown content.")
            return {
                "original_content": data,
                "parsed_text": "",
                "headers": [],
                "bold_texts": [],
                "summary": "Empty markdown processed."
            }

        parsed_headers = []
        parsed_bold_texts = []
        transformed_lines = []

        # Regular expression to find Markdown headers (e.g., # Header, ## Subheader)
        header_pattern = re.compile(r"^(#+)\s*(.*)$", re.MULTILINE)
        # Regular expression to find bold text (e.g., **bold** or __bold__)
        bold_pattern = re.compile(r"(\*\*|__)(.*?)\1")

        # --- Simulate header parsing and transformation ---
        for line in data.splitlines():
            header_match = header_pattern.match(line)
            if header_match:
                level_hashes, content = header_match.groups()
                level = len(level_hashes)
                header_text = content.strip()
                parsed_headers.append({"level": level, "text": header_text})
                logger.debug(f"[{self.node_name}] Identified H{level}: '{header_text}'")
                # Transform header lines into a simplified HTML-like representation
                transformed_lines.append(f"<h{level}>{header_text}</h{level}>")
            else:
                transformed_lines.append(line) # Keep non-header lines as is

        # --- Simulate bold text parsing ---
        # We iterate over the original data for bold texts as they can be inline
        for match in bold_pattern.finditer(data):
            # Group 2 contains the actual bold text
            bold_text = match.group(2).strip()
            if bold_text: # Ensure not to add empty strings from e.g., "** **"
                parsed_bold_texts.append(bold_text)
                logger.debug(f"[{self.node_name}] Identified bold text: '{bold_text}'")

        processed_text = "\n".join(transformed_lines)

        summary_msg = (
            f"[{self.node_name}] Successfully processed markdown. "
            f"Found {len(parsed_headers)} header(s) and {len(parsed_bold_texts)} bold text instance(s)."
        )
        logger.info(summary_msg)

        # Return a structured dictionary representing the parsed outcome
        return {
            "original_content": data,
            "parsed_text": processed_text,
            "headers": parsed_headers,
            "bold_texts": parsed_bold_texts,
            "summary": summary_msg
        }
