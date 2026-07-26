import re
import logging
from typing import Any, Dict, List

# Assuming vishustra_core.nodes.base_node is available in the environment
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class MarkdownParserNode(BaseNode):
    """
    A Vishustra node designed to parse Markdown text into a structured list of blocks.

    This node simulates the parsing of markdown by identifying common elements like
    headings and paragraphs, representing them as a list of dictionaries for
    subsequent processing by other nodes in the orchestration framework.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node.
        """
        return "MarkdownParser"

    def process(self, data: Any, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Processes the input data, expecting a Markdown string, and parses it
        into a structured list of blocks (headings, paragraphs).

        The `context` dictionary is available for passing operational data across
        nodes but is not directly utilized in this node's basic parsing logic.

        Args:
            data: The input data, expected to be a string containing Markdown content.
            context: A dictionary containing contextual information for processing.

        Returns:
            A list of dictionaries. Each dictionary represents a parsed block and
            contains at least a 'type' (e.g., 'heading', 'paragraph') and 'content'.
            Heading blocks will additionally include a 'level' attribute (integer).

        Raises:
            ValueError: If the input data is not a string, indicating an invalid
                        payload for this parser node.
        """
        if not isinstance(data, str):
            error_msg = (
                f"Invalid input data type for {self.node_name}. "
                f"Expected str, but received {type(data).__name__}."
            )
            logger.error(
                error_msg,
                extra={"node_name": self.node_name, "input_type": type(data).__name__, "context": context},
            )
            raise ValueError(error_msg)

        parsed_blocks: List[Dict[str, Any]] = []
        lines = data.strip().split('\n')
        current_paragraph_lines: List[str] = []

        def flush_paragraph() -> None:
            """
            Helper function to consolidate accumulated paragraph lines into a
            single paragraph block and add it to `parsed_blocks`.
            """
            if current_paragraph_lines:
                paragraph_content = '\n'.join(current_paragraph_lines).strip()
                if paragraph_content:  # Only add if content is not empty after stripping
                    parsed_blocks.append({"type": "paragraph", "content": paragraph_content})
                current_paragraph_lines.clear()

        for line in lines:
            line = line.strip()
            if not line:
                # An empty line often demarcates paragraphs or blocks
                flush_paragraph()
                continue

            # Attempt to match markdown headings (e.g., '# Heading', '## Subheading')
            heading_match = re.match(r"^(#+)\s*(.*)", line)
            if heading_match:
                flush_paragraph()  # Any preceding paragraph must be finalized before a new heading
                level = len(heading_match.group(1))
                content = heading_match.group(2).strip()
                parsed_blocks.append({"type": "heading", "level": level, "content": content})
            else:
                # If it's not a heading and not an empty line, it's part of a paragraph
                current_paragraph_lines.append(line)

        # Ensure any remaining accumulated paragraph content is flushed after the loop
        flush_paragraph()

        logger.debug(
            f"Successfully parsed markdown data into {len(parsed_blocks)} blocks.",
            extra={"node_name": self.node_name, "num_blocks": len(parsed_blocks)},
        )
        return parsed_blocks
