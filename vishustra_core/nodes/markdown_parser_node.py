import logging
import re
from typing import Any, Dict, List, Optional
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class MarkdownParserNode(BaseNode):
    """
    A processing node designed to parse raw Markdown strings into structured data objects.
    It decomposes the document into headers, content blocks, and optionally extracts
    frontmatter metadata.
    """

    def __init__(self):
        # Regex to capture Markdown headers (H1 through H6)
        self._header_pattern = re.compile(r"^(#{1,6})\s+(.*)$")
        # Regex to capture YAML-style frontmatter
        self._frontmatter_pattern = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)

    @property
    def node_name(self) -> str:
        """Returns the unique identifier for this node."""
        return "MarkdownParser"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parses the input markdown string and returns a structured dictionary.
        
        Args:
            data: The raw markdown string to be processed.
            context: Execution context containing shared state and configuration.
            
        Returns:
            A dictionary containing parsed metadata and a list of section objects.
            
        Raises:
            TypeError: If the input data is not a string.
            ValueError: If the input string is empty.
        """
        if not isinstance(data, str):
            logger.error("MarkdownParserNode received non-string input type: %s", type(data).__name__)
            raise TypeError(f"MarkdownParserNode requires 'str' input, received '{type(data).__name__}'")

        if not data.strip():
            logger.warning("MarkdownParserNode received an empty string.")
            return {"metadata": {}, "sections": [], "raw": ""}

        try:
            logger.debug("Starting markdown parsing sequence.")
            
            # 1. Extract Frontmatter if present
            metadata, content_body = self._extract_frontmatter(data)
            
            # 2. Parse sections by headers
            sections = self._parse_sections(content_body)

            result = {
                "metadata": metadata,
                "sections": sections,
                "content_length": len(data),
                "node_execution": self.node_name
            }

            logger.info("Successfully parsed markdown into %d sections.", len(sections))
            return result

        except Exception as e:
            logger.exception("An unexpected error occurred during markdown parsing.")
            raise RuntimeError(f"Failed to process node '{self.node_name}': {str(e)}") from e

    def _extract_frontmatter(self, text: str) -> tuple[Dict[str, Any], str]:
        """Extracts YAML frontmatter and returns (metadata_dict, remaining_text)."""
        match = self._frontmatter_pattern.match(text)
        if match:
            raw_yaml = match.group(1)
            content_body = text[match.end():]
            # Simple manual split for frontmatter to avoid external YAML dependencies
            metadata = {}
            for line in raw_yaml.splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    metadata[key.strip()] = value.strip()
            return metadata, content_body
        return {}, text

    def _parse_sections(self, text: str) -> List[Dict[str, str]]:
        """Splits the markdown body into sections based on headers."""
        sections = []
        lines = text.splitlines()
        
        current_header = "Introduction"
        current_level = 0
        current_lines: List[str] = []

        for line in lines:
            header_match = self._header_pattern.match(line)
            if header_match:
                # Save previous section if it exists
                if current_lines or current_header:
                    sections.append({
                        "header": current_header,
                        "level": current_level,
                        "content": "\n".join(current_lines).strip()
                    })
                
                # Reset for new section
                current_level = len(header_match.group(1))
                current_header = header_match.group(2).strip()
                current_lines = []
            else:
                current_lines.append(line)

        # Append the final section
        if current_lines or current_header:
            sections.append({
                "header": current_header,
                "level": current_level,
                "content": "\n".join(current_lines).strip()
            })

        return sections

def _validate_node():
    """Internal validation for local testing/debugging."""
    parser = MarkdownParserNode()
    sample = "---\ntitle: test\n---\n# Hello\nWorld"
    try:
        output = parser.process(sample, {})
        assert "sections" in output
        assert output["sections"][0]["header"] == "Hello"
    except Exception as e:
        logger.error(f"Node validation failed: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _validate_node()