import logging
import re
from typing import Any, Dict, List, Optional
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class MarkdownParserNode(BaseNode):
    """
    MarkdownParserNode processes raw Markdown strings and transforms them into 
    a structured dictionary format. It segments content by headers and identifies 
    key Markdown elements for downstream LLM context injection or data routing.
    """

    @property
    def node_name(self) -> str:
        """Returns the unique identifier for this node type."""
        return "MarkdownParserNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parses the input Markdown text.
        
        Args:
            data: Expected to be a string containing Markdown content.
            context: Shared execution context for the pipeline.
            
        Returns:
            A dictionary containing the original text, extracted sections, 
            and basic statistics.
            
        Raises:
            ValueError: If the input data is not a string.
            Exception: For unexpected parsing failures.
        """
        if not isinstance(data, str):
            logger.error(f"[{self.node_name}] Invalid input type: {type(data)}. Expected string.")
            raise ValueError(f"{self.node_name} expects a string input.")

        try:
            logger.info(f"[{self.node_name}] Initializing parse sequence for payload (size: {len(data)} chars).")
            
            sections = self._parse_sections(data)
            code_blocks = self._extract_code_blocks(data)
            
            result = {
                "document_structure": sections,
                "extracted_code": code_blocks,
                "metadata": {
                    "total_sections": len(sections),
                    "contains_code": len(code_blocks) > 0,
                    "source_ref": context.get("source_id", "unknown")
                }
            }
            
            logger.debug(f"[{self.node_name}] Successfully identified {len(sections)} structural sections.")
            return result

        except Exception as e:
            logger.error(f"[{self.node_name}] Critical failure during markdown parsing: {str(e)}", exc_info=True)
            raise

    def _parse_sections(self, text: str) -> List[Dict[str, Any]]:
        """
        Segments the text into sections based on Markdown headers (H1-H6).
        """
        sections = []
        # Regex to capture header level and title
        header_regex = re.compile(r'^(#{1,6})\s+(.*)$', re.MULTILINE)
        
        matches = list(header_regex.finditer(text))
        
        if not matches:
            return [{"level": 0, "title": "Body", "content": text.strip()}]
            
        for i, match in enumerate(matches):
            level = len(match.group(1))
            title = match.group(2).strip()
            
            start_index = match.end()
            end_index = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[start_index:end_index].strip()
            
            sections.append({
                "level": level,
                "title": title,
                "content": content
            })
            
        return sections

    def _extract_code_blocks(self, text: str) -> List[Dict[str, str]]:
        """
        Extracts fenced code blocks including the language identifier if present.
        """
        code_blocks = []
        # Regex for fenced code blocks (```language ... ```)
        code_regex = re.compile(r'```(\w*)\n([\s\S]*?)\n```')
        
        for match in code_regex.finditer(text):
            language = match.group(1).lower() or "text"
            code_content = match.group(2)
            code_blocks.append({
                "language": language,
                "content": code_content
            })
            
        return code_blocks