import logging
import re
from typing import Any, Dict

# Assuming vishustra_core is available in the project's Python environment.
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class MarkdownParserNode(BaseNode):
    """
    A processing node designed to parse Markdown text and transform it into 
    a simplified HTML representation.
    
    This node demonstrates the capability of Vishustra to handle text-based
    transformations by simulating the parsing of common Markdown elements
    such as headers, bold text, italics, and basic list items. The output
    is a string containing HTML-like tags.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this processing node.
        """
        return "MarkdownParser"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, expecting a Markdown string, and returns
        a simplified HTML string.
        
        Args:
            data (Any): The input data, which must be a string containing Markdown.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the processing node. While not directly
                                       utilized in this initial implementation,
                                       it's available for future enhancements such
                                       as parser configuration or dynamic rules.
        
        Returns:
            Any: A string representing the simplified HTML output derived from
                 the input Markdown.
            
        Raises:
            TypeError: If the input `data` is not a string, as this node
                       specifically expects text-based Markdown content.
            Exception: For any unexpected errors encountered during the parsing
                       and transformation process.
        """
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input type. Expected 'str', "
                f"but received '{type(data).__name__}'."
            )
            raise TypeError(
                f"MarkdownParserNode expects 'data' to be a string, "
                f"but received {type(data).__name__}."
            )

        markdown_text: str = data
        parsed_html_lines = []

        try:
            lines = markdown_text.split('\n')
            
            for line in lines:
                processed_line = line
                
                # --- Block-level parsing (order matters) ---
                
                # Handle Headers (H1-H6)
                header_match = re.match(r"^(#{1,6})\s*(.*)", processed_line)
                if header_match:
                    level = len(header_match.group(1))
                    content = header_match.group(2).strip()
                    parsed_html_lines.append(f"<h{level}>{content}</h{level}>")
                    continue
                
                # Handle unordered list items (simple simulation)
                list_item_match = re.match(r"^\s*[-*+]\s+(.*)", processed_line)
                if list_item_match:
                    content = list_item_match.group(1).strip()
                    # Note: A full Markdown parser would wrap these <li> elements
                    # within <ul> or <ol> tags. For this simplified demonstration,
                    # we output individual list items, expecting subsequent processing
                    # or the consumer to handle block structuring if needed.
                    parsed_html_lines.append(f"<li>{content}</li>")
                    continue

                # --- Inline-level parsing (within remaining lines) ---

                # Bold: **text**
                processed_line = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", processed_line)
                # Italic: *text* (ensure it doesn't conflict with bold before/after)
                processed_line = re.sub(r"\*(.*?)\*", r"<em>\1</em>", processed_line)
                
                # Treat remaining non-empty lines as paragraphs
                if processed_line.strip(): 
                    parsed_html_lines.append(f"<p>{processed_line.strip()}</p>")
                
            # Combine processed lines into a single HTML string
            result_html = "\n".join(parsed_html_lines)
            
            logger.info(f"[{self.node_name}] Successfully parsed Markdown data into HTML.")
            return result_html

        except Exception as e:
            logger.error(
                f"[{self.node_name}] An unexpected error occurred during Markdown parsing: {e}", 
                exc_info=True
            )
            # Re-raise the exception to propagate the error up the orchestration chain.
            raise