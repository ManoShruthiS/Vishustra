
import logging
import re
from typing import Any, Dict

# Assuming BaseNode is available at this path within the Vishustra framework
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class MarkdownParserNode(BaseNode):
    """
    A Vishustra processing node designed to parse Markdown content.

    This node takes a Markdown string as input and transforms it into
    a structured dictionary containing simulated HTML and plain text representations.
    It demonstrates a foundational content transformation capability within the framework.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "MarkdownParser"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data, interpreting it as Markdown, and converts it into
        simulated HTML and a stripped plain text format.

        This implementation provides a basic, rule-based simulation of Markdown parsing
        to illustrate data transformation. It handles headers, lists, bold/italic text,
        and links.

        Args:
            data: The input data, expected to be a string containing Markdown text.
            context: A dictionary containing contextual information relevant to the
                     current orchestration run. This node does not currently utilize
                     the context, but it is provided for future extensibility.

        Returns:
            A dictionary containing the following keys:
            - 'html_content': A string representing the simulated HTML version of the Markdown.
            - 'plain_text_content': A string representing a stripped plain text version.
            - 'original_markdown': The original input Markdown string.

        Raises:
            ValueError: If the input data is not a string.
            Exception: For any unexpected errors encountered during the parsing process.
        """
        logger.debug(f"[{self.node_name}] Starting Markdown parsing for input data.")

        if not isinstance(data, str):
            logger.error(f"[{self.node_name}] Invalid input data type. Expected 'str', but received '{type(data).__name__}'.")
            raise ValueError(
                f"Input data for '{self.node_name}' must be a string containing Markdown. "
                f"Received type: {type(data).__name__}."
            )

        markdown_input = data
        html_output_lines = []
        plain_text_output_lines = []
        
        # State tracking for block elements (e.g., paragraphs, lists)
        in_paragraph = False
        in_list = False

        try:
            lines = markdown_input.split('\n')
            
            for line_num, line in enumerate(lines):
                stripped_line = line.strip()

                if not stripped_line:
                    # An empty line often implies a paragraph break or end of a block
                    if in_paragraph:
                        html_output_lines.append("</p>")
                        in_paragraph = False
                    if in_list: # End of list if followed by empty line
                        html_output_lines.append("</ul>")
                        in_list = False
                    plain_text_output_lines.append("")
                    continue

                # --- HTML Simulation ---
                current_html_line = stripped_line
                current_plain_text_line = stripped_line # Start with stripped line for plain text

                # 1. Headers (e.g., # H1, ## H2)
                header_match = re.match(r"^(#+)\s*(.*)", current_html_line)
                if header_match:
                    if in_paragraph:
                        html_output_lines.append("</p>")
                        in_paragraph = False
                    if in_list:
                        html_output_lines.append("</ul>")
                        in_list = False

                    level = len(header_match.group(1))
                    header_text = header_match.group(2).strip()
                    html_output_lines.append(f"<h{level}>{header_text}</h{level}>")
                    plain_text_output_lines.append(header_text) # For plain text, just the header text
                    continue # This line is fully processed as a header

                # 2. List Items (e.g., - Item, * Item)
                list_item_match = re.match(r"^\s*[-*+]\s+(.*)", current_html_line)
                if list_item_match:
                    if in_paragraph:
                        html_output_lines.append("</p>")
                        in_paragraph = False
                    
                    if not in_list:
                        html_output_lines.append("<ul>")
                        in_list = True
                    
                    item_text = list_item_match.group(1).strip()
                    html_output_lines.append(f"<li>{item_text}</li>")
                    plain_text_output_lines.append(f"- {item_text}") # For plain text, keep bullet
                    continue # This line is fully processed as a list item

                # 3. Default to Paragraph if no other block element matched
                if not in_paragraph and not in_list: # Only start new paragraph if not in a list
                    html_output_lines.append("<p>")
                    in_paragraph = True
                elif in_list: # If inside a list, a non-list item line could still be content
                    # A more sophisticated parser would handle paragraph breaks within list items.
                    # For this simulation, we consider a non-list item line breaks the list.
                    html_output_lines.append("</ul>")
                    in_list = False
                    html_output_lines.append("<p>") # Start new paragraph for the current line
                    in_paragraph = True


                # 4. Inline Formatting (applied to current_html_line and current_plain_text_line)
                
                # HTML: Bold (**text** -> <strong>text</strong>)
                current_html_line = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', current_html_line)
                # HTML: Italics (*text* -> <em>text</em>)
                current_html_line = re.sub(r'\*(.*?)\*', r'<em>\1</em>', current_html_line)
                # HTML: Links ([text](url) -> <a href="url">text</a>)
                current_html_line = re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\2">\1</a>', current_html_line)
                
                html_output_lines.append(current_html_line)

                # Plain Text: Remove bold/italic markers
                current_plain_text_line = re.sub(r'\*\*|__|\*|_', '', current_plain_text_line)
                # Plain Text: Extract link text only (remove URL part)
                current_plain_text_line = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', current_plain_text_line)
                plain_text_output_lines.append(current_plain_text_line)
            
            # Ensure all open tags are closed at the end of the document
            if in_paragraph:
                html_output_lines.append("</p>")
            if in_list:
                html_output_lines.append("</ul>")

            final_html_content = "\n".join(html_output_lines).strip()
            final_plain_text_content = "\n".join(plain_text_output_lines).strip()

            logger.info(f"[{self.node_name}] Successfully parsed Markdown content into simulated HTML and plain text.")
            return {
                "html_content": final_html_content,
                "plain_text_content": final_plain_text_content,
                "original_markdown": markdown_input
            }

        except Exception as e:
            logger.exception(f"[{self.node_name}] An unexpected error occurred during Markdown parsing.")
            raise # Re-raise the exception after logging for upstream handling

