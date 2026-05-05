import logging
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

# Import the markdown parsing library
try:
    from markdown_it import MarkdownIt
    from markdown_it.token import Token
except ImportError:
    # Log an error if the markdown-it-py dependency is missing.
    # This allows the file to be imported, but the node will fail at runtime.
    # In a production environment, dependency checks would typically be more
    # robust at the framework deployment level.
    logging.getLogger(__name__).error(
        "Required dependency 'markdown-it-py' not found. "
        "Please install it using 'pip install markdown-it-py' to enable the MarkdownParserNode."
    )
    # Define dummy classes to prevent NameError during import if the dependency is not met.
    class MarkdownIt: # type: ignore
        def parse(self, text: str) -> List[Any]:
            raise NotImplementedError("markdown-it-py is not installed.")
    class Token: # type: ignore
        pass


logger = logging.getLogger(__name__)

class MarkdownParserNode(BaseNode):
    """
    A processing node designed to parse Markdown text into a structured
    Abstract Syntax Tree (AST) representation.

    This node leverages the markdown-it-py library to convert raw Markdown
    strings into a list of token dictionaries. Each dictionary represents
    a distinct element or event in the Markdown document's structure,
    such as headings, paragraphs, lists, or code blocks. The output
    facilitates programmatic manipulation, analysis, or transformation
    of Markdown content by downstream nodes. Nesting information is
    implicitly provided by `*_open` and `*_close` token types and
    the `level` attribute within each token dictionary.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name for this processing node.
        """
        return "MarkdownParser"

    def _token_to_dict(self, token: Token) -> Dict[str, Any]:
        """
        Converts a markdown-it-py Token object into a standardized dictionary
        format for easier serialization and consumption by other nodes.

        Args:
            token: The markdown-it-py Token object to convert.

        Returns:
            A dictionary representation of the token.
        """
        token_dict = {
            "type": token.type,
            "tag": token.tag,
            "nesting": token.nesting,
            "attrs": token.attrs,
            "map": token.map,
            "level": token.level,
            "content": token.content,
            "markup": token.markup,
            "info": token.info,
            "meta": token.meta,
            "block": token.block,
            "hidden": token.hidden,
            "children": None # Placeholder, markdown-it-py's parse() returns a flat list
        }
        # While markdown-it-py's `parse()` returns a flat list of tokens,
        # individual Token objects *can* theoretically contain children
        # if constructed differently (e.g., by a plugin).
        # This ensures robustness for such cases.
        if token.children:
            token_dict["children"] = [self._token_to_dict(child) for child in token.children]
        return token_dict

    def process(self, data: Any, context: Dict[str, Any]) -> Union[List[Dict[str, Any]], None]:
        """
        Parses an input Markdown string into a list of structured token dictionaries (AST).

        Args:
            data: The input data, expected to be a string containing Markdown.
            context: A dictionary containing contextual information for the processing
                     (e.g., node configuration, session data). Not directly used
                     by this parser but available for future extensions.

        Returns:
            A list of dictionaries representing the Markdown AST, where each dictionary
            corresponds to a token. Nesting information is implied by `*_open`/`*_close`
            token types and the `level` attribute. Returns None if parsing fails
            due to invalid input or an internal error.
        """
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input type. Expected 'str' for Markdown "
                f"parsing, but received '{type(data).__name__}'. Returning None."
            )
            return None

        markdown_text: str = data
        md = MarkdownIt() # Initialize markdown-it parser with default configuration

        logger.debug(f"[{self.node_name}] Starting Markdown parsing for input of length {len(markdown_text)} bytes.")

        try:
            # Parse the Markdown text into a flat list of token objects
            tokens = md.parse(markdown_text)
            logger.debug(f"[{self.node_name}] Successfully generated {len(tokens)} raw tokens from Markdown.")

            # Convert each Token object into a dictionary for a consistent, serializable output.
            parsed_ast = [self._token_to_dict(token) for token in tokens]

            logger.info(
                f"[{self.node_name}] Markdown parsing complete. "
                f"Generated AST with {len(parsed_ast)} root-level token dictionaries."
            )
            return parsed_ast

        except NotImplementedError as e:
            # This specific error is raised if markdown-it-py is not installed
            logger.critical(
                f"[{self.node_name}] Processing failed: {e}. "
                f"Please ensure 'markdown-it-py' is installed. Returning None."
            )
            return None
        except Exception as e:
            # Catch any other unexpected errors during parsing
            logger.error(
                f"[{self.node_name}] An unexpected error occurred during Markdown parsing: {e}",
                exc_info=True # Include stack trace for detailed debugging
            )
            return None