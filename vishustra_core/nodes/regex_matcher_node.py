import re
import logging
from typing import Any, Dict, List, Optional, Union
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class RegexMatcherNode(BaseNode):
    """
    A processing node designed to perform regular expression matching, 
    extraction, and validation on textual input data.
    
    This node expects the regex pattern and optional flags to be provided 
    within the execution context.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the canonical name of the node.
        """
        return "RegexMatcherNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the regex matching logic against the input data.

        Args:
            data (Any): The input data to process. Expected to be a string.
            context (Dict[str, Any]): The orchestration context. 
                Expected keys: 
                - 'regex_pattern': The string pattern to compile.
                - 'regex_flags': (Optional) Integer flags for the re module.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - 'matches': A list of all found patterns.
                - 'groups': Captured named groups if present in the pattern.
                - 'is_match': Boolean indicating if at least one match was found.

        Raises:
            TypeError: If input data is not a string.
            ValueError: If the regex pattern is missing or invalid.
        """
        pattern: Optional[str] = context.get("regex_pattern")
        flags: int = context.get("regex_flags", 0)

        if not pattern:
            logger.error(f"[{self.node_name}] Execution failed: 'regex_pattern' not found in context.")
            raise ValueError("Context must provide 'regex_pattern' for RegexMatcherNode.")

        if not isinstance(data, str):
            logger.error(f"[{self.node_name}] Input type mismatch. Expected str, got {type(data).__name__}.")
            raise TypeError(f"RegexMatcherNode requires string input, but received {type(data).__name__}.")

        try:
            compiled_regex = re.compile(pattern, flags=flags)
            
            # Extract all matches
            matches: List[Any] = compiled_regex.findall(data)
            
            # Handle named groups for structured extraction
            search_result = compiled_regex.search(data)
            named_groups: Dict[str, Any] = search_result.groupdict() if search_result else {}

            result = {
                "matches": matches,
                "groups": named_groups,
                "is_match": len(matches) > 0
            }

            logger.debug(f"[{self.node_name}] Successfully processed text. Matches found: {len(matches)}")
            return result

        except re.error as e:
            logger.exception(f"[{self.node_name}] Failed to compile or execute regex pattern: {pattern}")
            raise ValueError(f"Invalid regex pattern provided: {str(e)}")
        except Exception as e:
            logger.exception(f"[{self.node_name}] An unexpected error occurred during processing.")
            raise RuntimeError(f"Regex processing error: {str(e)}") from e