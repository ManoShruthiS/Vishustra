import re
import logging
from typing import Any, Dict, List, Optional, Union
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class RegexMatcherNode(BaseNode):
    """
    A processing node designed to extract information from strings using regular expressions.
    
    This node can be configured with a default pattern or take a pattern dynamically 
    from the execution context. It supports both single match extraction and 
    finding all occurrences.
    """

    def __init__(self, default_pattern: Optional[str] = None, flags: int = re.IGNORECASE):
        """
        Initializes the RegexMatcherNode.

        :param default_pattern: The default regex pattern to use if none is provided in context.
        :param flags: Regex flags (e.g., re.IGNORECASE, re.MULTILINE).
        """
        self._default_pattern = default_pattern
        self._flags = flags

    @property
    def node_name(self) -> str:
        """Returns the canonical name of this node."""
        return "RegexMatcherNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data by applying a regex pattern.
        
        Expected context keys:
        - 'regex_pattern': (Optional) Override for the compiled regex.
        - 'find_all': (Optional) Boolean to return all matches instead of the first one.

        :param data: The input data, expected to be a string or convertible to string.
        :param context: The orchestration context containing configuration overrides.
        :return: A dictionary containing matches, match count, and success status.
        """
        try:
            # Ensure input data is treatable as a string
            input_text = str(data) if data is not None else ""
            
            # Resolve pattern from context or fall back to default
            pattern_str = context.get("regex_pattern", self._default_pattern)
            find_all = context.get("find_all", True)

            if not pattern_str:
                logger.warning(f"[{self.node_name}] No regex pattern provided in context or initialization.")
                return {
                    "matches": [],
                    "count": 0,
                    "status": "skipped"
                }

            # Attempt to compile and execute search
            regex = re.compile(pattern_str, self._flags)
            
            if find_all:
                matches = regex.findall(input_text)
            else:
                match = regex.search(input_text)
                matches = [match.group()] if match else []

            logger.debug(f"[{self.node_name}] Successfully processed text with pattern: {pattern_str}")
            
            return {
                "matches": matches,
                "count": len(matches),
                "status": "success"
            }

        except re.error as e:
            logger.error(f"[{self.node_name}] Invalid regular expression: {str(e)}")
            return {
                "matches": [],
                "error": "Invalid regex pattern",
                "status": "error"
            }
        except Exception as e:
            logger.exception(f"[{self.node_name}] Unexpected error during processing: {str(e)}")
            return {
                "matches": [],
                "error": str(e),
                "status": "error"
            }