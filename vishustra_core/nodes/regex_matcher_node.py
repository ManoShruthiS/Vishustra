import re
import logging
from typing import Any, Dict, List, Union, Optional
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class RegexMatcherNode(BaseNode):
    """
    A modular node within the Vishustra framework designed for pattern matching.
    It extracts or validates text based on regular expressions provided via
    initialization or runtime context.
    """

    def __init__(self, default_pattern: Optional[str] = None, flags: int = 0):
        """
        Initializes the RegexMatcherNode.

        :param default_pattern: The fallback regex pattern if none is provided in context.
        :param flags: Regex flags (e.g., re.IGNORECASE, re.MULTILINE).
        """
        self._default_pattern = default_pattern
        self._flags = flags

    @property
    def node_name(self) -> str:
        """Returns the canonical name of the node."""
        return "RegexMatcherNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data by applying a regular expression pattern.

        Expected 'data' format: str (the text to search).
        Expected 'context' keys (optional): 
            - 'regex_pattern': Overrides the default pattern.
            - 'return_full_match': Boolean, if True returns Match objects as strings.
        """
        try:
            # Pattern resolution: Context takes precedence over constructor default
            pattern_str = context.get("regex_pattern", self._default_pattern)
            
            if not pattern_str:
                logger.error("No regex pattern provided in context or constructor.")
                raise ValueError("RegexMatcherNode requires a valid pattern.")

            if not isinstance(data, str):
                logger.warning(f"Expected string input, received {type(data)}. Attempting conversion.")
                input_text = str(data)
            else:
                input_text = data

            logger.debug(f"Executing regex search with pattern: {pattern_str}")
            
            compiled_re = re.compile(pattern_str, self._flags)
            matches: List[str] = compiled_re.findall(input_text)
            
            is_match = len(matches) > 0
            
            result = {
                "matches": matches,
                "match_count": len(matches),
                "is_match": is_match,
                "metadata": {
                    "pattern_used": pattern_str,
                    "input_length": len(input_text)
                }
            }

            return result

        except re.error as e:
            logger.exception(f"Invalid regular expression encountered: {e}")
            return {
                "error": "InvalidRegexError",
                "message": str(e),
                "is_match": False
            }
        except Exception as e:
            logger.exception(f"Unexpected error during regex processing: {e}")
            return {
                "error": "ProcessingError",
                "message": str(e),
                "is_match": False
            }

    def __repr__(self) -> str:
        return f"<RegexMatcherNode(pattern='{self._default_pattern}')>"