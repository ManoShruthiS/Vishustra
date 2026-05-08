import re
import logging
from typing import Any, Dict, List, Optional, Union
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class RegexMatcherNode(BaseNode):
    """
    A processing node designed to identify and extract patterns from text using 
    Regular Expressions. This node is critical for validating LLM outputs, 
    detecting PII, or extracting specific tokens from unstructured data.
    """

    def __init__(self, pattern: Optional[str] = None, flags: int = 0):
        """
        Initializes the RegexMatcherNode.
        
        :param pattern: A default regex pattern to use if none is provided in the context.
        :param flags: Regex flags (e.g., re.IGNORECASE, re.MULTILINE).
        """
        self._pattern = pattern
        self._flags = flags

    @property
    def node_name(self) -> str:
        """Returns the canonical name of the node."""
        return "RegexMatcherNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data by applying a regex pattern match.
        
        :param data: The input data, expected to be a string or a type convertible to string.
        :param context: Orchestration context which can override 'pattern' and 'flags'.
        :return: A dictionary containing matches, the original text, and match status.
        :raises ValueError: If no pattern is provided or if data cannot be processed.
        """
        pattern_str = context.get("pattern", self._pattern)
        flags = context.get("flags", self._flags)

        if not pattern_str:
            logger.error("RegexMatcherNode: No pattern provided in constructor or context.")
            raise ValueError("Regex pattern must be specified.")

        if data is None:
            logger.warning("RegexMatcherNode: Received null input data.")
            return {"matches": [], "found": False, "data": data}

        text_to_process = str(data)

        try:
            logger.debug(f"Applying regex pattern: {pattern_str}")
            regex = re.compile(pattern_str, flags)
            
            # Find all matches (returns tuples if multiple groups are defined)
            matches: List[Union[str, tuple]] = regex.findall(text_to_process)
            
            result = {
                "matches": matches,
                "found": len(matches) > 0,
                "match_count": len(matches),
                "original_input": text_to_process
            }
            
            logger.info(f"RegexMatcherNode: Found {len(matches)} matches.")
            return result

        except re.error as e:
            logger.exception(f"RegexMatcherNode: Invalid regex pattern '{pattern_str}'.")
            raise ValueError(f"Failed to compile or execute regex: {str(e)}")
        except Exception as e:
            logger.exception("RegexMatcherNode: An unexpected error occurred during processing.")
            raise e

# End of file vishustra_core/nodes/regex_matcher_node.py