import re
import logging
from typing import Any, Dict, List, Optional, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class RegexMatcherNode(BaseNode):
    """
    A node designed to perform regex-based pattern matching and extraction.
    
    This node expects a string as input data and a regex pattern defined 
    within the context. It returns a list of dictionaries containing 
    the matched groups or full matches.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the unique identifier for the Regex Matcher node.
        """
        return "RegexMatcherNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes regex matching on the provided input data.

        Args:
            data (Any): The input string to be processed.
            context (Dict[str, Any]): The orchestration context. 
                Required keys: 
                    - 'regex_pattern': The raw regex string.
                Optional keys:
                    - 'regex_flags': Integer flags for the re module (e.g., re.IGNORECASE).

        Returns:
            Dict[str, Any]: A dictionary containing 'matches' (list of strings/dicts) 
                            and 'match_count' (int).

        Raises:
            TypeError: If input data is not a string.
            ValueError: If regex_pattern is missing or invalid.
        """
        if not isinstance(data, str):
            logger.error(f"Invalid data type received: {type(data)}. Expected str.")
            raise TypeError(f"{self.node_name} requires a string input.")

        pattern_str: Optional[str] = context.get("regex_pattern")
        if not pattern_str:
            logger.error("No 'regex_pattern' found in context.")
            raise ValueError("RegexMatcherNode requires 'regex_pattern' in context.")

        flags: int = context.get("regex_flags", 0)

        try:
            compiled_pattern: re.Pattern = re.compile(pattern_str, flags)
        except re.error as e:
            logger.exception(f"Failed to compile regex pattern: {pattern_str}")
            raise ValueError(f"Invalid regex pattern: {e}")

        logger.debug(f"Executing regex search on data of length {len(data)}")
        
        matches: List[Union[str, Dict[str, str]]] = []
        
        # Iterating through all matches
        for match in compiled_pattern.finditer(data):
            # If named groups exist, prioritize returning the group dict
            group_dict = match.groupdict()
            if group_dict:
                matches.append(group_dict)
            else:
                # Fallback to the whole matched string
                matches.append(match.group())

        result = {
            "matches": matches,
            "match_count": len(matches),
            "pattern_used": pattern_str
        }

        logger.info(f"Node {self.node_name} completed. Found {result['match_count']} matches.")
        return result
        
    def __repr__(self) -> str:
        return f"<{self.node_name}()>"