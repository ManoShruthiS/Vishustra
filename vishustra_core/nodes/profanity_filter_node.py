import logging
import re
from typing import Any, Dict, List, Set, Optional
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ProfanityFilterNode(BaseNode):
    """
    A specialized node within the Vishustra framework designed to identify and 
    sanitize profane or inappropriate language from textual data.
    
    This node acts as a safety layer for LLM pipelines, ensuring that either 
    incoming user prompts or outgoing model responses adhere to content guidelines.
    """

    def __init__(
        self, 
        custom_blacklist: Optional[List[str]] = None, 
        replacement_symbol: str = "****"
    ):
        """
        Initializes the ProfanityFilterNode.
        
        Args:
            custom_blacklist: An optional list of specific strings to filter.
            replacement_symbol: The string used to mask detected profanity.
        """
        self._replacement = replacement_symbol
        # In a production environment, this would likely load from a configuration 
        # or an external dictionary file.
        self._blacklist: Set[str] = {
            "profanity1", "profanity2", "slur_example", "toxic_term"
        }
        
        if custom_blacklist:
            self._blacklist.update(item.lower() for item in custom_blacklist)
            
        self._compile_regex()

    def _compile_regex(self) -> None:
        """
        Compiles the regex pattern for efficient matching across large datasets.
        """
        if not self._blacklist:
            self._pattern = None
            return
            
        # Create a boundary-aware regex pattern to avoid partial matches inside valid words
        pattern_str = r'\b(' + '|'.join(re.escape(word) for word in self._blacklist) + r')\b'
        self._pattern = re.compile(pattern_str, re.IGNORECASE)

    @property
    def node_name(self) -> str:
        """
        Returns the unique identifier for this node type.
        """
        return "ProfanityFilterNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to remove blacklisted terms.
        
        Args:
            data: Expected to be a string or a dictionary containing a 'text' key.
            context: Shared execution context for the pipeline.
            
        Returns:
            The sanitized version of the input data.
        """
        try:
            if isinstance(data, str):
                return self._sanitize(data)
            
            if isinstance(data, dict):
                # If data is a dictionary, we attempt to sanitize the 'text' or 'content' fields
                return self._process_dict(data)

            logger.debug(f"[{self.node_name}] Passing through data of type {type(data)} without modification.")
            return data

        except Exception as e:
            logger.error(f"[{self.node_name}] Execution failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Node {self.node_name} failed to process data.") from e

    def _sanitize(self, text: str) -> str:
        """
        Internal logic to perform regex substitution on raw strings.
        """
        if not text or not self._pattern:
            return text
        
        sanitized_text = self._pattern.sub(self._replacement, text)
        if sanitized_text != text:
            logger.info(f"[{self.node_name}] Content sanitized: matches found and replaced.")
            
        return sanitized_text

    def _process_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively searches for text fields in a dictionary to sanitize.
        """
        new_data = data.copy()
        target_keys = {"text", "content", "message", "body"}
        
        for key, value in new_data.items():
            if key in target_keys and isinstance(value, str):
                new_data[key] = self._sanitize(value)
            elif isinstance(value, dict):
                new_data[key] = self._process_dict(value)
                
        return new_data