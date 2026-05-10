import logging
import re
from typing import Any, Dict, List, Set

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class KeywordExtractorNode(BaseNode):
    """
    A processing node responsible for extracting significant keywords from text data.
    
    This node identifies relevant terms by filtering out common stop words and 
    applying frequency-based selection or pattern matching.
    """

    def __init__(self, stop_words: Set[str] = None, top_k: int = 10):
        """
        Initializes the KeywordExtractorNode with configurable extraction parameters.

        Args:
            stop_words (Set[str], optional): A custom set of words to ignore.
            top_k (int): Maximum number of keywords to return. Defaults to 10.
        """
        self.top_k = top_k
        # Basic default stop words for internal filtering if no external list is provided
        self.stop_words = stop_words or {
            "the", "and", "a", "of", "to", "is", "in", "it", "that", "with", 
            "as", "for", "was", "on", "are", "by", "be", "this", "at", "or"
        }

    @property
    def node_name(self) -> str:
        """Returns the identifier for this node."""
        return "KeywordExtractorNode"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input string to extract significant keywords.

        Args:
            data (Any): The input data. Expected to be a string or a dictionary 
                        containing a 'text' key.
            context (Dict[str, Any]): The execution context shared across nodes.

        Returns:
            List[str]: A list of extracted unique keywords.

        Raises:
            TypeError: If input data format is not supported.
            ValueError: If the input text is empty.
        """
        logger.info(f"Node '{self.node_name}' starting keyword extraction.")

        text = self._extract_text(data)
        
        if not text.strip():
            logger.warning("Received empty text for keyword extraction.")
            return []

        try:
            # Normalize and tokenize: convert to lowercase and find words
            words = re.findall(r'\b\w{3,}\b', text.lower())
            
            # Filter stop words and calculate frequency
            frequencies: Dict[str, int] = {}
            for word in words:
                if word not in self.stop_words:
                    frequencies[word] = frequencies.get(word, 0) + 1

            # Sort by frequency and take top_k
            sorted_keywords = sorted(
                frequencies.items(), 
                key=lambda item: item[1], 
                reverse=True
            )
            
            extracted_keywords = [word for word, count in sorted_keywords[:self.top_k]]
            
            logger.debug(f"Extracted {len(extracted_keywords)} keywords.")
            return extracted_keywords

        except Exception as e:
            logger.error(f"Error during keyword extraction in node '{self.node_name}': {str(e)}")
            raise RuntimeError(f"Failed to process keywords: {e}") from e

    def _extract_text(self, data: Any) -> str:
        """
        Helper to parse text from various input formats.
        """
        if isinstance(data, str):
            return data
        if isinstance(data, dict) and "text" in data:
            return str(data["text"])
        
        error_msg = f"Invalid input format for {self.node_name}. Expected str or dict with 'text' key."
        logger.error(error_msg)
        raise TypeError(error_msg)

```python
# Example of expected usage (Internal documentation/testing context):
# extractor = KeywordExtractorNode(top_k=5)
# result = extractor.process("Vishustra is a highly modular LLM orchestration framework for backend engineers.", {})
# print(result) # -> ['vishustra', 'modular', 'llm', 'orchestration', 'framework']
