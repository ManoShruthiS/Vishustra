import logging
from typing import Any, Dict, List, Set

# Assuming vishustra_core.nodes.base_node exists in the project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class KeywordExtractorNode(BaseNode):
    """
    A Vishustra node designed to extract a predefined set of keywords from input text data.

    This node performs a simple, case-insensitive check for the presence of specified
    keywords within the input string. In a production environment, this might be
    backed by more sophisticated NLP techniques (e.g., tokenization, stemming,
    lemmatization, or library-based keyword extraction).
    """

    def __init__(self, keywords_to_extract: List[str] = None):
        """
        Initializes the KeywordExtractorNode with an optional list of keywords to look for.

        Args:
            keywords_to_extract: An optional list of strings representing the keywords
                                 this node should attempt to extract. If None,
                                 a default set of general LLM/orchestration-related
                                 keywords will be used. All keywords are converted
                                 to lowercase for case-insensitive matching.
        """
        if keywords_to_extract is None:
            self._keywords_to_extract: Set[str] = {
                "llm", "ai", "orchestration", "framework", "vishustra",
                "node", "data", "processing", "model", "text", "information",
                "component", "system", "engine", "flow"
            }
        else:
            # Ensure keywords are stored in lowercase for efficient case-insensitive matching
            self._keywords_to_extract = {k.lower() for k in keywords_to_extract}
        logger.debug(f"{self.node_name} initialized with {len(self._keywords_to_extract)} keywords.")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "KeywordExtractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to extract predefined keywords.

        The method expects the input `data` to be a string. It performs a case-insensitive
        substring check for each configured keyword.

        Args:
            data: The input data, expected to be a string containing text to analyze.
            context: A dictionary containing contextual information for processing.
                     This node does not currently utilize the context for keyword extraction
                     but receives it as part of the standard `BaseNode` interface.

        Returns:
            A sorted list of unique keywords found in the input text. If no keywords
            are found, an empty list is returned.

        Raises:
            ValueError: If the input 'data' is not a string, indicating an invalid
                        input type for this node's operation.
        """
        if not isinstance(data, str):
            error_msg = (
                f"Invalid input data type for '{self.node_name}'. Expected 'str', "
                f"but received '{type(data).__name__}'. Data: {data}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        text_lower = data.lower()
        extracted_keywords: Set[str] = set()

        for keyword in self._keywords_to_extract:
            # Perform a simple substring presence check.
            # For robust keyword extraction, this would typically involve
            # NLP libraries (e.g., NLTK, spaCy) for tokenization,
            # lemmatization, and more sophisticated matching.
            if keyword in text_lower:
                extracted_keywords.add(keyword)
        
        found_count = len(extracted_keywords)
        if found_count > 0:
            logger.info(f"Node '{self.node_name}' successfully processed data. Found {found_count} keywords.")
            logger.debug(f"Extracted keywords: {sorted(list(extracted_keywords))}")
        else:
            logger.info(f"Node '{self.node_name}' processed data but found no matching keywords.")

        return sorted(list(extracted_keywords))