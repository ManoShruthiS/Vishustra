import logging
from typing import Any, Dict, List, Set

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class KeywordExtractorNode(BaseNode):
    """
    A Vishustra processing node that extracts keywords from a given text.
    This node simulates keyword extraction by searching for a predefined
    or context-provided set of keywords within the input data.
    """

    def __init__(self):
        """
        Initializes the KeywordExtractorNode with a default set of keywords
        to search for if none are provided in the processing context.
        """
        self._default_keywords: Set[str] = {
            "vishustra", "llm", "orchestration", "framework", "node",
            "processing", "data", "engineer", "backend", "python",
            "module", "component", "workflow"
        }
        logger.debug(f"KeywordExtractorNode initialized with default keywords: {', '.join(sorted(list(self._default_keywords)))}")

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this node.
        """
        return "KeywordExtractor"

    def process(self, data: Any, context: Dict[str, Any]) -> List[str]:
        """
        Processes the input data to extract relevant keywords.

        The `data` is expected to be a string (text). The method will look for
        keywords within this text.

        The `context` dictionary can optionally contain a 'keywords_to_extract'
        key, which should be a list or set of strings. If provided, these
        keywords will be used instead of the node's default set.

        Args:
            data (Any): The input data to process. Expected to be a string.
            context (Dict[str, Any]): A dictionary containing contextual
                                      information or configuration for the node.

        Returns:
            List[str]: A sorted list of unique keywords found in the input text.

        Raises:
            ValueError: If the input `data` is not a string.
        """
        node_id = context.get('node_id', self.node_name)
        logger.info(f"[{node_id}] Starting keyword extraction process.")

        if not isinstance(data, str):
            logger.error(f"[{node_id}] Invalid input data type. Expected 'str', but received '{type(data).__name__}'.")
            raise ValueError(f"{self.node_name} expects string input, but received {type(data).__name__}.")

        extracted_keywords: Set[str] = set()
        text_lower = data.lower()

        keywords_to_search_lower: Set[str] = set()
        if 'keywords_to_extract' in context:
            context_keywords = context['keywords_to_extract']
            if isinstance(context_keywords, (list, set)):
                try:
                    keywords_to_search_lower = set(str(k).lower() for k in context_keywords if isinstance(k, str))
                    logger.debug(f"[{node_id}] Using {len(keywords_to_search_lower)} keywords provided in context.")
                except Exception as e:
                    logger.warning(
                        f"[{node_id}] Could not properly parse 'keywords_to_extract' from context ({e}). "
                        "Falling back to node's default keywords."
                    )
                    keywords_to_search_lower = self._default_keywords
            else:
                logger.warning(
                    f"[{node_id}] 'keywords_to_extract' in context is not a list or set "
                    f"(type: {type(context_keywords).__name__}). Falling back to node's default keywords."
                )
                keywords_to_search_lower = self._default_keywords
        else:
            keywords_to_search_lower = self._default_keywords
            logger.debug(f"[{node_id}] Using node's default {len(self._default_keywords)} keywords.")

        if not keywords_to_search_lower:
            logger.warning(f"[{node_id}] No keywords specified for extraction. Returning empty list.")
            return []

        # Simulate keyword extraction by checking for substring presence
        for keyword_candidate_lower in keywords_to_search_lower:
            if keyword_candidate_lower in text_lower:
                extracted_keywords.add(keyword_candidate_lower)

        result = sorted(list(extracted_keywords))
        logger.info(f"[{node_id}] Finished keyword extraction. Found {len(result)} keywords.")
        return result
