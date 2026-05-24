import logging
import re
from typing import Any, Dict, List

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class FactCheckerNode(BaseNode):
    """
    A Vishustra node that performs simulated fact-checking on input text.

    This node identifies potential factual statements within the provided content
    and attempts to verify them against a mock internal knowledge base.
    In a production environment, this would integrate with real-time fact-checking
    APIs, structured knowledge bases, or advanced NLP models.
    """

    # A mock internal knowledge base for demonstration purposes.
    # In a real system, this would be a sophisticated, dynamically updated source.
    _MOCK_KNOWLEDGE_BASE = {
        "The Earth is round": True,
        "Water boils at 100 degrees Celsius at sea level": True,
        "Humans have three hearts": False,
        "The moon is made of cheese": False,
        "Python is a statically typed language": False,
        "Python is a dynamically typed language": True,
        "Vishustra is an LLM orchestration framework": True,
        "The capital of France is Paris": True,
        "The sun revolves around the Earth": False,
        "All birds can fly": False,
    }

    def __init__(self):
        """
        Initializes the FactCheckerNode.
        No external connections or heavy resource loading are performed here for this simulation.
        """
        logger.info(f"[{self.node_name}] Node initialized.")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "FactCheckerNode"

    def _extract_statements(self, text: str) -> List[str]:
        """
        Extracts potential factual statements (sentences) from the input text.
        This uses a basic regex-based sentence segmentation. In a real-world
        scenario, a more sophisticated NLP model would be employed for robust
        and accurate statement extraction.

        Args:
            text (str): The input text from which to extract statements.

        Returns:
            List[str]: A list of extracted statements.
        """
        # Split by common sentence delimiters while keeping the delimiter for better context
        # and handling of multiple spaces.
        sentences = re.split(r'(?<=[.!?])\s*', text)
        cleaned_sentences = [s.strip() for s in sentences if s.strip()]
        logger.debug(f"[{self.node_name}] Extracted {len(cleaned_sentences)} statements.")
        return cleaned_sentences

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to perform fact-checking.

        Expects `data` to be either a string (the content to check)
        or a dictionary with a 'content' key containing the string.
        The `context` parameter is available for passing configuration
        or runtime information, though not directly used for this node's
        core logic in this simulation.

        Args:
            data (Any): The input data. Expected to be a string directly,
                        or a dictionary containing a 'content' key with a string value.
            context (Dict[str, Any]): A dictionary for contextual information,
                                      e.g., global configuration or prior node outputs.

        Returns:
            Dict[str, Any]: A dictionary containing the original content and
                            a list of identified facts with their verification status,
                            confidence, and simulated sources.
                            Example structure:
                            {
                                "original_content": "...",
                                "checked_facts": [
                                    {"statement": "...", "is_verified": bool, "confidence": float, "sources": ["..."]}
                                ]
                            }

        Raises:
            ValueError: If the input `data` is not in the expected format.
            RuntimeError: For other unexpected errors during the fact-checking process.
        """
        logger.debug(f"[{self.node_name}] Starting fact-checking process for incoming data.")
        original_content: str = ""

        # Validate and extract content from the input data
        if isinstance(data, str):
            original_content = data
        elif isinstance(data, dict) and "content" in data and isinstance(data["content"], str):
            original_content = data["content"]
        else:
            error_msg = (
                f"[{self.node_name}] Invalid input data format. "
                f"Expected a string or a dictionary with a 'content' key containing a string. "
                f"Received type: {type(data)}, value: {data}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if not original_content.strip():
            logger.warning(f"[{self.node_name}] Received empty or whitespace-only content for fact-checking.")
            return {
                "original_content": original_content,
                "checked_facts": [],
                "warnings": ["Empty content provided for fact-checking."]
            }

        checked_facts_results: List[Dict[str, Any]] = []
        try:
            statements = self._extract_statements(original_content)
            logger.debug(f"[{self.node_name}] Processing {len(statements)} statements for verification.")

            for statement in statements:
                is_verified = False
                confidence = 0.0
                sources: List[str] = []
                best_match_fact = None
                highest_similarity = 0.0

                # Simulate fact checking against the internal mock knowledge base.
                # In a real system, this would involve advanced semantic search,
                # natural language inference, or external API calls.
                for known_fact, truth_value in self._MOCK_KNOWLEDGE_BASE.items():
                    # Simple case-insensitive exact or strong substring match for demo
                    stmt_lower = statement.lower()
                    known_fact_lower = known_fact.lower()

                    if known_fact_lower == stmt_lower: # Exact match
                        best_match_fact = known_fact
                        highest_similarity = 1.0
                        break
                    elif known_fact_lower in stmt_lower: # Known fact is a substring of the statement
                        # Prioritize exact, then longer matches
                        current_similarity = len(known_fact_lower) / len(stmt_lower)
                        if current_similarity > highest_similarity:
                            highest_similarity = current_similarity
                            best_match_fact = known_fact
                    elif stmt_lower in known_fact_lower: # Statement is a substring of a known fact
                        current_similarity = len(stmt_lower) / len(known_fact_lower)
                        if current_similarity > highest_similarity:
                            highest_similarity = current_similarity
                            best_match_fact = known_fact

                if best_match_fact:
                    is_verified = self._MOCK_KNOWLEDGE_BASE[best_match_fact]
                    # Scale confidence based on match quality.
                    # A verified truth gets high confidence, a known falsehood still high but slightly less.
                    base_confidence = 0.9 if is_verified else 0.8
                    confidence = highest_similarity * base_confidence
                    sources.append("Internal Mock Knowledge Base")
                    logger.debug(
                        f"[{self.node_name}] Matched statement '{statement}' to known fact '{best_match_fact}'. "
                        f"Verified: {is_verified}, Confidence: {confidence:.2f}."
                    )
                else:
                    # For statements not found or not strongly matching in our mock KB,
                    # mark as unverified with low confidence.
                    is_verified = False
                    confidence = 0.3  # Default low confidence for unverified statements
                    sources.append("External Data (Simulated/Unverified)")
                    logger.debug(
                        f"[{self.node_name}] Statement '{statement}' not strongly matched in mock KB. "
                        f"Marked as unverified with confidence: {confidence:.2f}."
                    )

                checked_facts_results.append({
                    "statement": statement,
                    "is_verified": is_verified,
                    "confidence": round(confidence, 2), # Round for cleaner output
                    "sources": sources
                })

        except Exception as e:
            error_msg = f"[{self.node_name}] An unexpected error occurred during fact-checking: {e}"
            logger.exception(error_msg)  # Log full traceback for unexpected exceptions
            # Re-raise as a RuntimeError to signal a critical processing failure
            raise RuntimeError(error_msg) from e

        logger.info(f"[{self.node_name}] Fact-checking completed for content of length: {len(original_content)}.")
        return {
            "original_content": original_content,
            "checked_facts": checked_facts_results
        }
