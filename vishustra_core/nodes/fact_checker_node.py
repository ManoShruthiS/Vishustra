import logging
from typing import Any, Dict, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class FactCheckerNode(BaseNode):
    """
    A processing node designed to simulate fact-checking a given statement.

    It uses an internal knowledge base to determine the factuality of a statement
    and provides a confidence score and reason for its determination.
    """

    # Internal knowledge base for demonstration purposes.
    # In a real-world scenario, this would interface with external fact-checking APIs
    # or a more sophisticated knowledge graph.
    _predefined_facts: Dict[str, bool] = {
        "the earth is flat": False,
        "water boils at 100 degrees celsius at sea level": True,
        "the sun rises in the west": False,
        "python is a programming language": True,
        "cats are canines": False,
        "vishustra is an llm orchestration framework": True,
        "the capital of france is paris": True,
        "bananas are berries": True, # A common surprising fact
        "tomatoes are vegetables": False # Biologically fruits, legally vegetables sometimes
    }

    def __init__(self):
        """
        Initializes the FactCheckerNode.
        """
        logger.info("FactCheckerNode initialized, ready to verify statements.")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "FactChecker"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Union[str, bool, float, None]]:
        """
        Processes the input data to determine the factuality of a statement.

        The `data` input is expected to be either:
        - A string representing the statement to be checked.
        - A dictionary containing a 'statement' key whose value is the string to check.

        The `context` dictionary can optionally contain keys for external fact-checking
        services, but for this simulation, it's primarily for logging context.

        Args:
            data: The statement to fact-check, as a string or a dict.
            context: A dictionary containing contextual information for the process.

        Returns:
            A dictionary containing:
            - 'statement': The original statement that was checked.
            - 'is_factual': True, False, or None if factuality cannot be determined.
            - 'reason': A textual explanation for the determination.
            - 'confidence': A float between 0.0 and 1.0 indicating confidence.

        Raises:
            ValueError: If the input data is not a string or a dict with a 'statement' key.
        """
        statement_to_check: str
        if isinstance(data, str):
            statement_to_check = data
        elif isinstance(data, dict) and "statement" in data:
            statement_to_check = str(data["statement"])
        else:
            logger.error("Invalid input data type for FactCheckerNode. Expected string or dict with 'statement' key.")
            raise ValueError("FactCheckerNode requires a string statement or a dict with a 'statement' key.")

        if not statement_to_check.strip():
            logger.warning("Received an empty or whitespace-only statement for fact-checking.")
            return {
                "statement": statement_to_check,
                "is_factual": None,
                "reason": "Empty statement provided for fact-checking.",
                "confidence": 0.0
            }

        logger.info(f"Attempting to fact-check statement: '{statement_to_check}'")
        logger.debug(f"Current context for FactCheckerNode: {context}")

        is_factual: bool | None = None
        reason: str = "Could not verify factuality based on available knowledge."
        confidence: float = 0.0

        # Normalize the statement for case-insensitive lookup
        normalized_statement = statement_to_check.strip().lower()

        if normalized_statement in self._predefined_facts:
            is_factual = self._predefined_facts[normalized_statement]
            reason = f"Statement matched internal knowledge base as {('true' if is_factual else 'false')}."
            confidence = 0.95 # High confidence for direct internal match
        else:
            # In a real system, this would trigger an external API call,
            # a more complex NLP model, or a deeper database query.
            # For simulation, we'll mark it as uncheckable with low confidence.
            logger.info(f"Statement '{statement_to_check}' not found in internal knowledge base. "
                        "Further external verification would be required.")
            reason = "Statement not found in direct knowledge base. External verification needed."
            confidence = 0.1 # Low confidence as it's an unknown fact

        result = {
            "statement": statement_to_check,
            "is_factual": is_factual,
            "reason": reason,
            "confidence": confidence
        }
        logger.debug(f"Fact-checking completed for '{statement_to_check}'. Result: {result}")
        return result
