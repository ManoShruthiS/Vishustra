import logging
from typing import Any, Dict, List, Union

# Assuming BaseNode is located here as per project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class FactCheckerNode(BaseNode):
    """
    A processing node designed to simulate fact-checking a given statement
    against a set of known truths and falsehoods provided within the context.

    This node aims to determine if a statement is factual, false, or unverified
    based on the available knowledge base.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this processing node."""
        return "FactChecker"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Union[str, bool, float, None]]:
        """
        Processes the input data to fact-check a statement.

        Expected `data` format:
        A dictionary containing at least a "statement" key with a string value.
        Example:
        ```python
        {"statement": "The capital of France is Paris."}
        ```

        Expected `context` format:
        A dictionary that may contain:
        - "known_facts": A list of strings representing statements considered true.
        - "known_falsehoods": A list of strings representing statements considered false.
        Example:
        ```python
        {
            "known_facts": ["The capital of France is Paris.", "Water boils at 100 degrees Celsius at sea level."],
            "known_falsehoods": ["Elephants can fly.", "The Earth is flat."]
        }
        ```

        Returns a dictionary containing the fact-checking outcome:
        - "original_statement": The statement that was checked.
        - "is_factual": `True` if verified as factual, `False` if verified as false,
                        `None` if the statement could not be definitively verified or disproven.
        - "fact_check_details": A string providing more information on the outcome.
        - "confidence": A float (0.0 to 1.0) indicating the confidence in the result.

        Raises:
            ValueError: If `data` is not a dictionary or does not contain a valid 'statement'.
        """
        if not isinstance(data, dict):
            logger.error("FactCheckerNode received invalid data: expected a dictionary.")
            raise ValueError("Input data must be a dictionary.")

        statement = data.get("statement")
        if not isinstance(statement, str) or not statement.strip():
            logger.error(f"FactCheckerNode received data without a valid 'statement' string. Data: {data}")
            raise ValueError("Input data must contain a non-empty 'statement' string.")

        # Initialize result with a neutral/unverified state
        result: Dict[str, Union[str, bool, float, None]] = {
            "original_statement": statement,
            "is_factual": None,
            "fact_check_details": "Statement could not be explicitly verified or disproven against available knowledge.",
            "confidence": 0.5
        }

        # Normalize the statement for case-insensitive comparison
        normalized_statement = statement.strip().lower()

        # Retrieve known facts from context
        known_facts: List[str] = context.get("known_facts", [])
        if not isinstance(known_facts, list):
            logger.warning("Context 'known_facts' is not a list. Skipping fact-checking against external knowledge.")
            known_facts = []

        # Retrieve known falsehoods from context
        known_falsehoods: List[str] = context.get("known_falsehoods", [])
        if not isinstance(known_falsehoods, list):
            logger.warning("Context 'known_falsehoods' is not a list. Skipping checking against known falsehoods.")
            known_falsehoods = []

        is_true = False
        is_false = False

        # Check against known facts
        for fact in known_facts:
            if not isinstance(fact, str):
                logger.debug(f"Skipping non-string item in 'known_facts': {fact}")
                continue
            if normalized_statement == fact.strip().lower():
                is_true = True
                break

        # Check against known falsehoods if not already found true
        if not is_true:
            for falsehood in known_falsehoods:
                if not isinstance(falsehood, str):
                    logger.debug(f"Skipping non-string item in 'known_falsehoods': {falsehood}")
                    continue
                if normalized_statement == falsehood.strip().lower():
                    is_false = True
                    break

        if is_true:
            result["is_factual"] = True
            result["fact_check_details"] = "Statement aligns with known facts."
            result["confidence"] = 1.0
            logger.info(f"Statement '{statement}' was verified as factual.")
        elif is_false:
            result["is_factual"] = False
            result["fact_check_details"] = "Statement contradicts known facts/is a known falsehood."
            result["confidence"] = 1.0
            logger.info(f"Statement '{statement}' was verified as false.")
        else:
            # If neither true nor false, it remains unverified with default details
            logger.info(f"Statement '{statement}' could not be definitively verified or disproven against available knowledge.")
            
        return result
