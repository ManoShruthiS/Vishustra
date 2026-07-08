import logging
from typing import Any, Dict, List
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class FactCheckerNode(BaseNode):
    """
    A processing node designed to simulate fact-checking of textual statements.

    This node takes a statement as input and compares it against a simplistic,
    internal "database" of known facts to determine its veracity. In a production
    environment, this would integrate with external fact-checking APIs, knowledge
    bases, or more sophisticated verification models.
    """

    def __init__(self):
        """
        Initializes the FactCheckerNode with a predefined set of known truths
        and falsehoods for demonstration purposes.
        """
        # In a real-world application, these would be loaded from a persistent
        # store, an external knowledge graph, or configured via a constructor
        # parameter or a dependency injection mechanism.
        self._known_truths: List[str] = [
            "The Earth revolves around the Sun.",
            "Water is H2O.",
            "Oxygen is a gas.",
            "The capital of France is Paris.",
            "Jupiter is the largest planet in our solar system."
        ]
        self._known_falsehoods: List[str] = [
            "The Earth is flat.",
            "The sky is green.",
            "Birds are mammals.",
            "Humans can photosynthesize.",
            "The moon is made of cheese."
        ]
        logger.debug("FactCheckerNode initialized with internal fact database for simulation.")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "FactChecker"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to fact-check a given statement.

        The `data` input is expected to be a dictionary containing at least
        a 'statement' key, whose value is the string to be checked.

        Args:
            data: A dictionary containing the statement to be fact-checked.
                  Expected format: `{"statement": "The statement to check."}`
            context: A dictionary containing workflow-specific context or
                     configuration parameters. This simulation does not directly
                     utilize the context for fact-checking logic but it's
                     available for future extensions (e.g., passing API keys).

        Returns:
            A dictionary containing the original statement, its derived veracity
            status, and a brief explanation.
            Example:
            ```json
            {
                "original_statement": "The Earth is flat.",
                "fact_check_result": "FALSE",
                "explanation": "This statement contradicts established scientific facts."
            }
            ```

        Raises:
            ValueError: If the input data is not a dictionary, or if the
                        'statement' key is missing, not a string, or empty.
            RuntimeError: If an unexpected error occurs during the fact-checking
                          logic execution.
        """
        if not isinstance(data, dict):
            logger.error("Invalid input data type for FactCheckerNode. Expected 'dict', received '%s'.", type(data))
            raise ValueError("FactCheckerNode expects 'data' to be a dictionary.")

        statement_raw = data.get("statement")
        if not isinstance(statement_raw, str) or not statement_raw.strip():
            logger.error(
                "Missing or invalid 'statement' key in input data. Expected a non-empty string. Data: %s",
                data
            )
            raise ValueError(
                "FactCheckerNode requires a non-empty 'statement' string "
                "in the input data dictionary."
            )

        statement = statement_raw.strip()
        result: Dict[str, Any] = {
            "original_statement": statement,
            "fact_check_result": "UNVERIFIED",
            "explanation": "Could not conclusively verify or refute the statement based on known facts."
        }

        try:
            if statement in self._known_truths:
                result["fact_check_result"] = "TRUE"
                result["explanation"] = "This statement is known to be factually accurate."
                logger.info("Statement '%s' fact-checked as TRUE.", statement)
            elif statement in self._known_falsehoods:
                result["fact_check_result"] = "FALSE"
                result["explanation"] = "This statement is known to be factually inaccurate."
                logger.info("Statement '%s' fact-checked as FALSE.", statement)
            else:
                logger.info("Statement '%s' remains UNVERIFIED.", statement)

        except Exception as e:
            # Catching broad exceptions to ensure robustness for unforeseen issues
            logger.exception(
                "An unexpected error occurred during fact-checking for statement: '%s'.",
                statement
            )
            raise RuntimeError(
                f"FactCheckerNode failed to process statement '{statement}' due to an internal error."
            ) from e

        return result