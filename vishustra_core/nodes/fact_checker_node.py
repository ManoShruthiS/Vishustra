import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class FactCheckerNode(BaseNode):
    """
    A processing node that simulates fact-checking an input statement against
    a predefined knowledge base.

    This node expects the input `data` to be a dictionary containing a 'statement' key.
    It returns a dictionary with the original statement, fact-checking status, and
    supporting evidence (or lack thereof).
    """

    def __init__(self):
        """
        Initializes the FactCheckerNode with a mock knowledge base.
        In a real-world scenario, this would interact with external fact-checking
        APIs, databases, or sophisticated NLP models.
        """
        self._knowledge_base = {
            "The capital of France is Paris.": {"status": "VERIFIED", "evidence": "Paris is the official capital city of France."},
            "Water boils at 100 degrees Celsius at sea level.": {"status": "VERIFIED", "evidence": "This is a standard scientific fact under normal atmospheric pressure."},
            "Humans can fly naturally.": {"status": "REFUTED", "evidence": "Humans lack the biological adaptations for natural flight, requiring technology to fly."},
            "The moon is made of cheese.": {"status": "REFUTED", "evidence": "The moon is composed primarily of silicate rocks and metals, not cheese."},
            "The Earth is flat.": {"status": "REFUTED", "evidence": "Scientific evidence overwhelmingly supports the Earth being an oblate spheroid."}
        }
        logger.info("FactCheckerNode initialized with mock knowledge base.")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "FactCheckerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to perform a simulated fact-check.

        Args:
            data (Any): The input data, expected to be a dictionary with a 'statement' key.
                        Example: `{'statement': 'The capital of France is Paris.'}`
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the current processing flow.

        Returns:
            Dict[str, Any]: A dictionary containing the fact-checking result:
                            - 'original_statement': The statement that was checked.
                            - 'is_fact_checked': True if the node attempted a check, False on error.
                            - 'status': 'VERIFIED', 'REFUTED', 'UNVERIFIED', or 'ERROR'.
                            - 'evidence': A string explaining the status or an error message.
        Raises:
            ValueError: If the input data does not conform to the expected format.
        """
        if not isinstance(data, dict) or 'statement' not in data:
            logger.error(
                f"FactCheckerNode received invalid data format. "
                f"Expected dict with 'statement' key, got: {type(data).__name__}"
            )
            raise ValueError(
                "FactCheckerNode requires input `data` to be a dictionary "
                "containing a 'statement' key."
            )

        statement_to_check = data['statement']
        logger.debug(f"Attempting to fact-check statement: '{statement_to_check}'")

        try:
            result = self._knowledge_base.get(
                statement_to_check,
                {"status": "UNVERIFIED", "evidence": "Could not find a direct match in the knowledge base."}
            )

            return {
                "original_statement": statement_to_check,
                "is_fact_checked": True,
                "status": result['status'],
                "evidence": result['evidence']
            }
        except Exception as e:
            logger.exception(
                f"An unexpected error occurred during fact-checking for statement "
                f"'{statement_to_check}': {e}"
            )
            return {
                "original_statement": statement_to_check,
                "is_fact_checked": False,
                "status": "ERROR",
                "evidence": f"An internal error prevented full fact verification: {e}"
            }
