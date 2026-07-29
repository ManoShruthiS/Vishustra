import logging
from typing import Any, Dict, Union

# Assuming vishustra_core.nodes.base_node is available as a module
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class FactCheckerNode(BaseNode):
    """
    A Vishustra processing node designed to simulate fact-checking of input statements.

    This node ingests a statement, either as a direct string or embedded within a
    dictionary under a designated key (e.g., 'statement'). It then attempts to
    simulate verification against a mock knowledge base. The primary output is
    a structured dictionary detailing the verification status and a brief
    explanation, alongside the original input and any encountered errors.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the unique, descriptive name of this processing node.
        """
        return "FactChecker"

    def process(self, data: Union[str, Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to simulate fact-checking a given statement.

        The node extracts a statement from the input `data` and attempts to
        "verify" it against a simple, internal mock knowledge base. In a
        production environment, this would typically involve integration with
        external fact-checking APIs, knowledge graphs, or sophisticated LLM-based
        verification mechanisms.

        Args:
            data: The input data containing the statement to be fact-checked.
                  This can be:
                  - A `str`: The statement itself.
                  - A `Dict[str, Any]`: Expected to contain the statement under
                    the key 'statement'.
            context: A dictionary providing contextual information for the node's
                     operation. While not directly used in this simulated version,
                     it could hold configurations like API endpoints or credentials
                     for a real-world implementation.

        Returns:
            A `Dict[str, Any]` containing the structured result of the fact-check:
            - 'original_data': The data initially passed to the node.
            - 'statement_to_check': The extracted string statement that was processed.
            - 'fact_check_status': A string indicating the verification outcome:
                                   'verified_true', 'verified_false', 'unverified', or 'error'.
            - 'reasoning': A brief explanatory message regarding the status.
            - 'error': An error message string if an issue occurred during processing,
                       otherwise `None`.
        """
        statement_to_check: Union[str, None] = None
        result: Dict[str, Any] = {
            "original_data": data,
            "statement_to_check": None,
            "fact_check_status": "error", # Default to error until processing starts
            "reasoning": "Failed to initiate fact-checking due to an unexpected error.",
            "error": None
        }

        try:
            # Step 1: Extract the statement from the input data
            if isinstance(data, str):
                statement_to_check = data
            elif isinstance(data, dict) and 'statement' in data and isinstance(data['statement'], str):
                statement_to_check = data['statement']
            else:
                log_msg = (
                    f"FactCheckerNode received unexpected data format. "
                    f"Expected 'str' or 'dict' with a 'statement' key. "
                    f"Received type: {type(data)}."
                )
                logger.warning(log_msg)
                result['reasoning'] = "Invalid input data format. Statement could not be extracted."
                result['error'] = "Input data must be a string or a dictionary with a 'statement' key."
                result['fact_check_status'] = "unverified" # Mark as unverified rather than error if input malformed
                return result

            result['statement_to_check'] = statement_to_check

            # Step 2: Simulate fact-checking against a mock knowledge base
            # In a real scenario, this would be an API call or a complex logic
            mock_knowledge_base = {
                "The sun is a star.": "verified_true",
                "Birds can fly.": "verified_true",
                "The Earth is flat.": "verified_false",
                "Humans have eight fingers.": "verified_false",
                "All cats love water.": "unverified_complex", # Represents a nuanced or generally false statement needing context
                "The capital of France is Paris.": "verified_true",
                "Elephants can jump.": "verified_false",
            }

            normalized_statement = statement_to_check.strip()

            if normalized_statement in mock_knowledge_base:
                status_from_db = mock_knowledge_base[normalized_statement]
                result['fact_check_status'] = status_from_db
                if status_from_db == "verified_true":
                    result['reasoning'] = "Statement verified as true based on mock knowledge."
                elif status_from_db == "verified_false":
                    result['reasoning'] = "Statement verified as false based on mock knowledge."
                else: # e.g., "unverified_complex"
                    result['fact_check_status'] = "unverified" # Map specific mock status to general 'unverified'
                    result['reasoning'] = "Statement found in mock knowledge, but requires deeper analysis or is context-dependent."
                logger.debug(
                    f"Statement '{normalized_statement}' status: {result['fact_check_status']} "
                    f"(Reason: {result['reasoning']})"
                )
            else:
                result['fact_check_status'] = "unverified"
                result['reasoning'] = "Statement not found in mock knowledge base; requires external verification."
                logger.info(
                    f"Statement '{normalized_statement}' not found in mock DB. "
                    f"Marked as '{result['fact_check_status']}'."
                )

        except Exception as e:
            # Catch any unexpected errors during processing
            logger.exception(
                f"An unhandled error occurred in FactCheckerNode while processing statement: "
                f"'{statement_to_check or data}'. Error: {e}"
            )
            result['fact_check_status'] = "error"
            result['reasoning'] = "An internal error prevented fact-checking from completing."
            result['error'] = str(e)

        return result