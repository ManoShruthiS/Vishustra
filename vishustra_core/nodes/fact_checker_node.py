import logging
from typing import Any, Dict

# Assuming BaseNode is located in this path relative to the project root
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class FactCheckerNode(BaseNode):
    """
    A processing node that simulates fact-checking an input statement
    against a predefined set of known facts.

    The node expects the input `data` to be a string representing the
    statement to be checked. It returns a dictionary detailing the
    verification status.
    """

    _known_facts: Dict[str, bool] = {
        "The Earth revolves around the Sun.": True,
        "Water boils at 100 degrees Celsius at sea level.": True,
        "Birds are mammals.": False,
        "The moon is a satellite of Earth.": True,
        "Humans can breathe underwater unaided.": False,
        "The capital of France is Paris.": True,
        "The sky is green.": False,
    }

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "FactCheckerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data by attempting to verify it as a fact.

        Args:
            data: The statement to be fact-checked (expected as a string).
            context: A dictionary containing contextual information
                     (e.g., session ID, user info). Not directly used for
                     fact-checking logic in this simulated implementation
                     but available for potential extensions like external API calls.

        Returns:
            A dictionary containing:
            - "original_statement": The input statement.
            - "is_fact_checked": True if the statement was processed, False on error.
            - "verification_status": "TRUE", "FALSE", or "UNVERIFIED".
            - "details": A brief explanation of the outcome.

        Raises:
            TypeError: If the input `data` is not a string.
            ValueError: If the input `data` is an empty string.
        """
        result: Dict[str, Any] = {
            "original_statement": data,
            "is_fact_checked": False,
            "verification_status": "ERROR",
            "details": "An unexpected error occurred."
        }

        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input type. Expected 'str', got '{type(data).__name__}'."
            )
            result["details"] = f"Input data must be a string, but received {type(data).__name__}."
            raise TypeError(result["details"])

        statement = data.strip()
        result["original_statement"] = statement # Update if stripped

        if not statement:
            logger.warning(
                f"[{self.node_name}] Received an empty statement for fact-checking."
            )
            result["is_fact_checked"] = True
            result["verification_status"] = "UNVERIFIED"
            result["details"] = "The provided statement was empty."
            return result

        logger.info(
            f"[{self.node_name}] Attempting to fact-check statement: '{statement}'"
        )

        found_fact = False
        for fact, truth_value in self._known_facts.items():
            if statement.lower() == fact.lower(): # Case-insensitive comparison
                result["is_fact_checked"] = True
                result["verification_status"] = "TRUE" if truth_value else "FALSE"
                result["details"] = f"Statement found in known facts: It is {'true' if truth_value else 'false'}."
                found_fact = True
                logger.info(
                    f"[{self.node_name}] Statement '{statement}' verified as '{result['verification_status']}'."
                )
                break

        if not found_fact:
            result["is_fact_checked"] = True
            result["verification_status"] = "UNVERIFIED"
            result["details"] = "Statement not found in the current known facts database."
            logger.warning(
                f"[{self.node_name}] Statement '{statement}' could not be verified against known facts."
            )

        return result

if __name__ == '__main__':
    # Basic usage example for demonstration and testing purposes
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.DEBUG) # Set higher for testing outputs

    fact_checker = FactCheckerNode()
    mock_context: Dict[str, Any] = {"session_id": "test-123", "user_id": "anon"}

    # Test cases
    test_statements = [
        "The Earth revolves around the Sun.",
        "birds are mammals.", # Case-insensitive test
        "The moon is a satellite of Earth.",
        "The sky is green.",
        "Python is the best programming language.", # Unknown fact
        "", # Empty string
        123, # Invalid type
        " The Capital of france is paris. ", # With leading/trailing spaces
    ]

    print(f"\n--- Running {fact_checker.node_name} tests ---")
    for stmt in test_statements:
        try:
            output = fact_checker.process(stmt, mock_context)
            print(f"\nInput: '{stmt}'")
            print(f"Output: {output}")
        except (TypeError, ValueError) as e:
            print(f"\nInput: '{stmt}'")
            print(f"Error: {e}")
        except Exception as e:
            print(f"\nInput: '{stmt}'")
            print(f"Unexpected Error: {e}")
    print("\n--- Tests finished ---")