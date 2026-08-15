import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class FactCheckerNode(BaseNode):
    """
    A processing node designed to simulate fact-checking of textual data.
    It evaluates an input statement against a (simulated) internal knowledge base
    and returns a verification status along with the original data.

    In a production environment, this node would integrate with external fact-checking
    APIs, knowledge graphs, or sophisticated NLP models to perform actual verification.
    For this simulation, a static dictionary of known facts is used.
    """

    # A simulated knowledge base for demonstration purposes.
    # In a real system, this would be dynamically loaded, an external service call,
    # or a more complex semantic matching engine.
    _KNOWN_FACTS: Dict[str, bool] = {
        "Water boils at 100 degrees Celsius": True,
        "The Earth is flat": False,
        "The sun rises in the west": False,
        "Python is a high-level programming language": True,
        "Elephants are capable of human speech": False,
        "The capital of France is Paris": True,
    }

    def __init__(self):
        """
        Initializes the FactCheckerNode.
        """
        logger.debug("FactCheckerNode initialized, ready for processing.")

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node.
        """
        return "FactCheckerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to simulate a fact-checking operation.

        This method extracts a statement from the input `data`, attempts to
        verify it against the node's simulated knowledge base, and returns
        a structured result indicating the verification status.

        Expected `data` formats:
        1. A `str` directly containing the statement to be checked.
        2. A `dict` with a 'text' key whose value is the statement `str`.

        The `context` dictionary can be used to pass configuration parameters
        for more advanced fact-checking (e.g., source preferences, confidence thresholds),
        though not fully implemented in this simulation.

        Args:
            data: The input data, expected to contain the statement for fact-checking.
            context: A dictionary of operational context or configuration parameters.

        Returns:
            A dictionary containing:
            - "original_data": The input `data` as received.
            - "fact_check_result": A sub-dictionary with:
                - "status": One of "verified", "unverified", "needs_review", "error".
                - "reason": A brief explanation for the given status.

        Raises:
            ValueError: If the input `data` is not in an expected format.
        """
        statement_to_check: str = ""
        original_input_data: Any = data
        result_status: str = "needs_review"
        result_reason: str = "Statement not found in simulated knowledge base and requires further review."

        try:
            if isinstance(data, str):
                statement_to_check = data.strip()
            elif isinstance(data, dict) and "text" in data and isinstance(data["text"], str):
                statement_to_check = data["text"].strip()
            else:
                error_msg = (
                    f"FactCheckerNode received unsupported data format. "
                    f"Expected string or dictionary with a 'text' key, but got type {type(data)}."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            if not statement_to_check:
                result_status = "error"
                result_reason = "Input statement is empty or only whitespace."
                logger.warning("FactCheckerNode received an empty statement for processing.")
            else:
                logger.info(f"FactCheckerNode initiated processing for statement: '{statement_to_check[:150]}{'...' if len(statement_to_check) > 150 else ''}'")

                # Simulate fact-checking by direct lookup
                if statement_to_check in self._KNOWN_FACTS:
                    if self._KNOWN_FACTS[statement_to_check]:
                        result_status = "verified"
                        result_reason = "Statement matched a known true fact in our simulated knowledge base."
                    else:
                        result_status = "unverified"
                        result_reason = "Statement matched a known false fact in our simulated knowledge base."
                # If not an exact match, it defaults to 'needs_review' and the initial reason.

        except ValueError as ve:
            # Re-raise explicit ValueErrors for malformed input
            raise ve
        except Exception as e:
            # Catch any other unexpected errors during processing
            logger.exception(
                f"An unhandled error occurred during fact-checking for "
                f"statement: '{statement_to_check[:150]}{'...' if len(statement_to_check) > 150 else ''}': {e}"
            )
            result_status = "error"
            result_reason = f"Internal processing error: {type(e).__name__} - {e}"

        final_result = {
            "original_data": original_input_data,
            "fact_check_result": {
                "status": result_status,
                "reason": result_reason
            }
        }
        logger.debug(f"FactCheckerNode completed processing with status: '{result_status}'")
        return final_result
