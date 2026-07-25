import logging
from typing import Any, Dict, Optional

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class FactCheckerNode(BaseNode):
    """
    A processing node that simulates fact-checking of input statements against a
    provided knowledge base in the context.

    This node expects the input `data` to be a dictionary containing a 'statement' key.
    The `context` dictionary should ideally contain a 'fact_checker_kb' key,
    which is a dictionary mapping statements (str) to their boolean truth value (bool).

    The output is a dictionary indicating the original statement, its verified status,
    and a confidence score based on the lookup.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "FactCheckerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to verify a statement.

        Args:
            data (Any): The input data, expected to be a dictionary with a 'statement' key.
            context (Dict[str, Any]): The execution context, expected to contain
                                      'fact_checker_kb' (Dict[str, bool]) for verification.

        Returns:
            Dict[str, Any]: A dictionary containing the original statement, its
                            verification status ('true', 'false', 'unverified'),
                            and a confidence score.

        Raises:
            ValueError: If the input data is not a dictionary or lacks a 'statement' key.
            RuntimeError: If the 'fact_checker_kb' is missing or malformed in the context.
        """
        logger.debug(f"[{self.node_name}] Starting process for data: {data}")

        if not isinstance(data, dict):
            logger.error(f"[{self.node_name}] Invalid input data type. Expected dict, got {type(data)}.")
            raise ValueError("FactCheckerNode requires input 'data' to be a dictionary.")

        statement_to_check: Optional[str] = data.get("statement")
        if not isinstance(statement_to_check, str) or not statement_to_check:
            logger.error(f"[{self.node_name}] Missing or invalid 'statement' key in input data: {data}.")
            raise ValueError("FactCheckerNode requires a non-empty 'statement' string in the input data.")

        fact_checker_kb: Optional[Dict[str, bool]] = context.get("fact_checker_kb")

        if not isinstance(fact_checker_kb, dict):
            logger.warning(
                f"[{self.node_name}] 'fact_checker_kb' not found or is malformed in context. "
                "Fact-checking will default to 'unverified'. Context keys: {list(context.keys())}"
            )
            fact_checker_kb = {} # Default to an empty KB if not provided or malformed

        verification_status: str = "unverified"
        confidence_score: float = 0.0

        if statement_to_check in fact_checker_kb:
            is_true = fact_checker_kb[statement_to_check]
            verification_status = "true" if is_true else "false"
            confidence_score = 1.0
            logger.info(f"[{self.node_name}] Statement '{statement_to_check}' verified as '{verification_status}'.")
        else:
            logger.info(f"[{self.node_name}] Statement '{statement_to_check}' not found in knowledge base. Status: unverified.")

        result = {
            "original_statement": statement_to_check,
            "verification_status": verification_status,
            "confidence_score": confidence_score,
            "node_processed_by": self.node_name,
        }

        logger.debug(f"[{self.node_name}] Finished processing. Result: {result}")
        return result

# Example Usage (for testing purposes, not part of Vishustra execution flow):
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    fact_checker = FactCheckerNode()

    # Define a sample knowledge base
    sample_kb = {
        "The Earth is flat": False,
        "The Sun is a star": True,
        "Water boils at 100 degrees Celsius at sea level": True,
        "Humans can breathe underwater": False,
        "The moon is made of cheese": False,
    }

    # Test cases
    test_context = {"fact_checker_kb": sample_kb}

    # Case 1: Known true fact
    data_true = {"statement": "The Sun is a star"}
    result_true = fact_checker.process(data_true, test_context)
    print(f"Result for '{data_true['statement']}': {result_true}")

    # Case 2: Known false fact
    data_false = {"statement": "The Earth is flat"}
    result_false = fact_checker.process(data_false, test_context)
    print(f"Result for '{data_false['statement']}': {result_false}")

    # Case 3: Unknown fact
    data_unknown = {"statement": "Pineapples grow on trees"}
    result_unknown = fact_checker.process(data_unknown, test_context)
    print(f"Result for '{data_unknown['statement']}': {result_unknown}")

    # Case 4: Missing statement in data
    try:
        data_missing_key = {"info": "some text"}
        fact_checker.process(data_missing_key, test_context)
    except ValueError as e:
        print(f"Caught expected error for missing statement: {e}")

    # Case 5: Invalid data type
    try:
        data_invalid_type = "Just a string"
        fact_checker.process(data_invalid_type, test_context)
    except ValueError as e:
        print(f"Caught expected error for invalid data type: {e}")

    # Case 6: Empty statement
    try:
        data_empty_statement = {"statement": ""}
        fact_checker.process(data_empty_statement, test_context)
    except ValueError as e:
        print(f"Caught expected error for empty statement: {e}")

    # Case 7: Context without KB
    data_no_kb_context = {"statement": "This is a statement"}
    result_no_kb = fact_checker.process(data_no_kb_context, {}) # Empty context
    print(f"Result for '{data_no_kb_context['statement']}' with no KB: {result_no_kb}")