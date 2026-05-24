import logging
from typing import Any, Dict, Optional
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class FactCheckerNode(BaseNode):
    """
    A Vishustra processing node that simulates fact-checking of input statements.
    
    This node expects 'data' to be a dictionary containing a 'statement' key
    with the text to be fact-checked. It provides a verdict on its factual
    correctness based on a simplified internal logic and can suggest corrections.
    
    It outputs a structured dictionary containing the original statement,
    the fact-checking result, and optionally a suggested correction.
    Robust error handling is included for invalid input formats.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "FactCheckerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data, simulating fact-checking a statement.

        The node expects 'data' to be a dictionary with a 'statement' key,
        containing the string to be fact-checked.

        Args:
            data (Any): The input data, expected to be `{'statement': str}`.
            context (Dict[str, Any]): A dictionary for contextual information,
                                      e.g., global settings or previous node outputs.
                                      Currently, this node does not heavily utilize context.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - 'original_statement': The statement that was checked.
                - 'is_factually_correct': `True` if all checked facts are correct,
                                          `False` if any checked fact is incorrect,
                                          `None` if no specific facts could be verified
                                          or an error occurred.
                - 'checked_items': A list of dictionaries, each describing a specific
                                 fact check (e.g., `{'fact': 'Earth is flat', 'status': False}`).
                - 'suggested_correction': `Optional[str]`, a suggested correction if applicable.
                - 'error': `Optional[str]`, an error message if processing failed due to bad input.
        """
        logger.debug(f"[{self.node_name}] Starting fact-checking process for input: {data}")

        # --- Input Validation ---
        if not isinstance(data, dict):
            error_msg = (
                f"[{self.node_name}] Invalid input data format. Expected a dictionary, "
                f"but received type: {type(data).__name__}. Skipping processing."
            )
            logger.error(error_msg)
            return {
                "original_statement": data,
                "is_factually_correct": None,
                "checked_items": [],
                "suggested_correction": None,
                "error": error_msg,
            }

        statement = data.get("statement")
        if not isinstance(statement, str) or not statement.strip():
            error_msg = (
                f"[{self.node_name}] Missing or invalid 'statement' key in input data. "
                f"Expected a non-empty string for 'statement'."
            )
            logger.error(error_msg)
            return {
                "original_statement": statement,
                "is_factually_correct": None,
                "checked_items": [],
                "suggested_correction": None,
                "error": error_msg,
            }

        results: Dict[str, Any] = {
            "original_statement": statement,
            "is_factually_correct": None,  # Can be True, False, or None (unverified/partial)
            "checked_items": [],
            "suggested_correction": None,
        }

        # --- Simulate Fact-Checking Logic ---
        overall_correctness_flags = []
        lower_statement = statement.lower()

        # Simulating a check for a common misconception
        if "earth is flat" in lower_statement:
            results["checked_items"].append({
                "fact": "Earth is flat",
                "status": False,
                "reason": "Scientific consensus overwhelmingly refutes this notion; Earth is an oblate spheroid."
            })
            overall_correctness_flags.append(False)
            if not results["suggested_correction"]:
                results["suggested_correction"] = "The Earth is an oblate spheroid, not flat."

        # Simulating a check for a universally accepted scientific fact
        if "sun is a star" in lower_statement:
            results["checked_items"].append({
                "fact": "Sun is a star",
                "status": True,
                "reason": "Astronomical classification confirms the Sun as a G-type main-sequence star."
            })
            overall_correctness_flags.append(True)

        # Simulating a check for a basic chemical fact
        if "water is h2o" in lower_statement or "h2o is water" in lower_statement:
            results["checked_items"].append({
                "fact": "Water is H2O",
                "status": True,
                "reason": "Chemical formula for water is H2O, representing two hydrogen atoms and one oxygen atom."
            })
            overall_correctness_flags.append(True)

        # Determine the overall verdict based on individual checks
        if not overall_correctness_flags:
            results["is_factually_correct"] = None  # No specific checks were triggered for this statement
            logger.warning(
                f"[{self.node_name}] No specific fact checks were triggered for statement: "
                f"'{statement[:75]}{'...' if len(statement) > 75 else ''}'. Result is unverified."
            )
        elif all(overall_correctness_flags):
            results["is_factually_correct"] = True
        elif any(not flag for flag in overall_correctness_flags):
            results["is_factually_correct"] = False

        logger.info(
            f"[{self.node_name}] Fact-checking complete for statement: "
            f"'{statement[:75]}{'...' if len(statement) > 75 else ''}'. "
            f"Overall verdict: {results['is_factually_correct']}"
        )
        return results