import logging
from typing import Any, Dict, Union

# Assuming the project context's import path for BaseNode
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class FactCheckerNode(BaseNode):
    """
    A processing node designed to simulate fact-checking a given statement.
    It attempts to verify the veracity of a statement against an optional
    internal knowledge base provided in the processing context.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this node."""
        return "FactChecker"

    def process(self, data: Union[str, Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to simulate a fact-checking operation.

        This method expects `data` to be either a string representing the statement
        to be checked, or a dictionary containing a 'statement' key.
        The `context` dictionary can optionally contain a 'known_facts' key,
        which should be a dictionary mapping known statements (str) to their
        boolean veracity (True/False).

        Args:
            data: The input data, which can be a string statement directly
                  or a dictionary with a 'statement' key.
            context: A dictionary that may contain a 'known_facts' key
                     for internal verification purposes.

        Returns:
            A dictionary containing the following keys:
            - 'original_statement': The statement that was subjected to the check.
            - 'verdict': A string indicating the verification outcome (e.g., 'TRUE',
                         'FALSE', 'UNVERIFIABLE_INTERNAL', 'ERROR').
            - 'reason': A descriptive string explaining the given verdict.

        Raises:
            TypeError: If the input `data` is neither a string nor a dictionary.
            ValueError: If `data` is a dictionary but lacks the required 'statement' key.
        """
        original_statement: str = ""
        verdict: str = "ERROR"
        reason: str = "An unhandled error prevented verification."

        try:
            if isinstance(data, str):
                original_statement = data
            elif isinstance(data, dict):
                if 'statement' not in data:
                    error_msg = f"Input data dictionary missing 'statement' key for fact-checking: {data}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                original_statement = str(data['statement'])
            else:
                error_msg = f"Invalid input data type for FactCheckerNode: {type(data)}. Expected str or dict."
                logger.error(error_msg)
                raise TypeError(error_msg)

            # Retrieve known facts from context, defaulting to an empty dict if not present
            known_facts: Any = context.get("known_facts", {})
            
            # Ensure known_facts is actually a dictionary before proceeding
            if not isinstance(known_facts, dict):
                logger.warning(
                    f"Context 'known_facts' key contained an invalid type ({type(known_facts)}). "
                    "Expected a dictionary. Skipping internal fact-checking for this reason."
                )
                known_facts = {} # Reset to empty to prevent further errors

            if original_statement in known_facts:
                is_true = known_facts[original_statement]
                verdict = "TRUE" if is_true else "FALSE"
                reason = "Matched against internal knowledge base."
                logger.info(f"Statement '{original_statement}' found in internal knowledge. Verdict: {verdict}")
            else:
                verdict = "UNVERIFIABLE_INTERNAL"
                reason = "Statement not found in internal knowledge base. Further external verification may be required."
                logger.info(f"Statement '{original_statement}' not found internally. Verdict: {verdict}")

        except (TypeError, ValueError) as e:
            verdict = "ERROR"
            reason = f"Input validation or data extraction failed: {e}"
            logger.error(f"FactCheckerNode failed due to an input-related error: {e}", exc_info=True)
        except Exception as e:
            verdict = "ERROR"
            reason = f"An unexpected error occurred during processing: {e}"
            logger.critical(f"FactCheckerNode encountered an unhandled exception: {e}", exc_info=True)
        
        return {
            "original_statement": original_statement,
            "verdict": verdict,
            "reason": reason
        }