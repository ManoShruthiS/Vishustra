import logging
from typing import Any, Dict, Union

# Assuming BaseNode is available at this path as per instructions
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class FactCheckerNode(BaseNode):
    """
    A processing node that simulates fact-checking of statements against
    a provided knowledge base within the context.

    The node expects either a string as `data` (the statement) or a dictionary
    containing a 'statement' key.
    The `context` dictionary should ideally contain a 'knowledge_base' key,
    which is a dictionary mapping known statements to their boolean truth values.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "FactCheckerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to fact-check a statement.

        Args:
            data: The input data. Expected to be a string (the statement) or
                  a dictionary containing a 'statement' key.
            context: A dictionary containing additional context, potentially
                     including a 'knowledge_base' for fact-checking.

        Returns:
            A dictionary containing the original statement, its verification status,
            and any relevant evidence or notes.

        Raises:
            ValueError: If the input `data` is not in the expected format.
        """
        statement: str = ""
        result: Dict[str, Any] = {
            "statement": None,
            "is_verified": False,
            "verification_source": "internal_knowledge_base",
            "notes": "Statement could not be processed due to unexpected input format.",
        }

        if isinstance(data, str):
            statement = data
        elif isinstance(data, dict) and 'statement' in data:
            statement = str(data['statement'])
        else:
            logger.error(
                "[%s] Invalid input data format. Expected a string or a dict with 'statement' key. Received: %s",
                self.node_name, type(data)
            )
            result["statement"] = data
            return result

        result["statement"] = statement
        result["notes"] = "Verification in progress."

        knowledge_base: Dict[str, bool] = context.get("knowledge_base", {})
        if not isinstance(knowledge_base, dict):
            logger.warning(
                "[%s] 'knowledge_base' in context is not a dictionary. Fact-checking will be limited.",
                self.node_name
            )
            knowledge_base = {}

        try:
            # Normalize statement for lookup (e.g., lower case, strip whitespace)
            normalized_statement = statement.strip().lower()

            found_match = False
            for known_statement, truth_value in knowledge_base.items():
                if normalized_statement == known_statement.strip().lower():
                    result["is_verified"] = truth_value
                    result["notes"] = f"Verified against knowledge base. Original known statement: '{known_statement}'"
                    found_match = True
                    break

            if not found_match:
                result["is_verified"] = False
                result["notes"] = "Statement not found in the provided knowledge base."
                result["verification_source"] = "knowledge_base_lookup_failure"

        except Exception as e:
            logger.exception(
                "[%s] An error occurred during fact-checking process for statement: '%s'. Error: %s",
                self.node_name, statement, e
            )
            result["is_verified"] = False
            result["notes"] = f"An internal error prevented full verification: {e}"
            result["verification_source"] = "internal_error"

        logger.info(
            "[%s] Processed statement '%s' with verification result: %s",
            self.node_name, statement, result["is_verified"]
        )
        return result

