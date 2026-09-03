import logging
from typing import Any, Dict, Union
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class FactCheckerNode(BaseNode):
    """
    A processing node designed to simulate fact-checking a given textual statement.

    This node expects the input `data` to be a string representing a statement.
    It attempts to verify the factual accuracy of this statement against a set
    of known facts. These facts can be provided via the `context` dictionary
    or defaults to a basic internal knowledge base for demonstration purposes.

    The fact-checking mechanism is a simple string matching lookup. For a real-world
    application, this would involve sophisticated NLP, knowledge graph lookups,
    or external API calls.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "FactChecker"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Union[str, bool, None]]:
        """
        Processes the input data, treating it as a statement to be fact-checked.

        The node prioritizes 'known_facts' provided in the `context` dictionary.
        If not present or invalid, it falls back to an internal mock knowledge base.

        Args:
            data: The statement to be fact-checked, expected to be a string.
            context: A dictionary that may optionally contain a 'known_facts' key.
                     'known_facts' should be a dictionary where keys are statements (str)
                     and values are their corresponding boolean truth values.

        Returns:
            A dictionary containing:
            - 'original_statement': The statement that was provided for checking.
            - 'is_factual': True if the statement is determined to be factual,
                            False if it's determined to be non-factual, or
                            None if its factuality could not be determined.
            - 'explanation': A string providing details about the fact-checking outcome.

        Raises:
            ValueError: If the input 'data' is not a string, indicating an invalid input type.
            Exception: Catches and logs any unexpected errors during the fact-checking process.
        """
        if not isinstance(data, str):
            logger.error(f"FactCheckerNode received invalid data type. Expected 'str', got '{type(data).__name__}'.")
            raise ValueError(f"Input data for FactCheckerNode must be a string. Got '{type(data).__name__}'.")

        statement_to_check = data.strip()
        logger.debug(f"FactCheckerNode initiated processing for statement: '{statement_to_check}'")

        # --- Knowledge Base Simulation ---
        # Prioritize 'known_facts' from the context, ensuring it's a dictionary.
        context_known_facts = context.get("known_facts", {})
        if not isinstance(context_known_facts, dict):
            logger.warning(
                "Context key 'known_facts' was found but is not a dictionary. "
                "Ignoring context-provided facts and falling back to internal mock facts."
            )
            context_known_facts = {}

        # Basic internal mock knowledge base for demonstration.
        # In a production system, this would be replaced by actual data sources or APIs.
        internal_mock_facts = {
            "the sun is a star": True,
            "water boils at 100 degrees celsius": True,
            "birds can fly": True,
            "fish can climb trees": False,
            "mount everest is the highest mountain": True,
            "humans have three hearts": False,
            "the earth is flat": False,
            "cats are mammals": True,
            "a triangle has three sides": True,
            "the capital of france is berlin": False, # Example of a false fact
            "dogs lay eggs": False,
        }
        
        # Merge internal mock facts with context-provided facts.
        # Context facts take precedence in case of conflicts.
        # All keys are converted to lower case for case-insensitive matching.
        merged_facts = {k.lower(): v for k, v in internal_mock_facts.items()}
        merged_facts.update({k.lower(): v for k, v in context_known_facts.items()})

        # Initialize the result structure
        result: Dict[str, Union[str, bool, None]] = {
            "original_statement": statement_to_check,
            "is_factual": None,
            "explanation": "Factuality could not be determined."
        }

        try:
            # Simple case-insensitive exact match for demonstration purposes.
            # Real-world fact-checking involves complex NLP, semantic understanding,
            # and potentially multiple corroborating sources.
            lower_statement = statement_to_check.lower()

            if lower_statement in merged_facts:
                is_true = merged_facts[lower_statement]
                result["is_factual"] = is_true
                result["explanation"] = (
                    f"Based on available knowledge, this statement is "
                    f"{'factual' if is_true else 'not factual'}."
                )
                logger.info(f"Statement '{statement_to_check}' fact-checked as {is_true}.")
            else:
                logger.info(
                    f"Statement '{statement_to_check}' was not found in the available knowledge base. "
                    f"Factuality undetermined."
                )
                result["explanation"] = (
                    "The statement could not be directly verified against the available knowledge base. "
                    "Consider expanding the 'known_facts' in the context."
                )

        except Exception as e:
            logger.exception(
                f"An unexpected error occurred during fact-checking for statement: '{statement_to_check}'."
            )
            result["explanation"] = f"An internal error prevented fact-checking: {type(e).__name__}: {str(e)}"
            # is_factual remains None, indicating an error state

        return result