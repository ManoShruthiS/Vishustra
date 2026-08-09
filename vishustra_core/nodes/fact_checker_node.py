import logging
from typing import Any, Dict, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class FactCheckerNode(BaseNode):
    """
    A processing node designed to simulate fact-checking of input statements.
    It verifies statements against an internal, predefined knowledge base,
    which can be extended or overridden via the processing context.

    This node provides a foundational mechanism for validating textual claims
    within the Vishustra framework, returning a structured result indicating
    the statement's factual status and confidence.
    """

    # A simple, internal mock knowledge base for demonstration purposes.
    # In a real-world scenario, this would interface with a sophisticated
    # external fact-checking service or a robust knowledge graph.
    _internal_fact_db: Dict[str, bool] = {
        "earth is round": True,
        "sun orbits earth": False,
        "water boils at 100 degrees celsius": True,
        "birds are mammals": False,
        "pi is exactly 3": False,
        "elephants can fly": False,
        "humans breathe oxygen": True,
        "internet was invented in the 1990s": False, # Actual origins trace back to ARPANET in 1960s
    }

    def __init__(self):
        """
        Initializes the FactCheckerNode.
        Logs the node's initialization for operational monitoring.
        """
        logger.info(f"'{self.node_name}' node initialized.")

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this node.
        """
        return "FactCheckerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to perform a simulated fact-check.

        The method expects `data` to be either a string representing the
        statement to check or a dictionary containing a 'statement' key
        with a string value.

        The `context` dictionary can optionally contain a 'fact_db' key.
        If present, this 'fact_db' (expected to be a dictionary mapping
        statement fragments to boolean truth values) will augment or
        override the node's internal knowledge base for the current processing
        cycle.

        Returns a dictionary with the following keys:
        - 'original_statement': The exact statement that was provided for checking.
        - 'is_factual': A boolean (True/False) indicating the factual status.
                        None if the statement could not be verified.
        - 'confidence': A float between 0.0 and 1.0, representing the confidence
                        in the fact-check result.
        - 'explanation': A string detailing the outcome of the fact-check.
        """
        statement_to_check: str = ""
        result: Dict[str, Any] = {
            "original_statement": None,
            "is_factual": None,
            "confidence": 0.0,
            "explanation": "Input data format invalid or missing statement."
        }

        # Validate input data format
        if isinstance(data, str):
            statement_to_check = data
        elif isinstance(data, dict) and 'statement' in data and isinstance(data['statement'], str):
            statement_to_check = data['statement']
        else:
            logger.warning(
                f"[{self.node_name}] Received unexpected data format: {type(data)}. "
                "Expected 'str' or 'dict' with a 'statement' key. Skipping fact-check."
            )
            return result

        result["original_statement"] = statement_to_check
        statement_lower = statement_to_check.lower()

        # Compile the current knowledge base, prioritizing context-provided facts
        current_fact_db = self._internal_fact_db.copy()
        if 'fact_db' in context and isinstance(context['fact_db'], dict):
            # Ensure context facts are also lowercased for consistent matching
            context_facts_lower = {k.lower(): v for k, v in context['fact_db'].items()}
            current_fact_db.update(context_facts_lower)
            logger.debug(
                f"[{self.node_name}] Context-provided 'fact_db' augmented internal knowledge base. "
                f"Total active facts: {len(current_fact_db)}"
            )
        else:
            logger.debug(f"[{self.node_name}] Using only internal knowledge base.")

        found_match = False
        # Iterate through the fact database to find a matching phrase
        for fact_phrase, truth_value in current_fact_db.items():
            if fact_phrase in statement_lower:
                result['is_factual'] = truth_value
                result['confidence'] = 1.0
                result['explanation'] = (
                    f"Statement contains known fact: '{fact_phrase}', which is "
                    f"{'True' if truth_value else 'False'}."
                )
                found_match = True
                logger.info(
                    f"[{self.node_name}] Fact-checked '{statement_to_check[:50]}...': "
                    f"Result: {result['is_factual']}"
                )
                break  # For this simulation, the first direct match is considered decisive

        if not found_match:
            # If no direct match is found, the statement is considered unverifiable
            result['is_factual'] = None
            result['confidence'] = 0.5  # Neutral confidence for unverifiable statements
            result['explanation'] = (
                "No direct match found in the available knowledge base. "
                "The statement's factual status is currently unverifiable."
            )
            logger.warning(
                f"[{self.node_name}] Fact-check for '{statement_to_check[:50]}...': Unverifiable."
            )

        return result
