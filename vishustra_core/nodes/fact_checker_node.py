import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class FactCheckerNode(BaseNode):
    """
    A processing node designed to simulate fact-checking of a given textual statement.

    This node expects the input `data` to be a dictionary containing a 'statement' key
    whose value is the string to be fact-checked. It provides a simulated truth value
    and an explanation based on a simplified internal logic.

    The node returns a dictionary containing the original statement, the simulated
    fact-check result, and an accompanying explanation.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "FactCheckerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to perform a simulated fact-check.

        Args:
            data (Any): The input data, expected to be a dictionary with a 'statement' key
                        (str). Example: `{'statement': 'The sky is blue.'}`
            context (Dict[str, Any]): A dictionary for shared contextual information or
                                       configuration across the orchestration pipeline.
                                       Currently not used by this node but available for future
                                       enhancements (e.g., external API keys, knowledge base URLs).

        Returns:
            Any: A dictionary containing the `original_statement`, `fact_check_result` (e.g.,
                 'TRUE', 'FALSE', 'NEEDS_REVIEW'), and an `explanation`.
                 Example: `{'original_statement': 'The sky is blue.',
                           'fact_check_result': 'TRUE',
                           'explanation': 'Commonly accepted fact based on visual observation.'}`

        Raises:
            ValueError: If the input data is not a dictionary, or if it lacks the 'statement' key.
            TypeError: If the value associated with the 'statement' key is not a string.
        """
        logger.debug("FactCheckerNode received data: %s", data)

        # Validate input data type
        if not isinstance(data, dict):
            logger.error("Invalid input data type for FactCheckerNode. Expected dict, got %s.", type(data))
            raise ValueError(f"FactCheckerNode requires input data to be a dictionary, but received {type(data)}.")

        # Extract and validate the 'statement'
        statement = data.get('statement')
        if statement is None:
            logger.error("Missing required 'statement' key in input data for FactCheckerNode.")
            raise ValueError("FactCheckerNode requires a 'statement' key in the input data.")

        if not isinstance(statement, str):
            logger.error("Invalid type for 'statement' in FactCheckerNode. Expected str, got %s.", type(statement))
            raise TypeError(f"The 'statement' value must be a string, but received {type(statement)}.")

        # --- Simulated Fact-Checking Logic ---
        # In a real-world scenario, this would involve calling external APIs,
        # querying a knowledge graph, or performing complex natural language processing.
        # For this simulation, we use a simple rule-based approach.
        
        statement_lower = statement.lower()
        result = "NEEDS_REVIEW"
        explanation = "Could not definitively verify the statement with currently available internal knowledge."

        # Simple pattern matching for common facts/misconceptions
        if "sky is blue" in statement_lower:
            result = "TRUE"
            explanation = "Commonly accepted natural phenomenon based on atmospheric light scattering."
        elif "earth is flat" in statement_lower:
            result = "FALSE"
            explanation = "Scientifically disproven fact; the Earth is an oblate spheroid."
        elif "water boils at 100 degrees celsius" in statement_lower:
            result = "TRUE"
            explanation = "Standard scientific fact at sea level pressure (1 atmosphere)."
        elif "cats are dogs" in statement_lower:
            result = "FALSE"
            explanation = "Biologically incorrect; cats (Felidae) and dogs (Canidae) are distinct families."
        elif "sun revolves around earth" in statement_lower:
            result = "FALSE"
            explanation = "Heliocentric model is scientifically accepted; Earth revolves around the Sun."

        logger.info("Fact-checked statement '%s': Result - %s", statement, result)

        return {
            'original_statement': statement,
            'fact_check_result': result,
            'explanation': explanation
        }
