import logging
from typing import Any, Dict, Union

# Assuming BaseNode is located here as per project context
from vishustra_core.nodes.base_node import BaseNode

# Initialize logger for this module
logger = logging.getLogger(__name__)

class FactCheckerNode(BaseNode):
    """
    A Vishustra processing node that simulates fact-checking a given statement.

    This node takes a string statement and attempts to determine its factual accuracy
    based on a simplified, internal rule set for demonstration purposes. In a production
    environment, this would typically integrate with external fact-checking APIs,
    knowledge graphs, or sophisticated NLP models to perform real verification.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "FactCheckerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Union[str, bool, None, float]]:
        """
        Processes the input data, attempting to fact-check a statement.

        Expects `data` to be a string representing the statement to be checked.
        Returns a dictionary containing the original statement, a factual verdict,
        and a simulated justification and confidence score.

        Args:
            data: The input data, expected to be a string statement for fact-checking.
            context: A dictionary containing contextual information for processing.
                     This could include API keys, service endpoints, or additional
                     configuration for a real-world fact-checker. Not strictly used
                     for core logic in this simulation but available for extensibility.

        Returns:
            A dictionary with the following keys:
            - "original_statement": The exact string that was input for checking.
            - "is_factual": True if the statement is considered factual, False if non-factual,
                            or None if the factuality could not be determined.
            - "justification": A string explaining the simulated verdict.
            - "confidence": A float from 0.0 to 1.0 indicating the simulated confidence
                            in the verdict.

        Raises:
            ValueError: If the input `data` is not a string, as this node is designed
                        to process textual statements.
        """
        if not isinstance(data, str):
            error_msg = (
                f"FactCheckerNode received invalid data type. Expected 'str', "
                f"but got '{type(data).__name__}'. Unable to process."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        statement = data.lower().strip()
        is_factual: Union[bool, None] = None
        justification: str = "Factuality could not be conclusively determined by internal rules."
        confidence: float = 0.5 # Default neutral confidence

        logger.info(
            f"FactCheckerNode initiated processing for statement: "
            f"'{statement[:100]}{'...' if len(statement) > 100 else ''}'"
        )

        # --- Simulate fact-checking logic based on keywords ---
        # This is a highly simplified mock for demonstration purposes.
        # A real system would use external services or a robust knowledge base.

        if "gravity is a force" in statement or "water boils at 100 degrees celsius" in statement:
            is_factual = True
            justification = "Statement aligns with widely accepted scientific principles."
            confidence = 0.98
        elif "earth is flat" in statement or "birds aren't real" in statement or "vaccines cause autism" in statement:
            is_factual = False
            justification = "Statement contradicts widely accepted scientific consensus or verifiable facts."
            confidence = 0.99
        elif "sun rises in the east" in statement or "human heart has four chambers" in statement:
            is_factual = True
            justification = "Statement is a verifiable observational or anatomical fact."
            confidence = 0.95
        elif "unicorns exist" in statement or "flying spaghetti monster" in statement:
            is_factual = False
            justification = "Statement describes a mythological or fictional entity/concept."
            confidence = 0.90
        else:
            logger.info("Statement did not match any specific fact-checking rules. Returning inconclusive.")

        # Log the outcome of the processing
        if is_factual is True:
            logger.info(f"Statement found to be factual with confidence {confidence:.2f}.")
        elif is_factual is False:
            logger.warning(f"Statement found to be non-factual with confidence {confidence:.2f}.")
        else:
            logger.info("Statement factuality inconclusive with default confidence.")

        return {
            "original_statement": data,
            "is_factual": is_factual,
            "justification": justification,
            "confidence": confidence
        }