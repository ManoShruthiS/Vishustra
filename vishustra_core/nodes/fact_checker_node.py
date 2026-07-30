import logging
from typing import Any, Dict, List

# Assuming vishustra_core is a package, and base_node is a module within its nodes subpackage
from vishustra_core.nodes.base_node import BaseNode 

logger = logging.getLogger(__name__)

class FactCheckerNode(BaseNode):
    """
    A Vishustra processing node that simulates fact-checking of input statements.

    This node takes a statement (either directly as a string or within a dictionary
    under the 'statement' key) and attempts to determine its veracity. It returns
    a simulated verdict, confidence score, and a list of supporting or refuting evidence.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "FactChecker"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to simulate fact-checking a given statement.

        The node expects the input `data` to be either a raw string containing the
        statement to be checked, or a dictionary that includes a 'statement' key
        whose value is the string to check.

        Args:
            data: The input data, expected to be a string statement or a dictionary
                  containing a 'statement' key.
            context: A dictionary of contextual information. While not heavily
                     utilized in this simulation, it can provide configuration
                     or shared state for more complex fact-checking implementations.

        Returns:
            A dictionary containing the original statement, a simulated verdict,
            a confidence score (0.0 to 1.0), a list of simulated evidence,
            and the name of the processing node.

        Raises:
            ValueError: If the input data is not in the expected format (str or
                        dict with a 'statement' string key).
            Exception: For unexpected internal errors encountered during the
                       simulated fact-checking process.
        """
        statement_to_check: str = ""
        
        # Extract the statement from the input data
        if isinstance(data, str):
            statement_to_check = data
        elif isinstance(data, dict) and 'statement' in data and isinstance(data['statement'], str):
            statement_to_check = data['statement']
        else:
            error_msg = (
                f"Invalid input data format for FactCheckerNode. "
                f"Expected str or dict with a 'statement' string key, received {type(data)}."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"FactCheckerNode received statement for checking: '{statement_to_check}'")
        
        # Initialize simulated fact-checking results
        verdict: str = "UNVERIFIABLE"
        confidence: float = 0.5
        evidence: List[str] = []

        try:
            lower_statement = statement_to_check.lower()

            # Simulate fact-checking based on keywords
            if "earth is flat" in lower_statement or "flat earth" in lower_statement:
                verdict = "FALSE"
                confidence = 0.99
                evidence = ["Overwhelming scientific consensus.", "Satellite imagery and geodesic measurements."]
            elif "sun orbits earth" in lower_statement:
                verdict = "FALSE"
                confidence = 0.95
                evidence = ["Heliocentric model of the solar system.", "Astronomical observations and physics."]
            elif "humans landed on the moon" in lower_statement:
                verdict = "TRUE"
                confidence = 0.98
                evidence = ["NASA Apollo mission records.", "Lunar samples.", "Independent verifications."]
            elif "water is wet" in lower_statement:
                verdict = "PARTIALLY TRUE" 
                confidence = 0.7
                evidence = ["Water makes things wet, but its 'wetness' is subjective and depends on context."]
            elif "vishustra" in lower_statement and "framework" in lower_statement:
                verdict = "TRUE"
                confidence = 0.90
                evidence = ["Project documentation and public announcements."]
            else:
                verdict = "NEEDS MANUAL REVIEW"
                confidence = 0.6
                evidence = ["Automated checks inconclusive. Requires human expert analysis or deeper search."]
            
            logger.info(
                f"FactCheckerNode completed check for '{statement_to_check}': "
                f"Verdict='{verdict}', Confidence={confidence:.2f}"
            )

        except Exception as e:
            logger.error(
                f"An unexpected error occurred during fact-checking simulation for '{statement_to_check}': {e}", 
                exc_info=True
            )
            # Re-raise the exception, possibly wrapping it in a custom framework exception
            raise Exception(f"Failed to process statement due to internal simulation error: {e}") from e

        # Prepare the structured result
        result = {
            "statement": statement_to_check,
            "verdict": verdict,
            "confidence": confidence,
            "evidence": evidence,
            "processed_by": self.node_name,
            # Example of how context might be reflected in the output
            "context_source": context.get("source_id", "N/A") 
        }
        
        return result
