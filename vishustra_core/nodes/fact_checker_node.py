import logging
from typing import Any, Dict, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class FactCheckerNode(BaseNode):
    """
    A processing node designed to simulate fact-checking an input statement
    against a predefined or external knowledge base.

    This node takes a statement (string) or a dictionary containing a
    statement and attempts to determine its veracity, returning a structured
    result with verification status, confidence, and evidence.
    """

    # In a real-world scenario, this would interact with an external fact-checking
    # API, a knowledge graph service, or a database. For this simulation,
    # we use a simple internal dictionary as a mock knowledge base.
    _KNOWN_FACTS: Dict[str, Dict[str, Any]] = {
        "The Earth is round.": {"is_fact": True, "evidence": "Satellite imagery, historical observations."},
        "Water boils at 100 degrees Celsius at sea level.": {"is_fact": True, "evidence": "Scientific consensus, experiments."},
        "The moon is made of cheese.": {"is_fact": False, "evidence": "Lunar samples, geological analysis."},
        "Birds can fly.": {"is_fact": True, "evidence": "Observation of most bird species."},
        "Fish can breathe air.": {"is_fact": False, "evidence": "Most fish use gills for underwater respiration, not air."},
        "The capital of France is Paris.": {"is_fact": True, "evidence": "Geographic and political records."},
        "Humans can survive indefinitely without water.": {"is_fact": False, "evidence": "Biological and medical science."},
    }

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "FactCheckerNode"

    def process(self, data: Union[str, Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data, attempting to fact-check a statement against
        the node's internal (simulated) knowledge base.

        Expected `data` formats:
        - A string: The statement to be fact-checked directly.
        - A dictionary: Must contain a 'statement' key with the string to check.
                        e.g., {'statement': 'The Earth is flat.'}

        Args:
            data: The input data containing the statement to check.
            context: A dictionary of contextual information. While not directly
                     used to modify the internal knowledge base in this simulation,
                     it's available for passing external services or configuration
                     in a more dynamic, real-world implementation.

        Returns:
            A dictionary containing the original statement, its verified status,
            a confidence score, and supporting evidence.
            Example for a verified fact:
            {
                "original_statement": "The Earth is round.",
                "is_fact": True,
                "confidence": 0.95,
                "evidence": "Satellite imagery, historical observations.",
                "explanation": "Statement found in known facts and confirmed as true."
            }
            Example for an unverified statement:
            {
                "original_statement": "The Earth is flat.",
                "is_fact": None,
                "confidence": 0.0,
                "evidence": "No definitive information found in knowledge base.",
                "explanation": "Could not verify statement against available knowledge."
            }

        Raises:
            ValueError: If the input `data` is not a string or a dictionary
                        with a 'statement' key containing a string value.
        """
        statement_to_check: str = ""

        if isinstance(data, str):
            statement_to_check = data
        elif isinstance(data, dict):
            if "statement" in data and isinstance(data["statement"], str):
                statement_to_check = data["statement"]
            else:
                logger.error(
                    "FactCheckerNode received a dictionary without a 'statement' key "
                    "or with a non-string value for 'statement'. Received data: %s", data
                )
                raise ValueError(
                    "Input dictionary 'data' must contain a string value for the 'statement' key."
                )
        else:
            logger.error(
                "FactCheckerNode received unexpected data type: %s. Expected str or dict.", type(data)
            )
            raise ValueError(
                f"FactCheckerNode expects 'data' to be a string or a dictionary "
                f"with a 'statement' key, but received type: {type(data)}."
            )

        logger.info("Initiating fact-check for statement: '%s'", statement_to_check)

        # Simulate checking against our internal knowledge base
        verification_result = self._KNOWN_FACTS.get(statement_to_check)

        if verification_result:
            logger.info("Statement '%s' found in known facts.", statement_to_check)
            # Assign confidence based on known veracity
            confidence_score = 0.95 if verification_result["is_fact"] else 0.85
            explanation_text = "Statement found in known facts and its veracity is established."
            
            return {
                "original_statement": statement_to_check,
                "is_fact": verification_result["is_fact"],
                "confidence": confidence_score,
                "evidence": verification_result["evidence"],
                "explanation": explanation_text
            }
        else:
            logger.warning("Statement '%s' not found in known facts. Cannot verify with current knowledge.", statement_to_check)
            return {
                "original_statement": statement_to_check,
                "is_fact": None,  # Indicates unknown or unverified status
                "confidence": 0.0,
                "evidence": "No definitive information found in knowledge base.",
                "explanation": "Could not verify statement against available knowledge."
            }