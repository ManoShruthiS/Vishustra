import logging
from typing import Any, Dict, Union
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class FactCheckerNode(BaseNode):
    """
    A processing node designed to simulate fact-checking of input statements.
    It can verify statements against a predefined set of facts provided in the
    context, or use a basic, hardcoded database if no specific facts are supplied.
    The node aims to determine the truth status of a given statement.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "FactChecker"

    def process(self, data: Union[str, Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to verify facts against a known database.

        The `data` input can be:
        - A direct string representing the statement to check.
        - A dictionary containing a 'statement' key whose value is the string to check.

        The `context` dictionary can optionally contain:
        - 'fact_database': A dictionary where keys are known statements (strings)
          and values are their corresponding boolean truth status (True for factual,
          False for counter-factual). If not provided, a hardcoded fallback database
          will be used for simulation purposes.

        Returns a dictionary containing:
        - 'original_statement': The statement that was checked.
        - 'verification_status': A string indicating the outcome ('TRUE', 'FALSE',
          'UNVERIFIED', 'ERROR').
        - 'reason': A brief explanation for the verification status.
        - 'confidence': A float between 0.0 and 1.0 indicating the certainty of the result.
        """
        statement_to_check: str = ""
        result: Dict[str, Any] = {
            "original_statement": None,
            "verification_status": "ERROR",
            "reason": "An unexpected error occurred during processing.",
            "confidence": 0.0
        }

        # 1. Extract the statement to be fact-checked from the input data.
        try:
            if isinstance(data, str):
                statement_to_check = data
            elif isinstance(data, dict) and 'statement' in data and isinstance(data['statement'], str):
                statement_to_check = data['statement']
            else:
                raise ValueError(
                    "Input 'data' must be a string or a dictionary containing a 'statement' key of type string."
                )
            
            result["original_statement"] = statement_to_check
            logger.info(f"FactCheckerNode received statement for processing: '{statement_to_check}'")

        except ValueError as ve:
            logger.error(f"FactCheckerNode: Invalid input data format. Error: {ve}. Data received: {data}")
            result["reason"] = f"Invalid input data format: {ve}"
            return result
        except Exception as e:
            logger.error(f"FactCheckerNode: An unexpected error occurred while extracting statement. Error: {e}. Data received: {data}")
            result["reason"] = f"Unexpected error during statement extraction: {e}"
            return result

        # 2. Retrieve or initialize the fact database.
        # This allows for dynamic facts to be passed via context,
        # or defaults to a hardcoded set for basic functionality.
        fact_database: Dict[str, bool] = context.get('fact_database', {})

        if not fact_database:
            logger.debug("No 'fact_database' found in context. Using a hardcoded fallback for simulation.")
            # A simple, illustrative hardcoded database for demonstration purposes.
            # In a real-world scenario, this would involve a robust data source.
            fact_database = {
                "The sky is blue": True,
                "Birds can fly": True,
                "Water boils at 100 degrees Celsius at sea level": True,
                "Fish can climb trees": False,
                "The Earth is flat": False,
                "Humans have gills": False,
                "The sun is a star": True,
            }
        
        # Normalize the statement for consistent lookup.
        # This is a basic form of normalization; advanced scenarios might use NLP.
        normalized_statement = statement_to_check.strip()

        # 3. Perform the simulated fact-check.
        if normalized_statement in fact_database:
            is_true = fact_database[normalized_statement]
            result["verification_status"] = "TRUE" if is_true else "FALSE"
            result["reason"] = "Statement verified against known facts database."
            result["confidence"] = 1.0 if is_true else 0.95 # Slight confidence drop for false
            logger.info(f"Statement '{statement_to_check}' verified as '{result['verification_status']}'.")
        else:
            result["verification_status"] = "UNVERIFIED"
            result["reason"] = "Statement could not be verified against available facts in the database."
            result["confidence"] = 0.5  # Lower confidence for unverified statements
            logger.warning(f"Statement '{statement_to_check}' could not be verified with available facts.")

        return result
