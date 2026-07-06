import logging
from typing import Any, Dict, Union
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class FactCheckerNode(BaseNode):
    """
    A processing node that simulates fact-checking for claims.

    This node takes a claim (string or within a dictionary) and attempts to verify
    its veracity against a predefined, simplistic knowledge base. In a real-world
    scenario, this would involve external API calls, database lookups, or
    advanced NLP models.

    Input `data` can be:
    - A string representing the claim directly.
    - A dictionary containing a 'claim' key with the claim as its value.

    Output `data` will be the original input data (if dictionary) or a new dictionary
    (if string input), augmented with a 'fact_check_result' and a 'reason' key.
    """

    _KNOWLEDGE_BASE = {
        "water boils at 100 degrees Celsius": "TRUE",
        "earth is round": "TRUE",
        "sun revolves around the earth": "FALSE", # Common misconception
        "birds are mammals": "FALSE",
        "human body has 206 bones": "TRUE",
        "the capital of France is Paris": "TRUE"
    }
    """A simplistic, in-memory knowledge base for simulation purposes."""

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "FactCheckerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to perform a simulated fact-check.

        Args:
            data: The input data, expected to be a string claim or a dictionary
                  containing a 'claim' key.
            context: A dictionary containing contextual information for the node.
                     (e.g., API keys, configuration, session data).

        Returns:
            The processed data, augmented with fact-checking results.
            If the input data was a string, a new dictionary is returned.
            If the input data was a dictionary, the original dictionary is
            modified and returned.

        Raises:
            ValueError: If the input data is not in an expected format.
            Exception: For unexpected errors during processing.
        """
        logger.info(f"[{self.node_name}] Starting fact-checking process for data type: {type(data)}")
        logger.debug(f"[{self.node_name}] Context received: {list(context.keys()) if context else 'Empty'}")

        claim_to_check: str = ""
        original_data = data # Keep a reference to original data

        if isinstance(data, str):
            claim_to_check = data
            result_container = {"original_claim": data} # Create new dict for string input
        elif isinstance(data, dict):
            if 'claim' in data and isinstance(data['claim'], str):
                claim_to_check = data['claim']
                result_container = data # Modify existing dict for dict input
            else:
                logger.error(f"[{self.node_name}] Input dictionary 'data' must contain a string 'claim' key. Received: {data}")
                raise ValueError("Input dictionary 'data' must contain a 'claim' key with a string value.")
        else:
            logger.error(f"[{self.node_name}] Invalid input data type. Expected str or dict, got: {type(data)}")
            raise ValueError(f"Input data must be a string or a dictionary with a 'claim' key. Got {type(data)}.")

        try:
            # Simulate fact-checking by looking up in a predefined knowledge base
            normalized_claim = claim_to_check.lower().strip()
            fact_check_status = self._KNOWLEDGE_BASE.get(normalized_claim, "UNVERIFIED")

            if fact_check_status == "TRUE":
                reason = "Claim found to be true based on internal knowledge."
            elif fact_check_status == "FALSE":
                reason = "Claim found to be false based on internal knowledge."
            else:
                reason = "Claim could not be definitively verified or falsified by the current knowledge base."
            
            result_container['fact_check_result'] = fact_check_status
            result_container['reason'] = reason
            
            logger.info(f"[{self.node_name}] Fact-checked claim: '{claim_to_check[:50]}...' -> Result: {fact_check_status}")

            return result_container
        except Exception as e:
            logger.error(f"[{self.node_name}] An unexpected error occurred during fact-checking: {e}", exc_info=True)
            # Depending on desired error handling, you might re-raise, return partial, or default.
            # Here, we'll augment the error into the result if it's a dict, otherwise re-raise for string.
            if isinstance(original_data, dict):
                original_data['fact_check_result'] = "ERROR"
                original_data['error_message'] = f"Processing failed: {str(e)}"
                return original_data
            else:
                raise # Re-raise for string input if no dict to hold error
