import logging
import random
from typing import Any, Dict, List

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class FactCheckerNode(BaseNode):
    """
    A processing node that simulates fact-checking for given textual claims.

    This node takes a dictionary containing text and/or a list of claims,
    and returns the input augmented with simulated fact-checking results
    for each claim.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "FactChecker"

    def __init__(self):
        """
        Initializes the FactCheckerNode.
        A rudimentary internal knowledge base is set up for simulation purposes.
        """
        self._known_facts = {
            "sky is blue": True,
            "water is wet": True,
            "sun is hot": True,
            "earth is flat": False,
            "birds can fly": True,
            "fish breathe air": False,
            "cats are dogs": False,
            "humans have three eyes": False,
        }
        logger.debug(f"[{self.node_name}] Initialized with dummy knowledge base for simulation.")

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data by simulating fact-checking on identified claims.

        Expected `data` structure:
        A dictionary that *must* contain one of the following:
        - 'claims': A list of strings, where each string is a claim to be checked.
        - 'text': A string from which claims might be implicitly derived or assumed to be checked.
                  If 'text' is provided but 'claims' is not, the 'text' itself will be treated as a single claim.

        The `context` dictionary can be used for configuration,
        but is not explicitly used for dynamic fact-checking in this simulation.

        Returns:
            A dictionary containing the original data along with a 'fact_check_results' key.
            'fact_check_results' will be a list of dictionaries, each containing:
            - 'claim': The original claim string.
            - 'veracity': 'TRUE', 'FALSE', or 'UNVERIFIED'.
            - 'reason': A brief explanation for the veracity status.

        Raises:
            ValueError: If `data` is not a dictionary or does not contain expected keys.
        """
        logger.info(f"[{self.node_name}] Starting fact-checking process for incoming data.")

        if not isinstance(data, dict):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected 'dict', received '{type(data).__name__}'."
            )
            raise ValueError(f"Input data for {self.node_name} must be a dictionary.")

        claims_to_check: List[str] = []
        if 'claims' in data and isinstance(data['claims'], list):
            valid_claims = [claim for claim in data['claims'] if isinstance(claim, str)]
            if len(valid_claims) < len(data['claims']):
                logger.warning(
                    f"[{self.node_name}] Some items in 'claims' list were not strings and were skipped. "
                    f"Original count: {len(data['claims'])}, Processed count: {len(valid_claims)}."
                )
            claims_to_check = valid_claims
        elif 'text' in data and isinstance(data['text'], str):
            # For simplicity, if only text is provided, treat the entire text as a single claim.
            # A real-world scenario might involve an NLP component to extract multiple claims.
            claims_to_check = [data['text']]
            logger.debug(f"[{self.node_name}] No explicit 'claims' list found, treating 'text' as a single claim.")
        else:
            logger.error(
                f"[{self.node_name}] Input data must contain either a 'claims' key (list of strings) "
                f"or a 'text' key (string). Keys found: {list(data.keys())}."
            )
            raise ValueError(
                f"Input data for {self.node_name} must contain either a 'claims' list or a 'text' string."
            )

        if not claims_to_check:
            logger.warning(f"[{self.node_name}] No valid claims found to process in the input data. Returning original data with empty results.")
            return {**data, 'fact_check_results': []}

        fact_check_results: List[Dict[str, Any]] = []
        for claim in claims_to_check:
            result = self._simulate_fact_check(claim)
            fact_check_results.append({
                'claim': claim,
                'veracity': result['veracity'],
                'reason': result['reason']
            })
            logger.debug(f"[{self.node_name}] Processed claim: '{claim}' -> Veracity: {result['veracity']}")

        output_data = {**data, 'fact_check_results': fact_check_results}
        logger.info(f"[{self.node_name}] Fact-checking process completed for {len(claims_to_check)} claims.")
        return output_data

    def _simulate_fact_check(self, claim: str) -> Dict[str, str]:
        """
        Simulates fact-checking for a given claim against a small internal knowledge base.
        This is a dummy implementation to demonstrate the node's behavior.
        """
        lower_claim = claim.lower()
        
        # Check against predefined known facts
        for fact_key, is_true in self._known_facts.items():
            if fact_key in lower_claim:
                veracity = 'TRUE' if is_true else 'FALSE'
                reason = f"Matched against known fact: '{fact_key}'."
                logger.debug(f"[{self.node_name}] Simulated match for '{claim}': {veracity} based on '{fact_key}'.")
                return {'veracity': veracity, 'reason': reason}

        # If no direct match, assign a random status for simulation
        rand_val = random.random()
        if rand_val < 0.2:  # 20% chance of being FALSE
            veracity = 'FALSE'
            reason = "Simulated falsehood: No direct match found in knowledge base, random assignment."
        elif rand_val < 0.6:  # 40% chance of being TRUE (remaining 80%, so 50% of remaining is 40%)
            veracity = 'TRUE'
            reason = "Simulated truth: No direct match found in knowledge base, random assignment."
        else:  # 40% chance of being UNVERIFIED
            veracity = 'UNVERIFIED'
            reason = "Could not verify: No direct match found in knowledge base."
        
        logger.debug(f"[{self.node_name}] Simulated random check for '{claim}': {veracity}.")
        return {'veracity': veracity, 'reason': reason}