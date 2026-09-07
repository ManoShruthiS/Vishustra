import logging
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class FactCheckerNode(BaseNode):
    """
    A Vishustra processing node designed to simulate fact-checking of input claims.

    This node evaluates claims against a simplified, internal knowledge base to
    determine if they are supported, refuted, or unverified. It's intended to
    demonstrate a basic claim verification mechanism within the orchestration
    framework.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this processing node."""
        return "FactChecker"

    def __init__(self):
        """
        Initializes the FactCheckerNode.

        This constructor sets up a mock knowledge base for demonstration purposes.
        In a production environment, this would involve connections to external
        fact-checking APIs, semantic databases, or a more sophisticated internal
        knowledge graph.
        """
        self._knowledge_base = {
            "the sky is blue": {"status": "supported", "reason": "Common scientific observation regarding Rayleigh scattering."},
            "water boils at 100 degrees celsius at sea level": {"status": "supported", "reason": "Standard physical property of water."},
            "mars is the fourth planet from the sun": {"status": "supported", "reason": "Astronomical fact about planetary order."},
            "earth is the third planet from the sun": {"status": "supported", "reason": "Astronomical fact about planetary order."},
            "the sun revolves around the earth": {"status": "refuted", "reason": "The heliocentric model, where Earth revolves around the Sun, is scientifically accepted."},
            "birds cannot fly": {"status": "refuted", "reason": "Many bird species are capable of flight, it's a defining characteristic for many."},
            "all humans have wings": {"status": "refuted", "reason": "Humans are mammals and do not possess wings."},
            "vishustra is an llm orchestration framework": {"status": "unverified", "reason": "This is a project-specific detail, not general public knowledge, and requires internal project context verification."}
        }
        logger.info(f"{self.node_name} node initialized with a mock knowledge base.")

    def _check_single_claim(self, claim_text: str) -> Dict[str, Any]:
        """
        Internal helper method to check a single claim against the mock knowledge base.

        Args:
            claim_text: The textual claim to be checked.

        Returns:
            A dictionary containing the claim, its verification status, and simulated evidence.
        """
        lower_claim = claim_text.lower().strip()
        
        # Simple exact match check
        for fact, details in self._knowledge_base.items():
            if lower_claim == fact:
                logger.debug(f"Claim '{claim_text}' directly matched a known fact with status: {details['status']}.")
                return {
                    "claim": claim_text,
                    "status": details["status"],
                    "evidence": details["reason"],
                    "node": self.node_name
                }
            
            # Very basic negation check: "not X" where X is a known fact
            if "not " in lower_claim:
                negated_part = lower_claim.replace("not ", "", 1).strip()
                if negated_part == fact:
                    if details["status"] == "supported":
                        logger.debug(f"Claim '{claim_text}' appears to refute a supported fact: '{fact}'.")
                        return {
                            "claim": claim_text,
                            "status": "refuted",
                            "evidence": f"Opposite of known supported fact: '{fact}'.",
                            "node": self.node_name
                        }
                    elif details["status"] == "refuted":
                        logger.debug(f"Claim '{claim_text}' appears to support by refuting a refuted fact: '{fact}'.")
                        return {
                            "claim": claim_text,
                            "status": "supported",
                            "evidence": f"Opposite of known refuted fact: '{fact}'.",
                            "node": self.node_name
                        }

        logger.info(f"Claim '{claim_text}' could not be definitively verified or refuted by the internal mock knowledge base.")
        return {
            "claim": claim_text,
            "status": "unverified",
            "evidence": "No direct match or clear contradiction found in the internal knowledge base. Requires external verification.",
            "node": self.node_name
        }

    def process(self, data: Union[str, Dict[str, Any], List[Union[str, Dict[str, Any]]]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Processes the input data to perform fact-checking on one or more claims.

        The `data` input is flexible and can be provided as:
        - A `str`: A single claim string.
        - A `Dict[str, Any]`: A dictionary expected to contain a 'claim' key whose value is a string.
        - A `List[Union[str, Dict[str, Any]]]`: A list where each item is either a string claim
          or a dictionary containing a 'claim' key.

        The `context` parameter is a dictionary intended for passing configuration or
        runtime information. For this node's current implementation, it is acknowledged
        but not actively used in the fact-checking logic.

        Args:
            data: The input claims to be processed.
            context: A dictionary for contextual information (currently not used by this node).

        Returns:
            A list of dictionaries. Each dictionary represents a processed claim and
            includes the original claim text, its determined 'status' ('supported',
            'refuted', 'unverified', or 'error'), and simulated 'evidence'.

        Raises:
            ValueError: If the input `data` is not a recognized type (string, dictionary
                        with 'claim', or a list containing these types).
        """
        processed_results: List[Dict[str, Any]] = []
        claims_to_process: List[str] = []

        if isinstance(data, str):
            claims_to_process.append(data)
        elif isinstance(data, dict) and 'claim' in data and isinstance(data['claim'], str):
            claims_to_process.append(data['claim'])
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    claims_to_process.append(item)
                elif isinstance(item, dict) and 'claim' in item and isinstance(item['claim'], str):
                    claims_to_process.append(item['claim'])
                else:
                    logger.warning(f"Skipping malformed item in input list; expected string or dict with 'claim' key. Item: {item}")
        else:
            logger.error(f"Invalid input data type encountered: {type(data)}. Expected str, dict with 'claim', or list of these.")
            raise ValueError(f"FactCheckerNode received unsupported data type: {type(data)}. Data must be a string, dict with 'claim', or a list thereof.")

        if not claims_to_process:
            logger.warning("No valid claims were extracted from the input data for processing.")
            return []

        logger.info(f"Initiating fact-checking for {len(claims_to_process)} claims using the {self.node_name} node.")

        for claim_text in claims_to_process:
            try:
                result = self._check_single_claim(claim_text)
                processed_results.append(result)
            except Exception as e:
                logger.error(f"An unexpected error occurred while checking claim '{claim_text}': {e}", exc_info=True)
                processed_results.append({
                    "claim": claim_text,
                    "status": "error",
                    "evidence": f"An internal error prevented fact-checking: {e}",
                    "node": self.node_name
                })
        
        logger.debug(f"Fact-checking completed for {len(processed_results)} claims.")
        return processed_results
