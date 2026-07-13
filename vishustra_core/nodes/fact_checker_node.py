import logging
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class FactCheckerNode(BaseNode):
    """
    A processing node that simulates fact-checking for given claims or statements.
    It verifies input text against a set of predefined 'known facts' and
    'known falsehoods'.
    """

    def __init__(self):
        """
        Initializes the FactCheckerNode with a set of simulated facts and falsehoods.
        In a real-world scenario, this would involve integrating with external
        fact-checking APIs, knowledge bases, or internal models.
        """
        self._known_facts = {
            "the earth is round": True,
            "water boils at 100 degrees celsius at sea level": True,
            "the sun is a star": True,
            "grass is green": True,
            "human beings need oxygen to survive": True,
        }
        self._known_falsehoods = {
            "the earth is flat": False,
            "humans can fly without aid": False,
            "pigs can fly": False,
            "water boils at 50 degrees celsius at sea level": False,
            "the moon is made of cheese": False,
        }
        logger.info("FactCheckerNode initialized with simulated knowledge base.")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "FactCheckerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data, attempting to verify a claim.

        The `data` input is expected to be either:
        - A string containing the claim to be checked.
        - A dictionary with a 'claim' key holding the string claim.

        The `context` dictionary can be used for configuration (e.g., 'fact_source_priority')
        but is not extensively used in this simulated implementation.

        Args:
            data: The input data, expected to be a string claim or a dictionary
                  containing a 'claim' key.
            context: A dictionary containing contextual information or configuration
                     for the processing.

        Returns:
            A dictionary containing the verification result:
            - 'original_claim': The claim as it was received.
            - 'normalized_claim': The claim after normalization (lowercase, stripped).
            - 'is_verified': True if the claim is factually correct, False if it's
                             a falsehood, or None if it cannot be verified by this node.
            - 'verification_status': A string describing the outcome (e.g., "verified",
                                     "debunked", "unverified_by_node", "invalid_input").
            - 'confidence': A float representing the confidence level (0.0 to 1.0).
            - 'sources': A list of strings indicating the source of verification (e.g.,
                         "internal_knowledge_base").

        Raises:
            ValueError: If the input `data` is not a string or a dictionary with a 'claim' key.
        """
        claim_text: str = ""
        result: Dict[str, Any] = {
            "original_claim": data,
            "normalized_claim": None,
            "is_verified": None,
            "verification_status": "unprocessed",
            "confidence": 0.0,
            "sources": [],
        }

        if isinstance(data, str):
            claim_text = data
        elif isinstance(data, dict) and "claim" in data and isinstance(data["claim"], str):
            claim_text = data["claim"]
        else:
            error_msg = (
                f"Invalid input data type for FactCheckerNode. Expected str or dict with 'claim' key, "
                f"but received {type(data)}."
            )
            logger.error(error_msg)
            result["verification_status"] = "invalid_input"
            result["error"] = error_msg
            return result

        result["original_claim"] = claim_text
        normalized_claim = claim_text.strip().lower()
        result["normalized_claim"] = normalized_claim

        logger.debug(f"Fact-checking claim: '{claim_text}' (normalized: '{normalized_claim}')")

        # Check against known facts
        if normalized_claim in self._known_facts:
            result["is_verified"] = self._known_facts[normalized_claim]
            result["verification_status"] = "verified" if result["is_verified"] else "debunked"
            result["confidence"] = 1.0
            result["sources"].append("internal_knowledge_base")
            logger.info(f"Claim '{claim_text}' {result['verification_status']} with high confidence.")
        # Check against known falsehoods
        elif normalized_claim in self._known_falsehoods:
            result["is_verified"] = self._known_falsehoods[normalized_claim]
            result["verification_status"] = "debunked" if not result["is_verified"] else "verified" # Should always be debunked here
            result["confidence"] = 1.0
            result["sources"].append("internal_knowledge_base")
            logger.info(f"Claim '{claim_text}' {result['verification_status']} with high confidence.")
        else:
            # If not found in internal knowledge base
            result["is_verified"] = None
            result["verification_status"] = "unverified_by_node"
            result["confidence"] = 0.5  # Neutral confidence for unverified claims by this node
            result["sources"].append("no_internal_match")
            logger.warning(f"Claim '{claim_text}' could not be verified by this FactCheckerNode's knowledge base.")

            # Example of using context (not fully implemented in simulation)
            if context.get("enable_external_api_check", False):
                logger.info("External API check enabled, but not implemented in this simulation.")
                # In a real scenario, this would call an external API
                # and update is_verified, verification_status, confidence, sources.

        return result
