import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class FactCheckerNode(BaseNode):
    """
    A Vishustra processing node that simulates fact-checking a given statement or claim.

    This node takes a string as input, representing a claim, and attempts to
    verify its truthfulness based on a simplified internal knowledge base.
    In a real-world scenario, this would involve integrating with external
    fact-checking APIs, knowledge graphs, or sophisticated NLP models.
    """

    def __init__(self):
        """
        Initializes the FactCheckerNode.

        For this simulation, a basic set of known truths and falsehoods is established.
        """
        super().__init__()
        logger.debug("FactCheckerNode initialized with a simulated knowledge base.")
        # A very basic, simulated internal knowledge base for demonstration purposes
        self._known_truths = [
            "water is wet",
            "the sky is blue during the day",
            "earth revolves around the sun",
        ]
        self._known_falsehoods = [
            "the sun is purple",
            "pigs can fly",
            "the moon is made of cheese",
        ]

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node.
        """
        return "FactChecker"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to simulate fact-checking.

        Expects `data` to be a string representing the claim to be checked.
        Returns a dictionary containing the original claim, its fact-check status,
        and a reason for the given status.

        Args:
            data: The input data, expected to be a string containing a claim.
            context: A dictionary containing context-specific information for the node.

        Returns:
            A dictionary with keys:
            - "original_claim": The claim as it was received.
            - "fact_check_status": One of "TRUE", "FALSE", "UNVERIFIED", "UNSUPPORTED_CLAIM".
            - "reason": A brief explanation for the status.

        Raises:
            TypeError: If the input `data` is not a string.
            RuntimeError: For unexpected internal processing errors.
        """
        if not isinstance(data, str):
            logger.error(
                f"FactCheckerNode received invalid data type. Expected str, got {type(data)}."
            )
            raise TypeError(
                f"FactCheckerNode requires input `data` to be a string, but received {type(data)}."
            )

        original_claim = data
        normalized_claim = data.strip().lower()  # Normalize for simple comparison
        fact_check_status = "UNVERIFIED"
        reason = "Claim could not be definitively verified or debunked by the internal knowledge base."

        logger.info(f"Attempting to fact-check claim: '{original_claim[:100]}...'")

        try:
            # Simulate fact-checking using simple keyword matching
            for truth in self._known_truths:
                if truth in normalized_claim:
                    fact_check_status = "TRUE"
                    reason = "Matches a known true statement."
                    break

            if fact_check_status == "UNVERIFIED":  # Only check falsehoods if not already true
                for falsehood in self._known_falsehoods:
                    if falsehood in normalized_claim:
                        fact_check_status = "FALSE"
                        reason = "Contains information known to be false."
                        break

            # If still unverified and contains common claim indicators, mark as unsupported.
            if fact_check_status == "UNVERIFIED" and any(
                keyword in normalized_claim
                for keyword in ["claim:", "i believe", "it is said"]
            ):
                fact_check_status = "UNSUPPORTED_CLAIM"
                reason = "Claim identified, but no definitive evidence found in internal knowledge base."

            logger.info(
                f"Fact-check completed for claim: '{original_claim[:50]}...'. Status: {fact_check_status}"
            )

            return {
                "original_claim": original_claim,
                "fact_check_status": fact_check_status,
                "reason": reason,
            }

        except Exception as e:
            logger.exception(
                f"An unexpected error occurred during fact-checking process for claim: '{original_claim[:100]}...'"
            )
            # Re-raise as a RuntimeError to indicate a processing failure
            raise RuntimeError(
                f"FactCheckerNode failed to process claim due to an internal error: {e}"
            ) from e