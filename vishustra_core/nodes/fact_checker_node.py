
import logging
import datetime
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

# Initialize logger for this module
logger = logging.getLogger(__name__)

class FactCheckerNode(BaseNode):
    """
    A processing node that simulates fact-checking for input claims.

    This node takes a claim (as a string, or within a dictionary) or a list of claims
    and returns a simulated fact-checking verdict along with supporting information.
    It's designed to be a placeholder for integration with real-world fact-checking
    APIs or knowledge bases.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node.
        """
        return "FactCheckerNode"

    def _check_single_claim(self, claim: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulates the fact-checking process for a single claim.
        In a production environment, this method would integrate with external fact-checking
        services, databases, or sophisticated NLP models to determine veracity.

        Args:
            claim (str): The specific claim text to be fact-checked.
            context (Dict[str, Any]): Contextual information which might include
                                       parameters for the fact-checking logic.

        Returns:
            Dict[str, Any]: A dictionary containing the original claim, a verdict,
                            confidence score, and simulated evidence.
        """
        verdict: str = "NEEDS_MORE_INFO"
        evidence: List[str] = []
        confidence: float = 0.5  # Scale from 0.0 (low confidence) to 1.0 (high confidence)

        lower_claim = claim.lower()

        # --- Simple Keyword-Based Simulation Logic ---
        # This section simulates how a fact-checker might determine a verdict.
        # In reality, this would be much more complex, involving data retrieval,
        # natural language understanding, and evidence synthesis.

        if "sun is hot" in lower_claim or "earth revolves around sun" in lower_claim:
            verdict = "TRUE"
            evidence.append("Common knowledge and scientific consensus.")
            confidence = 0.95
        elif "moon is made of cheese" in lower_claim or "pigs can fly" in lower_claim:
            verdict = "FALSE"
            evidence.append("Empirical and scientific evidence disproves this claim.")
            confidence = 0.98
        elif "water is wet" in lower_claim: # A common semantic debate point
            verdict = "TRUE"
            evidence.append("Standard definition of 'wetness' implies being covered or saturated with liquid.")
            confidence = 0.85
        elif "cats are dogs" in lower_claim:
            verdict = "FALSE"
            evidence.append("Biological classification distinguishes between Felidae and Canidae families.")
            confidence = 1.0
        else:
            # Simulate influence from context parameters
            if context.get("assume_positive_for_unverified", False):
                verdict = "PLAUSIBLE"
                evidence.append("No immediate disproving evidence found; contextual setting suggests positive assumption.")
                confidence = 0.6
            elif context.get("source_reliability", 0.0) > 0.8:
                 verdict = "LIKELY_TRUE"
                 evidence.append("Claim originates from a generally reliable source, but not independently verified here.")
                 confidence = 0.75
            else:
                verdict = "NEEDS_MORE_INFO"
                evidence.append("Insufficient data or clear evidence available to confirm or deny this claim.")
                confidence = 0.5

        logger.debug(
            f"Fact-checked claim '{claim[:70]}{'...' if len(claim) > 70 else ''}': "
            f"Verdict='{verdict}' (Confidence: {confidence:.2f})"
        )

        return {
            "original_claim": claim,
            "verdict": verdict,
            "confidence": confidence,
            "evidence": evidence,
            "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z"
        }

    def process(self, data: Any, context: Dict[str, Any]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Processes the input data, attempting to fact-check contained claims.

        Args:
            data (Any): The input data for fact-checking. Expected types are:
                        - A `str` representing a single claim.
                        - A `dict` expected to contain a claim under keys like 'claim', 'text', or 'content'.
                        - A `list` of `str` or `dict`, where each item is a claim to be processed.
            context (Dict[str, Any]): A dictionary containing contextual information or
                                       configuration for the node, such as API keys,
                                       source reliability thresholds, or flags for assumptions.

        Returns:
            Union[Dict[str, Any], List[Dict[str, Any]]]:
                - If the input `data` was a single `str` or `dict`, returns a single
                  dictionary containing the fact-checking results, potentially merged
                  with the original dictionary data.
                - If the input `data` was a `list`, returns a list of result dictionaries.

        Raises:
            ValueError: If the input `data` type is unsupported or if a `dict`
                        does not contain a recognizable claim.
        """
        if isinstance(data, str):
            logger.info(f"Processing single string claim: '{data[:70]}{'...' if len(data) > 70 else ''}'")
            return self._check_single_claim(data, context)

        elif isinstance(data, dict):
            claim_keys = ["claim", "text", "content"]
            claim_text = None
            found_key = None

            for key in claim_keys:
                if key in data and isinstance(data[key], str):
                    claim_text = data[key]
                    found_key = key
                    break
            
            if claim_text is not None:
                logger.info(
                    f"Processing dictionary claim found under key '{found_key}': "
                    f"'{claim_text[:70]}{'...' if len(claim_text) > 70 else ''}'"
                )
                result = self._check_single_claim(claim_text, context)
                # Merge original dictionary data with the fact-checking results
                return {**data, **result}
            else:
                error_msg = (
                    f"Input dictionary does not contain a recognizable claim "
                    f"in expected keys: {', '.join(claim_keys)}. "
                    f"Keys found: {list(data.keys())}."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

        elif isinstance(data, list):
            logger.info(f"Processing a list of {len(data)} potential claims.")
            results = []
            for i, item in enumerate(data):
                try:
                    # Recursively call process for each item to handle mixed list content (str or dict)
                    item_result = self.process(item, context)
                    results.append(item_result)
                except ValueError as e:
                    logger.warning(
                        f"Skipping item at index {i} due to processing error in FactCheckerNode: {e}. "
                        f"Original item: {item!r}"
                    )
                    # Append an error indicator for the failed item
                    results.append({
                        "original_item": item,
                        "status": "failed_to_process",
                        "error_message": str(e),
                        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z"
                    })
            return results

        else:
            error_msg = (
                f"Unsupported data type for FactCheckerNode: '{type(data).__name__}'. "
                f"Expected 'str', 'dict', or 'list'."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

