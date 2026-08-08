import logging
from typing import Any, Dict, Union

# Assuming vishustra_core.nodes.base_node exists as per project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class FactCheckerNode(BaseNode):
    """
    A Vishustra processing node that simulates fact-checking of claims.

    This node attempts to determine the veracity of a given claim by comparing
    it against a set of 'known facts' and 'disputed claims' provided via the
    processing context. If no external data is provided, it uses internal
    mock data for demonstration.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "FactCheckerNode"

    def process(self, data: Union[str, Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data, attempting to verify a claim.

        The input `data` can be a string representing the claim directly, or a dictionary
        containing the claim under the key 'claim'.

        The `context` dictionary can optionally include:
        - 'known_facts' (Dict[str, bool]): A dictionary where keys are factual statements
                                            and values are boolean indicating verified status.
        - 'disputed_claims' (Dict[str, str]): A dictionary where keys are claims
                                               and values are reasons for dispute or evidence.

        Args:
            data (Union[str, Dict[str, Any]]): The claim to be checked. Expected as a string
                                                or a dictionary with a 'claim' key whose value is a string.
            context (Dict[str, Any]): A dictionary providing environmental or configuration data,
                                      potentially including 'known_facts' and 'disputed_claims' databases.

        Returns:
            Dict[str, Any]: A dictionary containing the original claim, its fact-checking result
                            ('verified', 'disputed', 'unsubstantiated'), supporting evidence,
                            and a confidence score.

        Raises:
            ValueError: If the input data is not in an expected format or a valid claim
                        string cannot be extracted.
            RuntimeError: If an unexpected error occurs during the fact-checking process.
        """
        claim_to_check: str = ""
        try:
            if isinstance(data, str):
                claim_to_check = data
            elif isinstance(data, dict) and 'claim' in data and isinstance(data['claim'], str):
                claim_to_check = data['claim']
            else:
                logger.error(
                    f"FactCheckerNode received invalid input data type: {type(data)}. "
                    "Expected a string or a dictionary with a 'claim' key (string value)."
                )
                raise ValueError(
                    "FactCheckerNode requires input data as a string or a dictionary "
                    "with a 'claim' key containing a string value."
                )

            if not claim_to_check.strip():
                logger.warning("FactCheckerNode received an empty or whitespace-only claim.")
                return {
                    "original_claim": claim_to_check,
                    "status": "invalid_claim",
                    "evidence": "Claim was empty or whitespace-only.",
                    "confidence": 0.0
                }

            # Retrieve fact databases from context or use internal mocks if not provided
            known_facts = context.get('known_facts', {
                "The Earth orbits the Sun.": True,
                "Water boils at 100 degrees Celsius at sea level.": True,
                "Python is a versatile programming language.": True,
                "The capital of France is Paris.": True,
            })
            disputed_claims = context.get('disputed_claims', {
                "Humans only use 10% of their brain.": "Scientific consensus indicates humans use all parts of their brain.",
                "The Great Wall of China is visible from space.": "Only barely visible under specific conditions, not with the naked eye.",
                "Chewing gum stays in your stomach for seven years.": "Gum is not digestible but passes through the digestive system relatively quickly.",
            })
            
            # Initialize result structure
            result: Dict[str, Any] = {
                "original_claim": claim_to_check,
                "status": "unsubstantiated",
                "evidence": None,
                "confidence": 0.5
            }

            # Normalize claim for basic comparison (a real system would use NLP/embeddings)
            normalized_claim = claim_to_check.strip().lower()

            # Check against known facts
            for fact_statement, is_verified in known_facts.items():
                if is_verified and normalized_claim == fact_statement.lower():
                    result["status"] = "verified"
                    result["evidence"] = f"Matches a known factual statement: '{fact_statement}'."
                    result["confidence"] = 0.95
                    logger.info(f"Claim '{claim_to_check}' verified.")
                    return result

            # Check against disputed claims
            for disputed_statement, dispute_reason in disputed_claims.items():
                if normalized_claim == disputed_statement.lower():
                    result["status"] = "disputed"
                    result["evidence"] = dispute_reason
                    result["confidence"] = 0.1
                    logger.warning(f"Claim '{claim_to_check}' found to be disputed.")
                    return result

            # If no match found
            logger.info(f"Claim '{claim_to_check}' remains unsubstantiated (no direct match in known facts or disputes).")
            return result

        except ValueError:
            # Re-raise ValueError as it indicates an issue with input data format
            raise
        except Exception as e:
            # Catch any other unexpected errors and wrap them in a RuntimeError
            logger.exception(f"An unexpected error occurred during FactCheckerNode processing for data: {data}")
            raise RuntimeError(f"FactCheckerNode failed to process data due to an internal error: {e}") from e