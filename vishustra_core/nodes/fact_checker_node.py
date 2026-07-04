import logging
from typing import Any, Dict, Union

# Assume BaseNode is available at this path as per project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class FactCheckerNode(BaseNode):
    """
    A Vishustra processing node designed to simulate fact-checking of claims.

    This node takes an input claim (as a string or within a dictionary) and
    attempts to verify it against a small, internal knowledge base of facts.
    It returns a structured result indicating the verification status.
    """

    # A simple, static knowledge base for demonstration purposes.
    # In a real-world scenario, this would involve external API calls,
    # database lookups, or more sophisticated NLP models.
    _KNOWN_FACTS = {
        "the sky is blue": {"status": "verified", "reasoning": "Common observational fact, due to Rayleigh scattering of sunlight."},
        "water boils at 100 degrees celsius": {"status": "verified", "reasoning": "Standard boiling point of pure water at sea level (1 atmosphere)."},
        "the earth is flat": {"status": "debunked", "reasoning": "Overwhelming scientific and observational evidence confirms the Earth's oblate spheroid shape."},
        "vishustra is the best orchestration framework": {"status": "unverified", "reasoning": "Subjective claim, difficult to objectively verify without specific metrics and comparative analysis. Status depends on context and criteria."},
        "2 + 2 = 4": {"status": "verified", "reasoning": "Fundamental arithmetic truth."},
    }

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this processing node."""
        return "FactCheckerNode"

    def process(self, data: Union[str, Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to perform a simulated fact-check.

        The `data` input can be provided in two primary formats:
        1. A direct string representing the claim to be checked.
        2. A dictionary containing the claim under the key 'claim'.

        The `context` dictionary is available for passing runtime configuration
        or shared state across nodes, though it's not explicitly used in this
        simulated fact-checking logic.

        Args:
            data: The claim to be verified, either as a string or a dictionary.
            context: A dictionary holding contextual information for this process.

        Returns:
            A dictionary containing the following keys:
            - 'claim': The original claim that was processed.
            - 'status': A string indicating the verification status (e.g., 'verified',
                        'debunked', 'unverified', 'needs_review', 'error').
            - 'reasoning': A brief explanation for the assigned status.
            - 'original_input_type': The type of the input 'data'.
        """
        claim: str = ""
        original_data = data # Store for consistent output and logging
        input_type = type(data).__name__

        logger.debug(f"FactCheckerNode received input of type: {input_type}, data: {original_data}")

        if isinstance(data, str):
            claim = data
        elif isinstance(data, dict):
            # Attempt to extract the claim from a dictionary
            extracted_claim = data.get('claim')
            if extracted_claim is None:
                logger.warning(
                    f"FactCheckerNode received a dictionary without a 'claim' key. "
                    f"Returning 'error' status. Data: {original_data}"
                )
                return {
                    "claim": original_data,
                    "status": "error",
                    "reasoning": "Input dictionary is missing the 'claim' key.",
                    "original_input_type": input_type,
                }
            elif not isinstance(extracted_claim, str):
                logger.warning(
                    f"FactCheckerNode received a 'claim' value that is not a string "
                    f"in the input dictionary. Type: {type(extracted_claim).__name__}. Data: {original_data}"
                )
                return {
                    "claim": original_data,
                    "status": "error",
                    "reasoning": f"Value for 'claim' key must be a string, got {type(extracted_claim).__name__}.",
                    "original_input_type": input_type,
                }
            claim = extracted_claim
        else:
            # Handle unexpected data types gracefully
            logger.error(
                f"FactCheckerNode received an unexpected data type. Expected str or dict, "
                f"got {input_type}. Data: {original_data}"
            )
            return {
                "claim": original_data,
                "status": "error",
                "reasoning": f"Unexpected input data type: {input_type}. Expected str or dict.",
                "original_input_type": input_type,
            }

        # Validate the extracted or direct claim string
        if not claim or not claim.strip():
            logger.warning(
                f"FactCheckerNode received an empty or whitespace-only claim after extraction. "
                f"Original input: {original_data}"
            )
            return {
                "claim": claim,
                "status": "error",
                "reasoning": "The claim provided is empty or contains only whitespace.",
                "original_input_type": input_type,
            }

        # Normalize the claim for lookup (case-insensitive, trim whitespace)
        normalized_claim = claim.strip().lower()
        logger.debug(f"Attempting to fact-check normalized claim: '{normalized_claim}'")

        # Simulate fact-checking against the internal knowledge base
        if normalized_claim in self._KNOWN_FACTS:
            result = self._KNOWN_FACTS[normalized_claim]
            logger.info(f"Claim '{claim}' matched a known fact. Status: {result['status']}.")
            return {
                "claim": claim,
                "status": result["status"],
                "reasoning": result["reasoning"],
                "original_input_type": input_type,
            }
        else:
            logger.info(f"Claim '{claim}' not found in automated knowledge base. Marking as 'needs_review'.")
            return {
                "claim": claim,
                "status": "needs_review",
                "reasoning": "Claim not found in automated knowledge base; requires human verification or further processing.",
                "original_input_type": input_type,
            }
