
import logging
from typing import Any, Dict, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class FactCheckerNode(BaseNode):
    """
    A processing node designed to simulate fact-checking a given claim.

    This node takes a statement (claim) as input and attempts to verify its
    truthfulness against a simulated internal knowledge base. In a real-world
    scenario, this would involve integrating with external fact-checking APIs,
    semantic search over trusted sources, or advanced LLM reasoning.
    """

    # Simulated knowledge base for demonstration purposes.
    # This dictionary holds simple true/false statements.
    # In a production system, this would be replaced by a robust
    # data source or service interaction.
    _KNOWN_FACTS = {
        "water boils at 100 degrees celsius": True,
        "the earth is flat": False,
        "humans can fly naturally": False,
        "the sky is blue": True,
        "the sun orbits the earth": False,
        "vishustra is an llm orchestration framework": True,
        "elephants can jump": False,
        "birds are mammals": False,
        "penguins can fly": False,
        "chocolate is poisonous to dogs": True,
        "the great wall of china is visible from space": False,
        "napoleon was short": False, # Historically, he was average height for his era
    }

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "FactCheckerNode"

    def process(self, data: Union[str, Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data, attempting to fact-check a claim.

        The `data` input is expected to be either a string representing the claim
        directly, or a dictionary that contains a 'claim' key. The node attempts
        to verify this claim against its simulated internal knowledge base.

        The `context` dictionary is provided for broader orchestration, allowing
        for passing of additional metadata or session-specific information,
        though it is not extensively utilized in this specific simulation.

        Args:
            data: The claim to be fact-checked. Can be a string (the claim itself)
                  or a dictionary (e.g., `{"claim": "..."}`).
            context: A dictionary containing contextual information relevant to the
                     processing, such as user session data or external parameters.

        Returns:
            A dictionary containing the original claim, the fact-checking result,
            a confidence score, an explanation, and a list of sources checked.
            The `is_fact` field will be `True`, `False`, or `None` if unverified.

            Example output:
            {
                "original_claim": "The sky is blue.",
                "is_fact": True,
                "confidence": 0.95,
                "explanation": "Based on simulated knowledge base: 'The sky is blue' is recorded as true.",
                "sources_checked": ["Simulated_KB"]
            }

        Raises:
            ValueError: If the input `data` is not in an expected format (str or dict with 'claim').
            Exception: Propagates any unexpected errors encountered during the fact-checking simulation.
        """
        logger.info(f"[{self.node_name}] Starting fact-checking process for data input type: {type(data).__name__}")

        claim_text: str
        original_data_input = data # Retain original input for the output structure

        try:
            if isinstance(data, str):
                claim_text = data.strip()
            elif isinstance(data, dict):
                if 'claim' not in data:
                    logger.error(f"[{self.node_name}] Input dictionary missing required 'claim' key. Data: {data}")
                    raise ValueError("Input dictionary must contain a 'claim' key for fact-checking.")
                claim_text = str(data['claim']).strip()
            else:
                logger.error(f"[{self.node_name}] Invalid data type. Expected str or dict, received {type(data).__name__}.")
                raise ValueError(f"Invalid input data type. Expected str or dict with 'claim' key, got {type(data).__name__}. Cannot fact-check.")

            if not claim_text:
                logger.warning(f"[{self.node_name}] Received an empty claim for processing.")
                return {
                    "original_claim": original_data_input,
                    "is_fact": None,
                    "confidence": 0.0,
                    "explanation": "No valid claim text was provided to fact-check.",
                    "sources_checked": []
                }

            normalized_claim = claim_text.lower()
            result: Dict[str, Any] = {
                "original_claim": original_data_input,
                "is_fact": None,  # Represents unverified or needs more info
                "confidence": 0.0,
                "explanation": "Could not verify claim against available knowledge.",
                "sources_checked": []
            }

            if normalized_claim in self._KNOWN_FACTS:
                is_fact = self._KNOWN_FACTS[normalized_claim]
                result["is_fact"] = is_fact
                # Assign higher confidence if a direct match is found
                result["confidence"] = 0.95 if is_fact else 0.85
                result["explanation"] = f"Based on simulated knowledge base: '{claim_text}' is recorded as {'true' if is_fact else 'false'}."
                result["sources_checked"].append("Simulated_KB")
                logger.info(f"[{self.node_name}] Claim '{claim_text}' found in simulated KB. Is Fact: {is_fact}")
            else:
                # If not found in our simple KB, it's considered unverified for this simulation.
                # In a real system, this might trigger a call to a more advanced service.
                result["is_fact"] = None
                result["confidence"] = 0.1
                result["explanation"] = f"Claim '{claim_text}' not found in the simulated knowledge base. Further verification needed."
                logger.warning(f"[{self.node_name}] Claim '{claim_text}' not found in simulated KB. Marked as unverified.")

            logger.info(f"[{self.node_name}] Finished fact-checking for claim: '{claim_text}'. Result: Is Fact={result['is_fact']}")
            return result

        except ValueError as ve:
            logger.error(f"[{self.node_name}] Input validation error during fact-checking: {ve}")
            raise # Re-raise ValueError as it's an expected input failure
        except Exception as e:
            logger.exception(f"[{self.node_name}] An unexpected error occurred while processing claim '{original_data_input}': {e}")
            # Construct an error result before re-raising or returning a partial one
            error_result = {
                "original_claim": original_data_input,
                "is_fact": None,
                "confidence": 0.0,
                "explanation": f"An internal error prevented fact-checking: {type(e).__name__} - {e}",
                "sources_checked": []
            }
            # Depending on system design, one might return the error_result or re-raise.
            # For a critical processing error, re-raising is often preferred to halt the pipeline.
            raise
