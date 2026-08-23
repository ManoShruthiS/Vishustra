from vishustra_core.nodes.base_node import BaseNode
from typing import Any, Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class FactCheckerNode(BaseNode):
    """
    A processing node designed to simulate fact-checking a given claim.
    This node evaluates input data against a set of predefined simulated facts
    to determine its veracity, providing a structured output with confidence
    and reasoning.

    Input (`data`):
        - `str`: The claim directly as a string.
        - `Dict[str, Any]`: A dictionary expected to contain the claim under the key 'claim'.

    Output (`Dict[str, Any]`):
        A dictionary containing the fact-checking result:
        - `original_claim` (str): The claim as it was received.
        - `is_factual` (bool | None): `True` if factual, `False` if not, `None` if unverifiable.
        - `confidence` (float): A score from 0.0 to 1.0 indicating confidence in the factual assessment.
        - `evidence` (List[str]): A list of supporting or refuting points.
        - `reasoning` (str): A summary of the fact-checking process and outcome.
    """

    # For demonstration, this holds a very simplistic, hardcoded knowledge base.
    # In a real-world scenario, this would interface with a dedicated fact-checking
    # API or a robust internal knowledge graph.
    _SIMULATED_FACTS: List[Tuple[str, bool, float, str]] = [
        ("The sun rises in the east", True, 0.95, "Common astronomical observation and Earth's rotation."),
        ("Water boils at 100 degrees Celsius", True, 0.98, "Standard physical constant at sea level (1 atmosphere)."),
        ("Pigs can fly", False, 0.99, "Biologically impossible for pigs without external aid."),
        ("The moon is made of cheese", False, 0.99, "Geological composition is primarily silicate rock."),
        ("Blue is a color", True, 0.99, "It is a primary color in additive and subtractive color models."),
    ]

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "FactCheckerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to simulate a fact-checking operation.

        Args:
            data: The input claim, expected as a string or a dictionary
                  with a 'claim' key.
            context: A dictionary for contextual information. While not directly
                     used in this simulated version, it could contain credentials,
                     API endpoints, or configuration for external fact-checking services.

        Returns:
            A dictionary detailing the fact-checking result, including the original
            claim, its determined factual status, confidence, evidence, and reasoning.
        """
        claim_text: str = ""
        result: Dict[str, Any] = {
            "original_claim": None,
            "is_factual": None,
            "confidence": 0.0,
            "evidence": [],
            "reasoning": "Could not process claim due to internal error or invalid input."
        }

        try:
            if isinstance(data, str):
                claim_text = data.strip()
            elif isinstance(data, dict):
                # Attempt to extract claim from a dictionary, defaulting to empty string
                # if 'claim' key is missing or not a string.
                claim_text = str(data.get("claim", "")).strip()
            else:
                logger.warning(
                    f"FactCheckerNode received unsupported data type: {type(data)}. "
                    "Expected string or dictionary with 'claim' key."
                )
                result["reasoning"] = f"Invalid input data type: {type(data)}. Expected str or dict."
                return result

            if not claim_text:
                logger.warning("FactCheckerNode received an empty or no detectable claim to process.")
                result["reasoning"] = "No valid claim provided for fact-checking."
                return result

            result["original_claim"] = claim_text

            found_match = False
            # Simulate checking the claim against the predefined facts
            for fact_fragment, is_true, confidence, reason in self._SIMULATED_FACTS:
                if fact_fragment.lower() in claim_text.lower():
                    result["is_factual"] = is_true
                    result["confidence"] = confidence
                    result["evidence"].append(f"Identified phrase matching known fact: '{fact_fragment}'.")
                    result["reasoning"] = reason
                    found_match = True
                    break # In this simple simulation, we take the first matching fact.

            if not found_match:
                # If no direct match is found, the claim is considered unverifiable by this system.
                result["is_factual"] = None
                result["confidence"] = 0.1
                result["evidence"].append("No direct match found in the simulated knowledge base for verification.")
                result["reasoning"] = "Claim could not be definitively verified or refuted with the current simulated facts."

        except Exception as e:
            logger.error(f"An unexpected error occurred during FactCheckerNode processing for data: {data}. Error: {e}", exc_info=True)
            result["is_factual"] = None
            result["confidence"] = 0.0
            result["evidence"] = []
            result["reasoning"] = f"An unexpected system error prevented fact-checking: {str(e)}"

        logger.debug(f"FactCheckerNode processed claim '{claim_text[:100]}...'. Result: {result['is_factual']}, Confidence: {result['confidence']}")
        return result