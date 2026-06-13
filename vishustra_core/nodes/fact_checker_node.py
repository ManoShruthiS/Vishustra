import logging
from typing import Any, Dict, List, Optional

# Assuming vishustra_core is a package at the root of the project
# and base_node.py is inside vishustra_core/nodes/
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class FactCheckerNode(BaseNode):
    """
    A processing node that simulates fact-checking for given claims.

    This node expects a dictionary as input data, containing a 'claim' key.
    It returns a dictionary indicating the factual status, confidence,
    and any supporting details.
    """

    _KNOWN_FACTS = {
        "The sky is blue.": {"is_factual": True, "confidence": 0.95, "reasons": ["Common knowledge", "Scientific observation"], "sources": ["General science"]},
        "Water boils at 100 degrees Celsius at standard atmospheric pressure.": {"is_factual": True, "confidence": 0.99, "reasons": ["Physics principle"], "sources": ["Thermodynamics textbooks"]},
        "Pigs can fly.": {"is_factual": False, "confidence": 0.98, "reasons": ["Biological impossibility"], "sources": ["Biology"]},
        "The Earth is flat.": {"is_factual": False, "confidence": 0.99, "reasons": ["Extensive scientific evidence of spherical shape"], "sources": ["Astronomy", "Geodesy"]},
    }

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "FactCheckerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to perform a simulated fact-check on a claim.

        Args:
            data: The input data, expected to be a dictionary with a 'claim' key
                  containing the string to be fact-checked.
            context: A dictionary containing contextual information or
                     configuration for the node.

        Returns:
            A dictionary containing the original claim, its factual status,
            confidence, reasons, sources, and an optional error message.

            Example successful output:
            {
                "original_claim": "The sky is blue.",
                "is_factual": True,
                "confidence": 0.95,
                "reasons": ["Common knowledge"],
                "sources": ["General science"],
                "error": None
            }

            Example error output:
            {
                "original_claim": None,
                "is_factual": None,
                "confidence": None,
                "reasons": [],
                "sources": [],
                "error": "Input data must be a dictionary with a 'claim' key."
            }
        """
        logger.info(f"[{self.node_name}] Starting fact-checking process.")

        result: Dict[str, Any] = {
            "original_claim": None,
            "is_factual": None,
            "confidence": None,
            "reasons": [],
            "sources": [],
            "error": None
        }

        # Validate input data structure
        if not isinstance(data, dict):
            error_msg = f"[{self.node_name}] Invalid input data: expected a dictionary, got {type(data).__name__}."
            logger.error(error_msg)
            result["error"] = error_msg
            return result

        if "claim" not in data or not isinstance(data["claim"], str):
            error_msg = f"[{self.node_name}] Missing or invalid 'claim' in input data: expected a string under 'claim' key."
            logger.error(error_msg)
            result["error"] = error_msg
            return result

        claim = data["claim"]
        result["original_claim"] = claim

        # Simulate fact-checking using an internal knowledge base
        if claim in self._KNOWN_FACTS:
            fact_info = self._KNOWN_FACTS[claim]
            result.update(fact_info)
            logger.info(f"[{self.node_name}] Claim '{claim}' found in known facts. Is factual: {fact_info['is_factual']}.")
        else:
            # For unknown claims, simulate an "undetermined" state with low confidence
            # or a default assumption.
            logger.warning(f"[{self.node_name}] Claim '{claim}' not found in internal knowledge base. Simulating undetermined status.")
            result["is_factual"] = False # Default to 'false' for unknown, or 'undetermined' depending on policy
            result["confidence"] = 0.35 # Low confidence
            result["reasons"] = ["Claim not directly verifiable in internal knowledge base.", "Further investigation required."]
            result["sources"] = ["Simulated data"]

        logger.info(f"[{self.node_name}] Fact-checking complete for claim: '{claim}'.")
        return result

if __name__ == "__main__":
    # Example usage for testing purposes
    logging.basicConfig(level=logging.LINFO) # Configure logging for example run

    fact_checker = FactCheckerNode()
    dummy_context: Dict[str, Any] = {}

    print(f"\n--- Testing {fact_checker.node_name} ---")

    # Test case 1: Known true fact
    data_true = {"claim": "The sky is blue."}
    print(f"\nProcessing: {data_true['claim']}")
    result_true = fact_checker.process(data_true, dummy_context)
    print(f"Result: {result_true}")
    assert result_true["is_factual"] is True
    assert result_true["confidence"] == 0.95

    # Test case 2: Known false fact
    data_false = {"claim": "Pigs can fly."}
    print(f"\nProcessing: {data_false['claim']}")
    result_false = fact_checker.process(data_false, dummy_context)
    print(f"Result: {result_false}")
    assert result_false["is_factual"] is False
    assert result_false["confidence"] == 0.98

    # Test case 3: Unknown claim
    data_unknown = {"claim": "Vishustra is the best framework."}
    print(f"\nProcessing: {data_unknown['claim']}")
    result_unknown = fact_checker.process(data_unknown, dummy_context)
    print(f"Result: {result_unknown}")
    assert result_unknown["is_factual"] is False # Due to our simulation policy
    assert result_unknown["confidence"] == 0.35
    assert "Further investigation required." in result_unknown["reasons"]

    # Test case 4: Invalid input (not a dict)
    data_invalid_type = "This is not a dict."
    print(f"\nProcessing: {data_invalid_type}")
    result_invalid_type = fact_checker.process(data_invalid_type, dummy_context)
    print(f"Result: {result_invalid_type}")
    assert result_invalid_type["error"] is not None
    assert result_invalid_type["error"].startswith("[FactCheckerNode] Invalid input data:")

    # Test case 5: Invalid input (dict without 'claim')
    data_missing_claim = {"text": "Some text."}
    print(f"\nProcessing: {data_missing_claim}")
    result_missing_claim = fact_checker.process(data_missing_claim, dummy_context)
    print(f"Result: {result_missing_claim}")
    assert result_missing_claim["error"] is not None
    assert result_missing_claim["error"].startswith("[FactCheckerNode] Missing or invalid 'claim' in input data:")

    print("\n--- All tests passed (simulated logic) ---")