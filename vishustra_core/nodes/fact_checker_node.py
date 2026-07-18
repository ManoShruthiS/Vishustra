import logging
from typing import Any, Dict, List, Union
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class FactCheckerNode(BaseNode):
    """
    A processing node that simulates fact-checking on textual content.

    This node expects input data to be a dictionary, typically containing
    a 'text' key with the content to be fact-checked and optionally
    'claims_to_verify' for specific claims. It enriches the data
    with 'fact_check_results'.

    The current implementation uses a simplistic rule-based simulation.
    In a real-world scenario, this would integrate with external fact-checking
    APIs or knowledge bases.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "FactCheckerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to simulate fact-checking.

        Expects `data` to be a dictionary, potentially containing:
        - 'text' (str): The main content string to be fact-checked.
        - 'claims_to_verify' (List[str], optional): Specific claims within
          the text to verify. If not present, a general check is simulated.

        Args:
            data (Any): The input data. Expected to be a dictionary.
            context (Dict[str, Any]): The operational context dictionary.

        Returns:
            Any: The input data dictionary augmented with 'fact_check_results'.
                 Returns the original data if processing fails or input format
                 is unexpected, after logging an error.

        Raises:
            TypeError: If the input `data` is not a dictionary.
        """
        if not isinstance(data, dict):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected dict, got {type(data)}."
            )
            raise TypeError(
                f"[{self.node_name}] Input data must be a dictionary."
            )

        text_to_check: str = data.get("text", "")
        claims_to_verify: List[str] = data.get("claims_to_verify", [])
        fact_check_results: Dict[str, Dict[str, str]] = {}

        if not text_to_check:
            logger.warning(
                f"[{self.node_name}] 'text' key not found or empty in input data. "
                "Skipping detailed fact-checking."
            )
            # Add a placeholder result indicating no text was found
            fact_check_results["_general_status"] = {
                "status": "UNVERIFIED",
                "reason": "No text content provided for fact-checking."
            }
            data["fact_check_results"] = fact_check_results
            return data

        logger.info(
            f"[{self.node_name}] Initiating fact-check for text (length: {len(text_to_check)}) "
            f"with {len(claims_to_verify)} specific claims."
        )

        try:
            # --- SIMULATED FACT-CHECKING LOGIC ---
            # In a real system, this would involve calling an external API,
            # querying a knowledge graph, or using a sophisticated NLP model.
            # For this simulation, we use a simple keyword-based approach.

            simulated_knowledge_base = {
                "the sky is blue": {"status": "TRUE", "evidence": "Scientific observation of Rayleigh scattering."},
                "birds can fly": {"status": "TRUE", "evidence": "Common biological characteristic of most bird species."},
                "fish can climb trees": {"status": "FALSE", "evidence": "Fish are aquatic animals adapted to water environments."},
                "humans have three eyes": {"status": "FALSE", "evidence": "Human anatomy typically includes two eyes."},
            }

            if not claims_to_verify:
                # If no specific claims, try to find general statements in the text
                potential_claims = self._extract_potential_claims(text_to_check)
                if not potential_claims:
                    potential_claims = [text_to_check[:100].lower() + "..."] # Take a snippet if nothing specific
                claims_to_verify = potential_claims
                logger.info(f"[{self.node_name}] No specific claims provided. Auto-extracted {len(claims_to_verify)} potential claims.")

            for claim in claims_to_verify:
                normalized_claim = claim.strip().lower()
                if normalized_claim in simulated_knowledge_base:
                    fact_check_results[claim] = simulated_knowledge_base[normalized_claim]
                elif any(phrase in normalized_claim for phrase in ["sky is green", "sun is cold"]):
                    fact_check_results[claim] = {
                        "status": "FALSE",
                        "evidence": "Contradicts fundamental scientific facts."
                    }
                elif any(phrase in normalized_claim for phrase in ["water is wet", "fire is hot"]):
                    fact_check_results[claim] = {
                        "status": "TRUE",
                        "evidence": "Generally accepted physical properties."
                    }
                else:
                    fact_check_results[claim] = {
                        "status": "UNVERIFIED",
                        "evidence": "Could not verify using available knowledge base. Requires further investigation."
                    }
            # --- END SIMULATED FACT-CHECKING LOGIC ---

            data["fact_check_results"] = fact_check_results
            logger.info(
                f"[{self.node_name}] Fact-checking completed for {len(claims_to_verify)} claims. "
                f"Results: {fact_check_results}"
            )
            return data

        except Exception as e:
            logger.exception(
                f"[{self.node_name}] An unexpected error occurred during fact-checking: {e}"
            )
            # On error, return data with an error message in results
            data["fact_check_results"] = {
                "_error": {
                    "status": "ERROR",
                    "reason": f"Failed to perform fact-check due to an internal error: {e}",
                    "details": str(e)
                }
            }
            return data

    def _extract_potential_claims(self, text: str) -> List[str]:
        """
        A very basic simulated method to extract potential claims from text.
        In a real scenario, this would use NLP techniques (e.g., dependency parsing,
        open information extraction) to identify factual statements.
        """
        # For simulation, just split by common punctuation or assume simple sentences
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        claims = []
        for sentence in sentences:
            if len(sentence) > 10: # Avoid very short fragments
                claims.append(sentence)
        return claims[:3] # Limit to top 3 potential claims for brevity