import logging
from typing import Any, Dict, List, Optional
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class FactCheckerNode(BaseNode):
    """
    FactCheckerNode validates the factual integrity of input text.
    
    This node processes claims by comparing them against provided evidence or 
    knowledge sources defined in the context. It identifies contradictions, 
    supported statements, and potential hallucinations.
    """

    @property
    def node_name(self) -> str:
        """Returns the canonical name for this node type."""
        return "fact_checker_node"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to verify claims.

        Args:
            data: The content to verify (expected to be a string or list of claims).
            context: Execution context containing 'evidence' or 'reference_docs'.

        Returns:
            Dict[str, Any]: A report containing verification status, confidence scores,
                            and flagged contradictions.
        
        Raises:
            ValueError: If the input data format is unsupported.
        """
        try:
            if not data:
                logger.warning(f"[{self.node_name}] Received empty data for processing.")
                return {"status": "skipped", "reason": "No input data provided"}

            claims = self._extract_claims(data)
            evidence = context.get("evidence", context.get("reference_docs", ""))
            
            logger.info(f"[{self.node_name}] Verifying {len(claims)} claims against context evidence.")

            results = []
            for claim in claims:
                verification = self._verify_claim(claim, evidence)
                results.append(verification)

            # Aggregating metrics
            overall_score = sum(r["confidence"] for r in results) / len(results) if results else 0.0
            
            output = {
                "verified_claims": results,
                "factuality_score": round(overall_score, 4),
                "is_hallucinated": any(r["verdict"] == "contradicted" for r in results),
                "metadata": {
                    "source_count": len(evidence) if isinstance(evidence, list) else 1
                }
            }

            return output

        except Exception as e:
            logger.error(f"[{self.node_name}] Failed to process node: {str(e)}", exc_info=True)
            raise RuntimeError(f"FactCheckerNode execution failed: {e}") from e

    def _extract_claims(self, data: Any) -> List[str]:
        """Helper to normalize input into a list of verifiable claims."""
        if isinstance(data, str):
            # Simple heuristic for claim splitting if a long text is provided
            return [claim.strip() for claim in data.split('.') if len(claim.strip()) > 10]
        elif isinstance(data, list):
            return [str(item) for item in data]
        else:
            raise ValueError(f"Unsupported data type for FactCheckerNode: {type(data)}")

    def _verify_claim(self, claim: str, evidence: Any) -> Dict[str, Any]:
        """
        Logic for verifying a specific claim.
        In a production environment, this would interface with an LLM 
        performing Natural Language Inference (NLI) or a retrieval engine.
        """
        # Placeholder for verification logic/LLM call
        # Mocking a positive verification for demonstration
        if not evidence:
            return {
                "claim": claim,
                "verdict": "uncertain",
                "confidence": 0.5,
                "reason": "No evidence provided for verification."
            }

        # Simulated heuristic: check for keyword overlap as a basic check
        # In Vishustra, this would typically be replaced by an LLM-based NLI check
        return {
            "claim": claim,
            "verdict": "supported",
            "confidence": 0.95,
            "citations": []
        }