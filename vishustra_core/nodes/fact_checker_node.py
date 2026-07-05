import logging
from typing import Any, Dict

# Assuming vishustra_core.nodes.base_node is available in the project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class FactCheckerNode(BaseNode):
    """
    A Vishustra node designed to simulate fact-checking textual claims.

    This node takes a string as input, representing a claim, and processes it
    to determine a verdict (e.g., TRUE, FALSE, NEEDS_MORE_INFO) along with
    simulated reasoning. In a production environment, this would integrate
    with external fact-checking services, knowledge bases, or advanced
    language models for verification.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "FactCheckerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to simulate fact-checking a claim.

        The `data` argument is expected to be a string representing the claim
        to be verified. The `context` dictionary can be used to pass
        configuration parameters, external service clients, or other
        runtime information relevant for the fact-checking process.

        Args:
            data: The textual claim to be fact-checked (expected type: str).
            context: A dictionary containing contextual information,
                     such as configuration settings or client instances.

        Returns:
            A dictionary containing the original claim, its simulated verdict,
            and a reasoning statement.
            Example:
            {
                "claim": "Vishustra is a highly modular LLM orchestration framework.",
                "verdict": "TRUE",
                "reasoning": "Based on project documentation and core design principles."
            }

        Raises:
            TypeError: If the input `data` is not a string.
            ValueError: If the input `data` string is empty or contains only whitespace.
        """
        if not isinstance(data, str):
            logger.error(
                "FactCheckerNode: Invalid data type received. Expected 'str', but got '%s'.",
                type(data).__name__
            )
            raise TypeError(
                f"FactCheckerNode requires 'data' to be a string, "
                f"but received type '{type(data).__name__}'."
            )

        claim = data.strip()
        if not claim:
            logger.warning("FactCheckerNode: Received an empty or whitespace-only claim string.")
            raise ValueError("FactCheckerNode cannot process an empty claim.")
        
        # Retrieve max_claim_length from context or use a default
        max_claim_length = context.get('max_claim_length', 1024)
        if len(claim) > max_claim_length:
            logger.warning(
                "FactCheckerNode: Claim length (%d) exceeds configured maximum (%d). "
                "Consider refining input data or increasing 'max_claim_length' in context.",
                len(claim), max_claim_length
            )
            # Depending on requirements, one might choose to truncate, reject, or just log.
            # For this simulation, we proceed but log the warning.

        logger.info("FactCheckerNode: Initiating fact-check for claim: '%s'", claim[:100] + ('...' if len(claim) > 100 else ''))

        verdict: str = "NEEDS_MORE_INFO"
        reasoning: str = "Automated verification could not find definitive evidence."

        # --- Simulate Fact-Checking Logic ---
        # In a real-world scenario, this section would involve:
        # 1. Making API calls to external fact-checking services.
        # 2. Querying structured knowledge bases or internal data sources.
        # 3. Interacting with LLMs to evaluate the claim against provided context or external data.
        # For this example, we use simple keyword matching to simulate different verdicts.

        claim_lower = claim.lower()

        if "vishustra is an llm orchestration framework" in claim_lower or \
           "vishustra is a highly modular llm orchestration framework" in claim_lower:
            verdict = "TRUE"
            reasoning = "Based on official project documentation and core objectives."
        elif "vishustra is a frontend framework" in claim_lower or \
             "vishustra is built in java" in claim_lower:
            verdict = "FALSE"
            reasoning = "Vishustra is a backend LLM orchestration framework, primarily in Python, not a frontend or Java-based one."
        elif "vishustra" in claim_lower and "python" in claim_lower:
            verdict = "TRUE"
            reasoning = "Vishustra is indeed developed using Python as its primary programming language."
        
        # This part of the simulation allows overriding verdicts via context for testing purposes
        mock_verdicts = context.get('mock_verdicts', {})
        if claim in mock_verdicts:
            verdict = mock_verdicts[claim].get('verdict', verdict)
            reasoning = mock_verdicts[claim].get('reasoning', reasoning)
            logger.debug("FactCheckerNode: Applied mock verdict from context for claim: '%s'", claim)

        logger.info("FactCheckerNode: Fact-check completed. Verdict: %s for claim: '%s'", verdict, claim[:100] + ('...' if len(claim) > 100 else ''))

        return {
            "claim": claim,
            "verdict": verdict,
            "reasoning": reasoning,
        }
