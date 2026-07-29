import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class FactCheckerNode(BaseNode):
    """
    A Vishustra processing node that simulates fact-checking of a given statement.

    This node takes a statement as input and returns a simulated fact-check result
    (e.g., 'TRUE', 'FALSE', 'NEEDS_MORE_INFO', 'UNVERIFIED') along with a
    confidence score and explanatory details.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "FactCheckerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to simulate fact-checking a statement.

        The `context` dictionary is available for more advanced scenarios (e.g.,
        external API keys, session-specific parameters), but is not used in this
        simulated fact-checking logic.

        Expected `data` format:
        A dictionary containing at least a "statement" key.
        Optionally, it can include a "sources" key with a list of strings.
        Example:
        ```json
        {
            "statement": "The sky is green.",
            "sources": ["nasa.gov/color_of_sky"]
        }
        ```

        Returns:
        A dictionary containing the original statement, the simulated fact-check
        result, a confidence score, and explanatory details.
        Example:
        ```json
        {
            "statement": "The sky is green.",
            "fact_check_result": "FALSE",
            "confidence": 0.95,
            "details": "Statement contradicts widely accepted facts.",
            "checked_sources": ["nasa.gov/color_of_sky"]
        }
        ```

        Raises:
            ValueError: If the input data format is invalid or missing the 'statement' key.
            RuntimeError: If an unexpected error occurs during the fact-checking simulation.
        """
        logger.debug(f"[{self.node_name}] Starting process for input data.")

        if not isinstance(data, dict) or "statement" not in data:
            logger.error(
                f"[{self.node_name}] Invalid input data format. Expected a dictionary "
                f"with a 'statement' key. Received: {type(data).__name__}."
            )
            raise ValueError(
                "Input data must be a dictionary containing a 'statement' key."
            )

        statement = str(data["statement"]).strip()
        sources = data.get("sources", [])

        # --- Simulate Fact-Checking Logic ---
        # In a production system, this section would integrate with advanced
        # NLP models, external fact-checking APIs, knowledge graphs, or a
        # dedicated verification pipeline. This simulation uses simple keyword
        # matching for demonstration purposes.
        fact_check_result: str
        confidence: float
        details: str

        try:
            lower_statement = statement.lower()

            if "sky is blue" in lower_statement or "water is wet" in lower_statement:
                fact_check_result = "TRUE"
                confidence = 0.99
                details = "Statement aligns with widely accepted, fundamental facts."
            elif "sky is green" in lower_statement or "cats can fly" in lower_statement:
                fact_check_result = "FALSE"
                confidence = 0.95
                details = "Statement directly contradicts widely accepted facts or physical laws."
            elif "new scientific discovery" in lower_statement and sources:
                fact_check_result = "NEEDS_MORE_INFO"
                confidence = 0.70
                details = "Statement suggests emerging information; further verification with provided sources is required."
            elif any(keyword in lower_statement for keyword in ["unproven", "speculation", "allegedly"]):
                fact_check_result = "UNVERIFIED"
                confidence = 0.40
                details = "Statement contains qualifiers indicating lack of verification or is highly speculative."
            else:
                fact_check_result = "UNVERIFIED"
                confidence = 0.50
                details = "Could not definitively verify or refute the statement with simulated knowledge."

            result = {
                "statement": statement,
                "fact_check_result": fact_check_result,
                "confidence": confidence,
                "details": details,
                "checked_sources": sources,  # Reflect sources considered for checking
            }
            logger.debug(
                f"[{self.node_name}] Successfully processed statement: '{statement}'. "
                f"Result: {fact_check_result} (Confidence: {confidence:.2f})."
            )
            return result

        except Exception as e:
            logger.exception(
                f"[{self.node_name}] An unexpected error occurred during "
                f"fact-checking simulation for statement '{statement}'."
            )
            raise RuntimeError(
                f"FactCheckerNode failed to process statement: '{statement}' due to an internal error."
            ) from e