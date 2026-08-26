import logging
from typing import Any, Dict, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class FactCheckerNode(BaseNode):
    """
    A processing node designed to simulate fact-checking of a given statement.

    This node takes a statement (either directly as a string or embedded within a dictionary)
    and produces a simulated fact-checking verdict along with a brief explanation and
    a confidence score. It serves as a placeholder to be extended with actual external
    fact-checking APIs or services, which would typically be configured and passed
    via the `context` in a production Vishustra environment.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "FactChecker"

    def process(self, data: Union[str, Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to simulate a fact-checking operation.

        The `data` input is expected to represent a statement to be verified.
        It can be provided in two forms:
        - As a direct string: The text of the statement itself.
        - As a dictionary: Expected to contain a 'statement' key whose value is
          the string text to be fact-checked. Other keys in the dictionary
          will be preserved in the output for traceability.

        The `context` dictionary is currently not utilized in this simulated
        implementation but is available for passing external services (e.g.,
        API clients, database connections) in a real-world scenario.

        Args:
            data: The input statement to check, either as a string or a dictionary
                  containing a 'statement' key.
            context: A dictionary providing contextual information or external
                     service clients. (Not directly used in this simulation).

        Returns:
            A dictionary containing the original statement, the simulated
            fact-checking results (verdict, explanation, confidence), and
            the original input data for full traceability.

        Raises:
            ValueError: If the input data is not a string or a dictionary,
                        or if a dictionary is provided without a valid
                        'statement' string, or if the statement is empty.
            Exception: For any unexpected errors encountered during processing.
        """
        logger.debug(f"[{self.node_name}] Starting process for data: {data!r}")

        statement: str
        original_input_data = data  # Preserve the original input for the result structure

        try:
            if isinstance(data, str):
                statement = data
                logger.debug(f"[{self.node_name}] Input data identified as a string statement.")
            elif isinstance(data, dict):
                statement = data.get("statement")
                if not isinstance(statement, str) or not statement.strip():
                    raise ValueError("Dictionary input must contain a non-empty 'statement' string key.")
                logger.debug(f"[{self.node_name}] Input data identified as a dictionary, extracted statement.")
            else:
                raise ValueError("Input data must be a string or a dictionary with a 'statement' key.")

            statement = statement.strip()
            if not statement:
                raise ValueError("Statement to fact-check cannot be empty or whitespace-only.")

            # --- Simulated Fact-Checking Logic ---
            # This section uses simple keyword matching to simulate fact-checking.
            # In a production environment, this would involve calling external
            # fact-checking APIs, consulting knowledge bases, or using ML models.
            verdict: str
            explanation: str
            confidence: float = 0.5  # Default confidence for unverified statements

            statement_lower = statement.lower()

            if "water is h2o" in statement_lower:
                verdict = "TRUE"
                explanation = "Confirmed: Water is chemically represented as H2O. This is a universally accepted scientific fact."
                confidence = 0.99
            elif "all cats are dogs" in statement_lower:
                verdict = "FALSE"
                explanation = "Refuted: Cats and dogs are distinct species within the animal kingdom. This claim is biologically incorrect."
                confidence = 0.95
            elif "sun rises in the west" in statement_lower:
                verdict = "FALSE"
                explanation = "Refuted: The sun rises in the east and sets in the west due to Earth's rotation. This claim is geographically incorrect."
                confidence = 0.98
            elif "gravitational waves exist" in statement_lower:
                verdict = "TRUE"
                explanation = "Confirmed: Gravitational waves were predicted by Einstein and directly detected in 2015. This is a scientific fact."
                confidence = 0.97
            else:
                verdict = "UNVERIFIED"
                explanation = "The statement could not be definitively verified or refuted with the current simulated knowledge base. Further research may be required."
                confidence = 0.45  # Slightly lower confidence for general unverified statements

            result = {
                "original_statement": statement,
                "fact_check_result": {
                    "verdict": verdict,
                    "explanation": explanation,
                    "confidence": confidence,
                    "checked_by_node": self.node_name,
                    # In a real system, 'sources', 'timestamp', 'links' would be added here.
                },
                "original_input_data": original_input_data,  # Include for full context
            }
            logger.info(f"[{self.node_name}] Fact-checked statement: '{statement[:75]}{'...' if len(statement) > 75 else ''}' -> Verdict: {verdict}")
            logger.debug(f"[{self.node_name}] Process completed with result: {result}")
            return result

        except ValueError as ve:
            logger.error(f"[{self.node_name}] Data validation error during fact-checking: {ve}", exc_info=True)
            raise  # Re-raise the ValueError to propagate the issue
        except Exception as e:
            logger.critical(f"[{self.node_name}] An unexpected critical error occurred during fact-checking: {e}", exc_info=True)
            # Depending on the framework's error handling policy, a custom NodeExecutionError
            # might be raised here, but re-raising a general Exception is also common.
            raise