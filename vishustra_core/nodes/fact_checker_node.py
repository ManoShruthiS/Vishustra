import logging
from typing import Any, Dict

# Ensure BaseNode is imported from the specified project path
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class FactCheckerNode(BaseNode):
    """
    A processing node designed to simulate fact-checking of textual statements.

    This node accepts a statement (typically embedded within a dictionary) and
    attempts to ascertain its truthfulness by consulting a simulated internal
    knowledge base. In a production environment, this would integrate with
    external fact-checking services, proprietary databases, or advanced
    NLP-driven verification mechanisms.
    """

    # A simple, static knowledge base for simulation purposes.
    # This data structure would be dynamically loaded, configured, or replaced
    # by external service calls in a real-world implementation.
    _mock_knowledge_base = {
        "The sun is a star": {"truth_value": True, "confidence": 1.0, "evidence": ["Astrophysical observation", "Scientific consensus"]},
        "The Earth is flat": {"truth_value": False, "confidence": 1.0, "evidence": ["Satellite imaging", "Global navigation systems"]},
        "Water boils at 100 degrees Celsius at sea level": {"truth_value": True, "confidence": 0.98, "evidence": ["Basic thermodynamics", "Laboratory experiments"]},
        "Birds can breathe underwater indefinitely": {"truth_value": False, "confidence": 1.0, "evidence": ["Avian biology", "Respiratory system limitations"]},
        "Humanity has landed on Mars": {"truth_value": False, "confidence": 0.85, "evidence": ["NASA mission records (crewed missions still in planning)"]},
        "Python is a programming language": {"truth_value": True, "confidence": 1.0, "evidence": ["Computer science history", "Industry adoption"]},
    }

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "FactCheckerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to perform a simulated fact-check on a statement.

        The expected `data` input is a dictionary that must contain a 'statement'
        key whose value is the string text to be fact-checked.

        The `context` dictionary can optionally provide a custom 'knowledge_base'
        to override the default mock one, facilitating dynamic testing or specific
        workflow configurations.

        Args:
            data: The input payload, expected to be a dictionary with a 'statement' key.
            context: A dictionary containing operational context, which may include
                     a 'knowledge_base' for fact-lookup customization.

        Returns:
            A dictionary encapsulating the original statement, its fact-check result,
            an associated confidence score, and any supporting evidence found.
            Example structure:
            {
                "original_statement": "The sun is a star",
                "fact_check_result": "TRUE",  # or "FALSE", "UNVERIFIABLE"
                "confidence": 1.0,
                "evidence": ["Astrophysical observation", "Scientific consensus"]
            }

        Raises:
            ValueError: If the input `data` does not conform to the expected
                        dictionary structure or lacks a valid 'statement' key.
            RuntimeError: If an unforeseen error impedes the fact-checking process.
        """
        logger.info(f"[{self.node_name}] Initiating fact-check process.")

        if not isinstance(data, dict):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected 'dict', "
                f"received '{type(data).__name__}'."
            )
            raise ValueError(
                f"FactCheckerNode requires input 'data' to be a dictionary, "
                f"got '{type(data).__name__}'."
            )

        statement = data.get("statement")
        if not isinstance(statement, str) or not statement.strip():
            logger.error(
                f"[{self.node_name}] 'statement' key is missing, empty, or "
                f"not a string in the input data."
            )
            raise ValueError(
                "Input 'data' dictionary must contain a non-empty 'statement' "
                "string key for fact-checking."
            )

        # Utilize a custom knowledge base if provided in context, otherwise default.
        current_knowledge_base = context.get("knowledge_base", self._mock_knowledge_base)

        logger.debug(f"[{self.node_name}] Fact-checking statement: '{statement}'")
        result = {
            "original_statement": statement,
            "fact_check_result": "UNVERIFIABLE",
            "confidence": 0.0,
            "evidence": []
        }

        try:
            # Perform a case-insensitive lookup for enhanced matching robustness.
            matched_fact_info = None
            for known_statement, fact_info in current_knowledge_base.items():
                if statement.lower() == known_statement.lower():
                    matched_fact_info = fact_info
                    break

            if matched_fact_info:
                truth_value = matched_fact_info["truth_value"]
                result["fact_check_result"] = "TRUE" if truth_value else "FALSE"
                # Default confidence if not explicitly provided in the mock data
                result["confidence"] = matched_fact_info.get("confidence", 0.9)
                result["evidence"] = matched_fact_info.get(
                    "evidence", ["Reference to simulated internal knowledge base"]
                )
                logger.info(
                    f"[{self.node_name}] Statement '{statement}' fact-checked as "
                    f"{result['fact_check_result']} (Confidence: {result['confidence']:.2f})."
                )
            else:
                logger.warning(
                    f"[{self.node_name}] Statement '{statement}' not found in the "
                    f"current knowledge base. Result marked as UNVERIFIABLE."
                )

        except Exception as e:
            logger.exception(
                f"[{self.node_name}] An unexpected error occurred during "
                f"fact-checking for statement '{statement}'."
            )
            # Re-raise the exception, potentially wrapped, to propagate critical
            # operational failures up to the orchestrator layer.
            raise RuntimeError(
                f"Processing error in FactCheckerNode for statement "
                f"'{statement}': {e}"
            ) from e

        logger.debug(
            f"[{self.node_name}] Completed fact-check for '{statement}'. Result: {result}"
        )
        return result
