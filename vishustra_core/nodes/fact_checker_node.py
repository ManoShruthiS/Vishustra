import logging
from typing import Any, Dict

# Assuming vishustra_core exists at the root of the project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class FactCheckerNode(BaseNode):
    """
    A processing node designed to simulate fact-checking of input statements.
    It verifies statements against a mock knowledge base and provides a verdict
    along with a rationale.
    """

    def __init__(self):
        """
        Initializes the FactCheckerNode with a mock knowledge base.
        In a real-world scenario, this would interface with a dedicated
        fact-checking API, database, or a more sophisticated NLP model.
        """
        self._mock_knowledge_base: Dict[str, Dict[str, str]] = {
            "The Earth is flat.": {"verdict": "FALSE", "rationale": "Scientific consensus and satellite imagery confirm the Earth is an oblate spheroid."},
            "Water boils at 100 degrees Celsius at sea level.": {"verdict": "TRUE", "rationale": "This is a fundamental physical property of water at standard atmospheric pressure."},
            "The capital of France is Paris.": {"verdict": "TRUE", "rationale": "Paris is the official capital and largest city of France."},
            "Humans can breathe underwater indefinitely.": {"verdict": "FALSE", "rationale": "Humans are terrestrial mammals and require atmospheric oxygen to breathe."},
            "All birds can fly.": {"verdict": "FALSE", "rationale": "Many species of birds, such as penguins and ostriches, are flightless."},
            "The sun revolves around the Earth.": {"verdict": "FALSE", "rationale": "The Earth and other planets revolve around the Sun; this is known as heliocentrism."}
        }
        logger.info("FactCheckerNode initialized with mock knowledge base.")

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node.
        """
        return "FactChecker"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to perform a simulated fact-check.

        Expected `data` input:
        A dictionary containing at least a 'statement' key with the string
        to be fact-checked.
        e.g., {"statement": "The Earth is flat."}

        Expected `context` input:
        Optional, could contain parameters like 'confidence_threshold' or
        'api_key' for a real implementation, but not used in this simulation.

        Returns:
        A dictionary containing the original statement, the fact-check verdict,
        and a rationale.
        e.g., {
            "statement": "The Earth is flat.",
            "verdict": "FALSE",
            "rationale": "Scientific consensus and satellite imagery confirm the Earth is an oblate spheroid."
        }

        Raises:
            ValueError: If the input `data` is not a dictionary or lacks a 'statement' key.
            TypeError: If the 'statement' value is not a string.
        """
        logger.debug(f"[{self.node_name}] Starting process for data: {data}")

        if not isinstance(data, dict):
            logger.error(f"[{self.node_name}] Invalid input data type. Expected dict, got {type(data)}.")
            raise ValueError(f"Input data for FactCheckerNode must be a dictionary, but received {type(data)}.")

        statement = data.get("statement")
        if statement is None:
            logger.error(f"[{self.node_name}] Missing 'statement' key in input data: {data}")
            raise ValueError("Input dictionary must contain a 'statement' key for fact-checking.")

        if not isinstance(statement, str):
            logger.error(f"[{self.node_name}] Invalid 'statement' type. Expected str, got {type(statement)}.")
            raise TypeError(f"Value for 'statement' must be a string, but received {type(statement)}.")

        if not statement.strip():
            logger.warning(f"[{self.node_name}] Received an empty or whitespace-only statement.")
            return {
                "statement": statement,
                "verdict": "UNVERIFIABLE",
                "rationale": "The statement provided was empty or contained only whitespace."
            }

        result = self._mock_knowledge_base.get(statement, {
            "verdict": "UNVERIFIABLE",
            "rationale": "Statement not found in mock knowledge base. Cannot provide a definitive fact-check at this time."
        })

        output_data = {
            "statement": statement,
            "verdict": result["verdict"],
            "rationale": result["rationale"]
        }

        logger.info(f"[{self.node_name}] Fact-checked statement: '{statement}' -> Verdict: {result['verdict']}")
        logger.debug(f"[{self.node_name}] Process completed. Output: {output_data}")
        return output_data

if __name__ == '__main__':
    # Basic usage example and testing
    logging.basicConfig(level=logging.INFO) # Set to DEBUG for more verbose output

    fact_checker = FactCheckerNode()

    # Test cases
    test_statements = [
        {"statement": "The Earth is flat."},
        {"statement": "Water boils at 100 degrees Celsius at sea level."},
        {"statement": "The capital of Germany is Berlin."}, # Not in mock DB
        {"statement": "Humans can breathe underwater indefinitely."},
        {"statement": ""}, # Empty statement
        {"text": "This is not a statement key."}, # Missing key
        "This is not a dict.", # Wrong type
        {"statement": 123}, # Wrong type for statement value
        {"statement": "   "} # Whitespace-only statement
    ]

    for i, test_data in enumerate(test_statements):
        print(f"\n--- Test Case {i+1} ---")
        try:
            processed_data = fact_checker.process(test_data, {})
            print(f"Input: {test_data}")
            print(f"Output: {processed_data}")
        except (ValueError, TypeError) as e:
            print(f"Input: {test_data}")
            print(f"Error: {e}")
        except Exception as e:
            print(f"Input: {test_data}")
            print(f"An unexpected error occurred: {e}")