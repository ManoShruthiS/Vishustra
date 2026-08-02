import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class FactCheckerNode(BaseNode):
    """
    A processing node that simulates fact-checking of a given statement.
    It checks the input statement against a predefined set of mock facts
    or against facts provided via the context.
    """

    def __init__(self):
        # A simple, mock internal database of facts for simulation.
        # In a real scenario, this would interact with an external fact-checking API
        # or a sophisticated knowledge graph.
        self._default_known_facts: Dict[str, bool] = {
            "The sky is blue": True,
            "Water boils at 100 degrees Celsius at sea level": True,
            "The Earth is flat": False,
            "Birds can fly": True,
            "The moon is made of cheese": False,
            "Humans need oxygen to breathe": True,
        }
        logger.debug("FactCheckerNode initialized with default known facts.")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "FactChecker"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, attempting to fact-check a statement.

        Expected `data` format:
        {
            "statement": "The statement to be fact-checked."
        }

        Expected `context` keys (optional):
        - "known_facts": A dictionary mapping statements to their boolean truth value,
                         to override or augment the node's default facts.

        Returns a dictionary with the original statement, its verification status,
        and details.

        Return format:
        {
            "statement": "Original statement",
            "status": "verified" | "disputed" | "uncheckable" | "error",
            "details": "Reason for the status or error message."
        }
        """
        if not isinstance(data, dict):
            logger.error("FactCheckerNode received invalid data type. Expected dict, got %s.", type(data))
            return {
                "statement": None,
                "status": "error",
                "details": f"Invalid input data type. Expected dict, got {type(data).__name__}."
            }

        statement = data.get("statement")
        if not isinstance(statement, str) or not statement:
            logger.warning("FactCheckerNode received data without a valid 'statement' key.")
            return {
                "statement": statement,
                "status": "error",
                "details": "Missing or invalid 'statement' key in input data. Expected a non-empty string."
            }

        # Combine default facts with any provided in context
        known_facts = self._default_known_facts.copy()
        if "known_facts" in context and isinstance(context["known_facts"], dict):
            known_facts.update(context["known_facts"])
            logger.debug("FactCheckerNode using augmented known facts from context.")
        else:
            logger.debug("FactCheckerNode using only default known facts.")

        normalized_statement = statement.strip().lower()
        result_status = "uncheckable"
        result_details = "The statement could not be verified or disputed against known facts."

        # Simulate checking against known facts
        for fact, is_true in known_facts.items():
            normalized_fact = fact.strip().lower()
            if normalized_statement == normalized_fact:
                result_status = "verified" if is_true else "disputed"
                result_details = f"Matched against known fact: '{fact}' which is {'true' if is_true else 'false'}."
                logger.info("Statement '%s' matched known fact. Status: %s", statement, result_status)
                break
            # A more advanced check could look for keywords or semantic similarity
            # For this simulation, we'll stick to exact matches for simplicity
            # For instance, if "sky is blue" is known, a statement like "The color of the sky is blue"
            # could also be verified with more sophisticated logic.
            elif normalized_fact in normalized_statement:
                # Basic partial match, less reliable but demonstrates a concept
                if is_true:
                    result_status = "verified"
                    result_details = f"Partial match with known true fact: '{fact}'. Consider refining statement."
                else:
                    result_status = "disputed"
                    result_details = f"Partial match with known false fact: '{fact}'. Consider refining statement."
                logger.warning("Statement '%s' partially matched known fact. Status: %s", statement, result_status)
                # In a real system, partial matches would require confidence scores or further verification.
                # For this simulation, we'll let exact matches take precedence and return the first relevant finding.
                if result_status != "uncheckable":
                    break


        final_result = {
            "statement": statement,
            "status": result_status,
            "details": result_details
        }
        logger.debug("FactCheckerNode processed statement '%s'. Result: %s", statement, final_result)
        return final_result

# Example Usage (for local testing, not part of the required file content)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fact_checker = FactCheckerNode()

    test_context_with_extra_fact = {
        "known_facts": {
            "The capital of France is Paris": True,
            "Elephants can fly": False
        }
    }

    test_cases = [
        {"statement": "The sky is blue"},
        {"statement": "The Earth is flat"},
        {"statement": "Water boils at 100 degrees Celsius at sea level"},
        {"statement": "The moon is made of cheese"},
        {"statement": "The capital of France is Paris"}, # Should use context fact
        {"statement": "Sun revolves around Earth"},
        {"statement": "Birds can fly"},
        {"statement": "Elephants can fly"}, # Should use context fact
        {"statement": "This is an unknown fact"},
        {"statement": "Humans need oxygen"}, # Partial match test
        {"statement": 123}, # Invalid data type for statement
        {"not_a_statement": "Some text"}, # Missing statement key
        "not a dict", # Invalid data type for data
    ]

    print("\n--- Running FactCheckerNode Tests ---")
    for i, test_data in enumerate(test_cases):
        print(f"\nTest Case {i+1}: Input: {test_data}")
        if isinstance(test_data, dict) and test_data.get("statement") in ["The capital of France is Paris", "Elephants can fly"]:
            result = fact_checker.process(test_data, test_context_with_extra_fact)
        else:
            result = fact_checker.process(test_data, {})
        print(f"Result: {result}")
        print("-" * 30)

    print("\n--- FactCheckerNode Tests Complete ---")