import logging
from typing import Any, Dict, Union
import random # Used for simulating indeterminate fact-check outcomes

# Assuming BaseNode is located here as per project context and import requirement
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class FactCheckerNode(BaseNode):
    """
    A processing node designed to simulate fact-checking a given statement.

    This node accepts a statement (as a string or within a dictionary) and
    applies a set of simulated rules and a small "knowledge base" to determine
    its factual accuracy. It returns a structured result including a boolean
    factual indicator, a confidence score, and explanatory details.

    The simulation logic demonstrates how a real fact-checker might operate,
    identifying known facts, known falsehoods, and making inferences based
    on keywords, while also handling indeterminate cases.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the unique name of this node.
        """
        return "FactCheckerNode"

    def process(self, data: Union[str, Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to simulate fact-checking a statement.

        The `data` input can be either:
        1. A plain string representing the statement to be fact-checked.
        2. A dictionary containing a 'statement' key, whose value is the string
           to be fact-checked. This allows for more complex inputs in orchestration.

        The `context` dictionary can be used to pass configuration, such as
        'known_facts' or 'known_falsehoods' lists, which the simulation will use.

        Args:
            data: The statement to be fact-checked. Can be a string or a dict
                  containing a 'statement' key.
            context: A dictionary providing operational context and configuration.
                     May include 'known_facts' and 'known_falsehoods' lists.

        Returns:
            A dictionary containing the fact-checking result:
            - 'original_statement': The statement that was checked.
            - 'is_factual': A boolean indicating the simulated fact-check outcome.
            - 'confidence_score': A float (0.0 to 1.0) representing the certainty.
            - 'details': A string providing simulated reasoning for the outcome.

        Raises:
            ValueError: If the input `data` is malformed or does not contain a valid statement.
            Exception: For any other unexpected errors during the fact-checking process.
        """
        statement_to_check: str = ""

        try:
            if isinstance(data, str):
                statement_to_check = data
            elif isinstance(data, dict) and 'statement' in data and isinstance(data['statement'], str):
                statement_to_check = data['statement']
            else:
                logger.error("FactCheckerNode received invalid data format. Expected string or dict with 'statement' key. Received: %s (Type: %s)", data, type(data))
                raise ValueError("Input data must be a string or a dictionary containing a 'statement' key of type string.")

            if not statement_to_check.strip():
                logger.warning("FactCheckerNode received an empty or whitespace-only statement for fact-checking.")
                return {
                    "original_statement": statement_to_check,
                    "is_factual": False,
                    "confidence_score": 0.0,
                    "details": "Empty statement provided for fact-checking."
                }

            logger.info("Initiating fact-check for statement: '%s'", statement_to_check[:100] + ('...' if len(statement_to_check) > 100 else ''))

            # Simulate fact-checking logic using a simple "knowledge base"
            # These can be configured via the context dictionary
            known_facts = context.get("known_facts", [
                "The Earth orbits the Sun.",
                "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
                "Mount Everest is the highest mountain on Earth."
            ])
            known_falsehoods = context.get("known_falsehoods", [
                "Pigs can fly.",
                "The moon is made of green cheese.",
                "The Earth is flat."
            ])

            normalized_statement = statement_to_check.lower().strip()
            is_factual: bool = False
            confidence_score: float = 0.5  # Default to uncertain
            details: str = "Simulated fact-check: Could not definitively verify or refute."

            # Check against known facts
            for fact in known_facts:
                if fact.lower() in normalized_statement or normalized_statement == fact.lower():
                    is_factual = True
                    confidence_score = 0.95
                    details = f"Simulated fact-check: Matched known fact '{fact}'."
                    break

            # If not yet determined factual, check against known falsehoods
            if not is_factual:
                for falsehood in known_falsehoods:
                    if falsehood.lower() in normalized_statement or normalized_statement == falsehood.lower():
                        is_factual = False
                        confidence_score = 0.95
                        details = f"Simulated fact-check: Matched known falsehood '{falsehood}'."
                        break
            
            # More nuanced simulation based on keywords if not definitively matched above
            if confidence_score == 0.5: # Still uncertain
                if "capital of france is paris" in normalized_statement:
                    is_factual = True
                    confidence_score = 0.9
                    details = "Simulated fact-check: Statement aligns with general geographic knowledge."
                elif "fish can walk" in normalized_statement and "not" not in normalized_statement:
                    is_factual = False
                    confidence_score = 0.8
                    details = "Simulated fact-check: Generally false, with exceptions not covered by simple rule."
                elif "square has four equal sides" in normalized_statement:
                    is_factual = True
                    confidence_score = 0.85
                    details = "Simulated fact-check: Based on geometric definition."
                elif "perpetual motion machine" in normalized_statement:
                    is_factual = False
                    confidence_score = 0.9
                    details = "Simulated fact-check: Contradicts laws of physics."

            # For statements that fall outside explicit rules, introduce some variability
            if confidence_score < 0.7:
                if random.random() > 0.6:  # 40% chance of being deemed "true" with moderate confidence
                    is_factual = True
                    confidence_score = round(random.uniform(0.65, 0.8), 2)
                    details = "Simulated fact-check: Appears plausible based on broad inference."
                else:  # 60% chance of being deemed "false" or "unverified"
                    is_factual = False
                    confidence_score = round(random.uniform(0.5, 0.7), 2)
                    details = "Simulated fact-check: Insufficient evidence or unverified claim."


            result = {
                "original_statement": statement_to_check,
                "is_factual": is_factual,
                "confidence_score": confidence_score,
                "details": details
            }
            logger.info("Fact-check completed for '%s'. Result: is_factual=%s, confidence=%.2f",
                        statement_to_check[:50] + '...' if len(statement_to_check) > 50 else statement_to_check,
                        result['is_factual'], result['confidence_score'])
            return result

        except ValueError as ve:
            # Re-raise validation errors after logging
            logger.error("A validation error occurred in FactCheckerNode: %s", ve)
            raise
        except Exception as e:
            # Catch and log any other unexpected exceptions
            logger.exception("An unexpected error occurred in FactCheckerNode while processing data: %s", data)
            raise # Re-raise to ensure calling orchestration handles it