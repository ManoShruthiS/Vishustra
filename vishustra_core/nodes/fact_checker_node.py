import logging
import random
from typing import Any, Dict, List, Union

# Assuming BaseNode is located at this path within the project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class FactCheckerNode(BaseNode):
    """
    A Vishustra processing node that simulates fact-checking a given statement.

    This node takes a statement (string or part of a dictionary) and attempts
    to determine its factual accuracy, returning a structured result including
    a verdict, confidence score, and simulated sources.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initializes the FactCheckerNode with an optional configuration.

        Args:
            config (Dict[str, Any], optional): Configuration parameters for
                                               the fact-checking process.
                                               Currently not used in simulation.
        """
        self._config = config or {}
        logger.debug(f"FactCheckerNode initialized with config: {self._config}")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "Fact Checker"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulates fact-checking the input data.

        The `data` input is expected to be either:
        1. A string: This string is treated directly as the statement to check.
        2. A dictionary: This dictionary must contain a 'statement' key whose
           value is the string to be fact-checked. Other keys are ignored.

        The `context` dictionary can be used to pass global session information
        or shared resources, though it's not directly utilized in this
        simulation.

        Returns a dictionary containing the fact-checking results:
        - 'original_statement': The statement that was checked.
        - 'is_factual': A boolean indicating the simulated factual verdict.
        - 'confidence_score': A float (0.0 to 1.0) representing the simulated
                              confidence in the verdict.
        - 'checked_claims': A list of dictionaries detailing individual claims
                            and their simulated verdicts/details.
        - 'sources': A list of simulated sources for the fact-check.
        - 'processing_status': 'success', 'failed', or 'error'.
        - 'error_message': Detailed error message if processing failed.
        """
        statement_to_check: str = ""
        result: Dict[str, Any] = {
            "original_statement": None,
            "is_factual": False,
            "confidence_score": 0.0,
            "checked_claims": [],
            "sources": [],
            "processing_status": "error",
            "error_message": "An unexpected error occurred."
        }

        try:
            if isinstance(data, str):
                statement_to_check = data
            elif isinstance(data, dict):
                statement_to_check = data.get("statement", "")
                if not statement_to_check:
                    raise ValueError("Input dictionary must contain a non-empty 'statement' key for fact-checking.")
            else:
                raise TypeError(
                    "Input data must be a string or a dictionary containing a 'statement' key."
                    f" Received type: {type(data).__name__}"
                )

            result["original_statement"] = statement_to_check

            if not statement_to_check.strip():
                raise ValueError("Statement to check cannot be empty or consist only of whitespace.")

            # --- Start Fact-Checking Simulation ---
            logger.info(f"Simulating fact-check for statement: '{statement_to_check[:120]}...'")

            # Simple heuristic: statements containing certain keywords are more likely to be "factual"
            known_truth_keywords = ["sun", "earth", "gravity", "water", "sky", "science", "mathematics"]
            is_likely_factual_by_keyword = any(
                keyword in statement_to_check.lower() for keyword in known_truth_keywords
            )

            # Introduce randomness to simulate imperfect real-world fact-checking
            random_chance = random.random() # Value between 0.0 and 1.0

            if is_likely_factual_by_keyword and random_chance > 0.3: # 70% chance of being true if keyword matches
                result["is_factual"] = True
                result["confidence_score"] = round(0.75 + random.random() * 0.25, 2) # Score between 0.75 and 1.0
                result["sources"] = ["Verified Research Database", "Reputable Encyclopedia"]
                result["checked_claims"].append(
                    {"claim": statement_to_check, "verdict": "TRUE", "details": "Matches widely accepted knowledge based on keyword analysis and high confidence score."}
                )
            else:
                result["is_factual"] = False
                result["confidence_score"] = round(random.random() * 0.6, 2) # Score between 0.0 and 0.6
                result["sources"] = ["Online Forum", "Social Media Post (unverified)"]
                result["checked_claims"].append(
                    {"claim": statement_to_check, "verdict": "FALSE", "details": "Insufficient evidence or conflicting information found during simulated check."}
                )

            result["processing_status"] = "success"
            result["error_message"] = None
            logger.debug(
                f"Fact-checking simulation complete for '{statement_to_check[:60]}...'. "
                f"Verdict: {'Factual' if result['is_factual'] else 'Not Factual'} "
                f"with confidence: {result['confidence_score']}"
            )

        except (ValueError, TypeError) as e:
            logger.error(f"FactCheckerNode received invalid input: {e}", exc_info=True)
            result["error_message"] = str(e)
            result["processing_status"] = "failed_validation"
        except Exception as e:
            logger.error(f"An unexpected error occurred during fact-checking simulation: {e}", exc_info=True)
            result["error_message"] = f"Unexpected processing error: {type(e).__name__} - {str(e)}"
            result["processing_status"] = "failed_runtime"
        finally:
            return result
