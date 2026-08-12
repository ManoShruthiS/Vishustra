import logging
from typing import Any, Dict, List, Union

# Assuming this path is correct based on the prompt's context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class FactCheckerNode(BaseNode):
    """
    A processing node designed to simulate fact-checking of textual assertions.

    This node takes a statement (string or dictionary with a 'statement' key)
    and, for simulation purposes, returns a structured result indicating
    a simulated verification status. In a real-world scenario, this node
    would integrate with external fact-checking APIs or knowledge bases.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "FactChecker"

    def process(self, data: Union[str, Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to simulate fact-checking.

        The input `data` is expected to be either:
        1. A string: The statement to be fact-checked.
        2. A dictionary: Must contain a 'statement' key with the text to check.
           It can also contain an optional 'claims' key (List[str]) for specific
           sub-claims to check.

        The `context` dictionary can optionally contain:
        - 'mock_fact_results': A dictionary where keys are claims/statements
          and values are their simulated truth status (True/False) or a detailed
          result dictionary. This is primarily for testing and demonstration.

        Returns a dictionary containing the original input, the statement(s) checked,
        and simulated fact-checking results.

        Example output:
        {
            "original_input": "The sky is blue and cats can fly.",
            "statement_to_check": "The sky is blue and cats can fly.",
            "fact_check_results": [
                {
                    "claim": "The sky is blue",
                    "status": True,
                    "evidence": "Common knowledge: Sky appears blue due to Rayleigh scattering.",
                    "confidence": 0.99
                },
                {
                    "claim": "cats can fly",
                    "status": False,
                    "evidence": "Biological fact: Cats are mammals and do not possess wings or natural flight capabilities.",
                    "confidence": 0.98
                }
            ],
            "overall_status": "Partially Verified"
        }
        """
        statement_to_check: str = ""
        claims_to_check: List[str] = []
        original_input_data = data  # Store original for output

        if isinstance(data, str):
            statement_to_check = data
            claims_to_check.append(data)
        elif isinstance(data, dict):
            if 'statement' not in data:
                logger.error("FactCheckerNode received dictionary data without a 'statement' key.")
                raise ValueError("Input dictionary must contain a 'statement' key.")
            statement_to_check = str(data['statement']) # Ensure it's a string

            # Add the main statement as a claim to be checked
            claims_to_check.append(statement_to_check)

            # If specific sub-claims are provided, add them to the list
            if 'claims' in data and isinstance(data['claims'], list):
                valid_sub_claims = [c for c in data['claims'] if isinstance(c, str) and c.strip()]
                claims_to_check.extend(valid_sub_claims)
                if not valid_sub_claims and len(claims_to_check) == 1: # Only the main statement, sub_claims were invalid
                    logger.warning("FactCheckerNode received a 'claims' list, but it was empty or contained non-string elements. Defaulting to checking the main statement only.")
            
            # Remove duplicates if main statement was also in claims list
            claims_to_check = list(dict.fromkeys(claims_to_check)) 
        else:
            logger.error(f"FactCheckerNode received unsupported data type: {type(data)}. Expected str or dict.")
            raise TypeError(f"FactCheckerNode expects input data to be a string or a dictionary, got {type(data)}.")

        if not statement_to_check.strip():
            logger.warning("FactCheckerNode received an empty or whitespace-only statement to check.")
            return {
                "original_input": original_input_data,
                "statement_to_check": statement_to_check,
                "fact_check_results": [],
                "overall_status": "Uncheckable: Empty statement"
            }

        logger.info(f"FactCheckerNode processing statement: '{statement_to_check[:100]}...' with {len(claims_to_check)} claims.")

        mock_fact_results: Dict[str, Any] = context.get('mock_fact_results', {})
        results: List[Dict[str, Any]] = []
        overall_status_flags: List[bool] = [] # Collect boolean statuses to determine overall_status

        for claim in claims_to_check:
            claim_result: Dict[str, Any] = {
                "claim": claim,
                "status": "Unverified",
                "evidence": "No external service configured or specific mock data available.",
                "confidence": 0.0
            }

            # Simulate fact-checking using mock data from context
            if claim in mock_fact_results:
                mock_res = mock_fact_results[claim]
                if isinstance(mock_res, dict):
                    claim_result.update(mock_res)
                else: # Assume boolean status
                    claim_result["status"] = mock_res
                    claim_result["evidence"] = f"Simulated result from context.mock_fact_results for '{claim}'."
                    claim_result["confidence"] = 1.0 if mock_res else 0.8
            # Simple keyword-based simulated checks
            elif "the sky is blue" in claim.lower():
                claim_result["status"] = True
                claim_result["evidence"] = "Common knowledge: Sky appears blue due to Rayleigh scattering."
                claim_result["confidence"] = 0.99
            elif "cats can fly" in claim.lower():
                claim_result["status"] = False
                claim_result["evidence"] = "Biological fact: Cats are mammals and do not possess wings or natural flight capabilities."
                claim_result["confidence"] = 0.98
            elif "llms are sentient" in claim.lower():
                claim_result["status"] = False
                claim_result["evidence"] = "Current scientific consensus: LLMs are complex statistical models, not sentient beings."
                claim_result["confidence"] = 0.90
            elif "python is a programming language" in claim.lower():
                claim_result["status"] = True
                claim_result["evidence"] = "Common knowledge: Python is a widely used high-level, general-purpose programming language."
                claim_result["confidence"] = 0.99
            elif "vishustra is an llm orchestration framework" in claim.lower():
                claim_result["status"] = True
                claim_result["evidence"] = "Project context: Vishustra is designed as a highly modular LLM orchestration framework."
                claim_result["confidence"] = 0.99
            
            results.append(claim_result)
            if isinstance(claim_result["status"], bool):
                overall_status_flags.append(claim_result["status"])

        # Determine overall status based on individual claims
        overall_status: Union[bool, str]
        if not claims_to_check:
            overall_status = "Uncheckable: No claims processed"
        elif not overall_status_flags: # No claims returned a boolean status (all unverified)
            overall_status = "Unverified"
        elif all(overall_status_flags):
            overall_status = True
        elif not any(overall_status_flags):
            overall_status = False
        else:
            overall_status = "Partially Verified"

        logger.info(f"FactCheckerNode completed processing for '{statement_to_check[:50]}...'. Overall status: {overall_status}")
        return {
            "original_input": original_input_data,
            "statement_to_check": statement_to_check,
            "fact_check_results": results,
            "overall_status": overall_status
        }