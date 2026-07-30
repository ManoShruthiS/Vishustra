import logging
from typing import Any, Dict, List, Union

# Assuming BaseNode is available at this path as per project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class FactCheckerNode(BaseNode):
    """
    A Vishustra processing node designed to perform simulated fact-checking
    on input data.

    This node identifies claims within the input data and attempts to verify
    them against a predefined set of internal facts or an external fact database
    provided via the context. It then annotates the original data with the
    results of these fact-checks, including a status (accurate, inaccurate, unverified)
    and an explanation.
    """

    # A simple, hardcoded database of facts for simulation purposes.
    # In a real-world scenario, this would interface with an external
    # knowledge base, API, or LLM.
    _KNOWN_FACTS: Dict[str, bool] = {
        "The capital of France is Paris.": True,
        "The Earth is flat.": False,
        "Water boils at 100 degrees Celsius at sea level.": True,
        "Vishustra is an LLM orchestration framework.": True,
        "Python is a compiled language.": False,
        "The sun revolves around the Earth.": False,
        "The highest mountain in the world is Mount Everest.": True,
    }

    def __init__(self, confidence_threshold: float = 0.5):
        """
        Initializes the FactCheckerNode.

        Args:
            confidence_threshold (float): A simulated threshold for deeming a fact
                                          sufficiently checked or confident.
                                          Used to demonstrate configurability,
                                          though its direct impact in this
                                          basic simulation is limited.
        """
        self._confidence_threshold = confidence_threshold
        logger.debug("FactCheckerNode initialized with confidence_threshold: %s", confidence_threshold)

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "FactChecker"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to identify and fact-check claims.

        The input `data` can be:
        - A `str`: Treated as a single claim to be checked.
        - A `dict`: Expected to contain a 'text' key (string) or a 'claims'
          key (list of strings). If both are present, 'claims' takes precedence.

        The `context` dictionary can optionally contain a 'fact_database'
        (a `Dict[str, bool]`) to override or augment the node's internal
        `_KNOWN_FACTS` for checking.

        Args:
            data (Any): The input data to be processed.
            context (Dict[str, Any]): A dictionary providing additional
                                       context or resources for processing.
                                       Can contain 'fact_database'.

        Returns:
            Dict[str, Any]: The processed data, augmented with fact-checking results.
                            If an error occurs or no claims are found, a structured
                            error/skipped result is returned.
        """
        claims_to_check: List[str] = []
        processed_data: Dict[str, Any] = {}

        if isinstance(data, str):
            claims_to_check.append(data)
            processed_data = {"text": data}  # Wrap string for consistent output structure
            logger.debug("FactCheckerNode received a string as data, treating it as a single claim.")
        elif isinstance(data, dict):
            processed_data = dict(data)  # Create a mutable copy
            if 'claims' in processed_data and isinstance(processed_data['claims'], list):
                claims_to_check.extend([c for c in processed_data['claims'] if isinstance(c, str)])
                logger.debug("Identified %d claims from 'claims' key.", len(claims_to_check))
            elif 'text' in processed_data and isinstance(processed_data['text'], str):
                # If 'claims' is not present or valid, fall back to 'text'
                claims_to_check.append(processed_data['text'])
                logger.debug("Identified 'text' as a potential claim source.")
            else:
                logger.warning("FactCheckerNode received dictionary without 'claims' list or 'text' string key. No claims to check.")
                processed_data["fact_check_results"] = {
                    "status": "skipped",
                    "message": "No identifiable claims or text in input dictionary for fact-checking."
                }
                return processed_data
        else:
            logger.error("FactCheckerNode received unexpected data type: %s. Expected str or dict.", type(data))
            return {
                "original_data": data,
                "fact_check_results": {
                    "status": "error",
                    "message": "Input data must be a string or a dictionary.",
                    "details": f"Received type: {type(data).__name__}"
                }
            }

        if not claims_to_check:
            # This can happen if a dict was valid but contained no valid string claims in 'claims' list
            processed_data["fact_check_results"] = {
                "status": "skipped",
                "message": "No valid claims were found to check after parsing input data."
            }
            logger.info("Fact-checking skipped: No claims found in the input data.")
            return processed_data

        # Determine which fact database to use
        fact_database = context.get('fact_database', self._KNOWN_FACTS)
        if not isinstance(fact_database, dict):
            logger.warning("Context 'fact_database' is not a dictionary. Falling back to internal known facts.")
            fact_database = self._KNOWN_FACTS

        results = []
        for claim in claims_to_check:
            check_result = self._check_single_claim(claim, fact_database)
            results.append(check_result)

        processed_data["fact_check_results"] = {
            "status": "completed",
            "checked_claims": results
        }
        logger.info("Fact-checking completed for %d claims.", len(claims_to_check))
        return processed_data

    def _check_single_claim(self, claim: str, fact_database: Dict[str, bool]) -> Dict[str, Any]:
        """
        Simulates checking a single claim against the provided fact database.

        Args:
            claim (str): The claim string to check.
            fact_database (Dict[str, bool]): The database of known facts (claim -> truth_value).

        Returns:
            Dict[str, Any]: A dictionary containing the claim, its fact-checking status,
                            confidence, and an explanation.
        """
        normalized_claim = claim.strip()
        status: str
        confidence: float
        evidence: str

        is_accurate = fact_database.get(normalized_claim, None)

        if is_accurate is not None:
            status = "accurate" if is_accurate else "inaccurate"
            confidence = 1.0  # High confidence for direct match
            evidence = f"Directly matched against known fact in database: '{normalized_claim}'."
            logger.debug("Claim '%s' directly matched: %s", normalized_claim, status)
        else:
            # Simulate a heuristic check for claims not in the database
            # In a production system, this could involve external APIs, regex, or LLM calls.
            if "Vishustra" in normalized_claim and "LLM orchestration framework" in normalized_claim:
                status = "likely_accurate"
                confidence = 0.8
                evidence = "Heuristically marked as likely accurate due to keyword match for project context."
                logger.debug("Claim '%s' heuristically marked as likely accurate.", normalized_claim)
            else:
                status = "unverified"
                confidence = 0.0
                evidence = "Claim could not be directly verified or heuristically evaluated against available facts."
                logger.debug("Claim '%s' marked as unverified.", normalized_claim)

        return {
            "claim": claim,
            "status": status,
            "confidence": confidence,
            "checked_by": self.node_name,
            "evidence": evidence
        }