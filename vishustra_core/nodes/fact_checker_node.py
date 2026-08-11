import logging
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class FactCheckerNode(BaseNode):
    """
    A Vishustra processing node that simulates fact-checking of textual data.

    This node takes a piece of text (or a dictionary containing text) and
    attempts to determine its veracity based on simulated rules, marking it
    as verified, unverified, or requiring further review.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "FactCheckerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to simulate fact-checking.

        The input `data` can be:
        1. A string: The text directly to be fact-checked.
        2. A dictionary: Expected to contain a 'text' key with the content
           to be fact-checked. Other keys are ignored but passed through.

        The `context` dictionary can optionally contain specific fact-checking
        rules or known assertions, though for this simulation, it's not strictly
        required.

        Returns a dictionary containing:
        - 'original_input': The input data received.
        - 'text_checked': The extracted text that was checked.
        - 'is_verified': A boolean (True/False) indicating the verification
                         status, or None if indeterminate.
        - 'verification_status': A string describing the status (e.g.,
                                 "VERIFIED", "UNVERIFIED", "UNCHECKABLE",
                                 "REQUIRES_REVIEW").
        - 'issues_found': A list of strings detailing any reasons for
                          unverification or flags.
        - 'sources_consulted': A list of mock sources for this simulation.
        """
        original_input = data
        text_to_check: str = ""
        result: Dict[str, Any] = {
            "original_input": original_input,
            "text_checked": "",
            "is_verified": None,
            "verification_status": "UNCHECKABLE",
            "issues_found": [],
            "sources_consulted": ["Vishustra Internal Knowledge Base v1.0"]
        }

        # --- Input Data Extraction and Validation ---
        if isinstance(data, str):
            text_to_check = data
            result["text_checked"] = text_to_check
        elif isinstance(data, dict):
            if "text" in data and isinstance(data["text"], str):
                text_to_check = data["text"]
                result["text_checked"] = text_to_check
            else:
                logger.warning(
                    f"[{self.node_name}] Input dictionary missing 'text' key "
                    f"or 'text' key is not a string. Data: {data}"
                )
                result["issues_found"].append("Input dictionary missing 'text' key or 'text' value is not a string.")
                return result
        else:
            logger.warning(
                f"[{self.node_name}] Invalid input type received. Expected "
                f"str or dict, got {type(data).__name__}. Data: {data}"
            )
            result["issues_found"].append(f"Invalid input type: Expected str or dict, got {type(data).__name__}.")
            return result

        if not text_to_check.strip():
            logger.info(f"[{self.node_name}] Received empty or whitespace-only text for checking.")
            result["verification_status"] = "REQUIRES_REVIEW"
            result["issues_found"].append("Text to check was empty or whitespace-only.")
            return result

        # --- Simulated Fact-Checking Logic ---
        text_lower = text_to_check.lower()
        issues: List[str] = []
        is_verified_status: Union[bool, None] = None

        # Define some simulated flags for demonstration purposes
        red_flags = [
            ("false claim", "Contains a phrase indicative of a false claim."),
            ("unverified rumor", "Identified as an unverified rumor."),
            ("misinformation", "Suggests presence of misinformation."),
            ("not peer-reviewed", "Explicitly states lack of peer review."),
            ("conspiracy theory", "Phrase common in conspiracy theories."),
        ]
        green_flags = [
            ("scientifically proven", "Claim explicitly states scientific proof."),
            ("established fact", "Identified as an established fact."),
            ("official statement", "References an official statement."),
            ("peer-reviewed study", "References a peer-reviewed study."),
        ]

        # Check for red flags
        for flag_phrase, description in red_flags:
            if flag_phrase in text_lower:
                issues.append(f"Detected red flag: '{flag_phrase}' - {description}")
                # For this simulation, any red flag immediately marks as unverified
                is_verified_status = False

        # Check for green flags if not already marked as unverified
        if is_verified_status is not False: # Only proceed if not already definitively False
            for flag_phrase, description in green_flags:
                if flag_phrase in text_lower:
                    issues.append(f"Detected green flag: '{flag_phrase}' - {description}")
                    # A green flag indicates verification
                    is_verified_status = True

        # Determine final status if not set by flags
        if is_verified_status is None:
            # If no explicit flags, default to requiring review, as we can't definitively verify or unverify
            result["verification_status"] = "REQUIRES_REVIEW"
            issues.append("No explicit verification or unverification flags detected. Requires manual review.")
            logger.info(f"[{self.node_name}] Text requires review: '{text_to_check[:50]}...'")
        elif is_verified_status is True:
            result["verification_status"] = "VERIFIED"
            logger.info(f"[{self.node_name}] Text VERIFIED: '{text_to_check[:50]}...'")
        else: # is_verified_status is False
            result["verification_status"] = "UNVERIFIED"
            logger.warning(f"[{self.node_name}] Text UNVERIFIED: '{text_to_check[:50]}...'")

        result["is_verified"] = is_verified_status
        result["issues_found"].extend(issues)

        return result