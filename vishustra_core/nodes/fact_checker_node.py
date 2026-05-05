import logging
from typing import Any, Dict, List

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class FactCheckerNode(BaseNode):
    """
    A processing node designed to simulate fact-checking for textual claims.

    This node takes a claim (string) as input and attempts to determine its
    veracity based on predefined internal heuristics or a placeholder for
    external verification mechanisms. It outputs a structured dictionary
    containing the original claim, its verification status, and a summary
    of the evidence or reasoning.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this processing node.
        """
        return "Fact Checker"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes an input claim to determine its factual accuracy.

        This method simulates fact-checking by:
        1. Validating the input `data` as a string (the claim).
        2. Applying internal heuristics (keyword matching) to assign a
           preliminary verification status.
        3. Constructing a detailed output dictionary with the verification result.
        In a real-world scenario, this would involve querying external
        knowledge bases, truth-checking APIs, or leveraging sophisticated
        natural language inference models.

        Args:
            data (Any): The input data, expected to be a string representing the claim to be fact-checked.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       or configuration for the node.
                                       Optional keys:
                                       - 'confidence_threshold' (float): Minimum confidence for a claim to be considered verified. (default: 0.7)
                                       - 'positive_keywords' (List[str]): Keywords indicating truth.
                                       - 'negative_keywords' (List[str]): Keywords indicating falsehood.

        Returns:
            Dict[str, Any]: A dictionary containing the fact-checking results:
                            - 'original_claim' (str): The claim that was processed.
                            - 'is_verified' (bool): True if the claim is considered true, False if false. None if unverified.
                            - 'verification_status' (str): "True", "False", or "Unverified".
                            - 'evidence_summary' (str): A summary of the evidence or reasoning for the status.
                            - 'confidence_score' (float): A simulated confidence score (0.0 to 1.0).
                            - 'error' (str, optional): An error message if processing failed.
        """
        logger.debug(f"[{self.node_name}] Starting process for data type: {type(data)}")

        if not isinstance(data, str):
            error_msg = f"[{self.node_name}] Invalid input data type. Expected 'str', got '{type(data).__name__}'."
            logger.error(error_msg)
            return {
                "original_claim": data,
                "is_verified": None,
                "verification_status": "Error",
                "evidence_summary": error_msg,
                "confidence_score": 0.0,
                "error": error_msg
            }

        claim = data.strip()
        result: Dict[str, Any] = {
            "original_claim": claim,
            "is_verified": None,
            "verification_status": "Unverified",
            "evidence_summary": "Initial assessment required.",
            "confidence_score": 0.5
        }

        # Retrieve configuration from context or use defaults
        confidence_threshold: float = context.get('confidence_threshold', 0.7)
        positive_keywords: List[str] = context.get('positive_keywords', ['is a fact', 'is true', 'universally accepted', 'verified claim'])
        negative_keywords: List[str] = context.get('negative_keywords', ['is not true', 'is false', 'debunked', 'myth', 'fake news'])

        try:
            # Simulate a basic keyword-based fact check
            lower_claim = claim.lower()
            is_true_indicators = sum(1 for kw in positive_keywords if kw in lower_claim)
            is_false_indicators = sum(1 for kw in negative_keywords if kw in lower_claim)

            if is_true_indicators > is_false_indicators and is_true_indicators > 0:
                result["is_verified"] = True
                result["verification_status"] = "True"
                result["evidence_summary"] = f"Claim contains positive indicators. (e.g., {' '.join(positive_keywords[:2])}...)"
                result["confidence_score"] = min(0.95, 0.5 + (is_true_indicators * 0.1))
            elif is_false_indicators > is_true_indicators and is_false_indicators > 0:
                result["is_verified"] = False
                result["verification_status"] = "False"
                result["evidence_summary"] = f"Claim contains negative indicators. (e.g., {' '.join(negative_keywords[:2])}...)"
                result["confidence_score"] = min(0.95, 0.5 + (is_false_indicators * 0.1))
            else:
                result["is_verified"] = None
                result["verification_status"] = "Unverified"
                result["evidence_summary"] = "No strong internal indicators found. Requires external verification."
                result["confidence_score"] = 0.5

            # Apply confidence threshold if applicable
            if result["is_verified"] is not None and result["confidence_score"] < confidence_threshold:
                logger.debug(f"[{self.node_name}] Claim '{claim[:50]}...' status downgraded due to low confidence ({result['confidence_score']:.2f} < {confidence_threshold:.2f}).")
                result["is_verified"] = None
                result["verification_status"] = "Unverified (Low Confidence)"
                result["evidence_summary"] += f" Confidence score below threshold ({confidence_threshold})."

            logger.info(f"[{self.node_name}] Processed claim: '{claim[:75]}...' - Status: {result['verification_status']}, Confidence: {result['confidence_score']:.2f}")

        except Exception as e:
            error_msg = f"[{self.node_name}] An unexpected error occurred during fact-checking: {e}"
            logger.exception(error_msg)
            result.update({
                "is_verified": None,
                "verification_status": "Error",
                "evidence_summary": f"Processing failed: {e}",
                "confidence_score": 0.0,
                "error": error_msg
            })

        logger.debug(f"[{self.node_name}] Finished processing. Result: {result}")
        return result