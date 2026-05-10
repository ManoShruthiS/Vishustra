import re
import logging
from typing import Any, Dict, List, Pattern
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class PIIRedactorNode(BaseNode):
    """
    A node responsible for identifying and redacting Personally Identifiable Information (PII)
    from textual data. It uses regex-based patterns to mask sensitive information
    like emails, phone numbers, and credit card patterns.
    """

    def __init__(self, custom_patterns: Dict[str, str] = None):
        """
        Initializes the redactor with default patterns or optional custom patterns.
        """
        self._patterns: Dict[str, str] = {
            "EMAIL": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
            "PHONE": r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
            "CREDIT_CARD": r"\b(?:\d[ -]*?){13,16}\b",
            "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
        }
        
        if custom_patterns:
            self._patterns.update(custom_patterns)
            
        self._compiled_regex: List[Pattern] = [
            re.compile(pattern, re.IGNORECASE) for pattern in self._patterns.values()
        ]

    @property
    def node_name(self) -> str:
        """Returns the unique identifier for this node."""
        return "PII_Redactor_Node"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, scanning for PII and replacing it with a mask.
        
        Args:
            data: The input string or collection to be processed.
            context: Execution context containing configuration or metadata.
            
        Returns:
            The redacted data.
        """
        try:
            if not isinstance(data, str):
                logger.warning(f"[{self.node_name}] Received non-string data type: {type(data)}. Attempting conversion.")
                input_text = str(data)
            else:
                input_text = data

            redacted_text = input_text
            mask_token = context.get("redaction_token", "[REDACTED]")

            for regex in self._compiled_regex:
                redacted_text = regex.sub(mask_token, redacted_text)

            logger.info(f"[{self.node_name}] Successfully processed data redaction.")
            return redacted_text

        except Exception as e:
            logger.error(f"[{self.node_name}] Error during PII redaction: {str(e)}", exc_info=True)
            # Depending on safety requirements, we might want to return an empty string 
            # or raise the exception to stop the pipeline.
            raise RuntimeError(f"PII Redaction failed: {e}") from e

    def __repr__(self) -> str:
        return f"<PIIRedactorNode(patterns={list(self._patterns.keys())})>"