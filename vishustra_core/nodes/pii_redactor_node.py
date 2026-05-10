import re
import logging
from typing import Any, Dict, List, Union
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class PIIRedactorNode(BaseNode):
    """
    A processing node designed to identify and mask Personally Identifiable Information (PII).
    It scans input data for patterns such as emails, phone numbers, and credit card numbers
    to ensure data privacy before further downstream processing or LLM consumption.
    """

    # Common PII regex patterns
    PII_PATTERNS = {
        "EMAIL": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "PHONE": r"(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
        "CREDIT_CARD": r"\b(?:\d[ -]*?){13,16}\b",
        "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
        "IPV4": r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"
    }

    @property
    def node_name(self) -> str:
        """Returns the identifier for this node."""
        return "PII Redactor Node"

    def _redact_string(self, text: str) -> str:
        """Applies regex substitution to a string based on known PII patterns."""
        redacted_text = text
        for label, pattern in self.PII_PATTERNS.items():
            redacted_text = re.sub(pattern, f"[REDACTED_{label}]", redacted_text)
        return redacted_text

    def _traverse_and_redact(self, data: Any) -> Any:
        """Recursively traverses nested structures to find and redact strings."""
        if isinstance(data, str):
            return self._redact_string(data)
        elif isinstance(data, dict):
            return {k: self._traverse_and_redact(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._traverse_and_redact(item) for item in data]
        return data

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, redacting sensitive information.
        
        Args:
            data: The input data (string, dict, or list) to be processed.
            context: The orchestration context (unused in this node but required by API).
            
        Returns:
            The input data with sensitive patterns replaced by redaction placeholders.
            
        Raises:
            TypeError: If processing encounters an unresolvable type error.
        """
        try:
            if data is None:
                logger.debug(f"[{self.node_name}] Received null input; skipping redaction.")
                return None

            logger.info(f"[{self.node_name}] Executing PII redaction on input data.")
            
            result = self._traverse_and_redact(data)
            
            logger.info(f"[{self.node_name}] Successfully completed data anonymization.")
            return result

        except Exception as e:
            logger.exception(f"[{self.node_name}] An error occurred during the redaction process: {str(e)}")
            raise RuntimeError(f"PII Redactor node failed to process data: {e}") from e

    def __repr__(self) -> str:
        return f"<PIIRedactorNode(name='{self.node_name}')>"