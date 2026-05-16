import re
import logging
from typing import Any, Dict, List, Union
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class PIIRedactorNode(BaseNode):
    """
    A node responsible for identifying and masking Personally Identifiable Information (PII)
    within string data or nested dictionary structures.
    """

    def __init__(self, mask_char: str = "[REDACTED]"):
        self.mask_char = mask_char
        # Pre-compile regex patterns for performance
        self._patterns = {
            "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
            "phone": re.compile(r"\b(?:\+?\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}\b"),
            "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
            "ipv4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
            "credit_card": re.compile(r"\b(?:\d[ -]*?){13,16}\b")
        }

    @property
    def node_name(self) -> str:
        return "PIIRedactorNode"

    def _redact_text(self, text: str) -> str:
        """Applies regex masking to a single string."""
        if not isinstance(text, str):
            return text
            
        redacted = text
        for pii_type, pattern in self._patterns.items():
            try:
                redacted = pattern.sub(self.mask_char, redacted)
            except Exception as e:
                logger.error(f"Error redacting {pii_type} in {self.node_name}: {str(e)}")
                
        return redacted

    def _traverse_and_redact(self, data: Any) -> Any:
        """Recursively traverses dictionaries and lists to redact strings."""
        if isinstance(data, str):
            return self._redact_text(data)
        elif isinstance(data, dict):
            return {k: self._traverse_and_redact(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._traverse_and_redact(item) for item in data]
        return data

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes input data to redact sensitive information.
        
        Args:
            data: The input data (str, dict, or list).
            context: Execution context containing metadata or configuration.
            
        Returns:
            The processed data with PII masked.
        """
        logger.info(f"Executing {self.node_name} processing logic.")
        
        try:
            if data is None:
                logger.warning(f"{self.node_name} received None input.")
                return None

            result = self._traverse_and_redact(data)
            return result

        except Exception as e:
            logger.exception(f"Critical failure in {self.node_name}: {str(e)}")
            raise RuntimeError(f"Node {self.node_name} failed to process data.") from e