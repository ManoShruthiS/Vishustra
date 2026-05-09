import re
import logging
from typing import Any, Dict, List, Pattern
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class PIIRedactorNode(BaseNode):
    """
    A specialized node for the Vishustra framework designed to identify and redact 
    Personally Identifiable Information (PII) from strings or structured dictionaries.
    
    This node utilizes optimized regex patterns to detect common sensitive patterns
    such as emails, phone numbers, and IPv4 addresses, ensuring data privacy 
    before passing payloads to LLMs or external logging sinks.
    """

    # Pre-defined regex patterns for common PII categories
    PII_PATTERNS: Dict[str, str] = {
        "email": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
        "phone": r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "ipv4": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        "credit_card": r"\b(?:\d[ -]*?){13,16}\b"
    }

    def __init__(self, default_mask: str = "[REDACTED]"):
        """
        Initializes the redactor with compiled patterns for performance.
        
        :param default_mask: The string used to replace identified PII.
        """
        self._default_mask = default_mask
        self._compiled_patterns: List[Pattern] = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.PII_PATTERNS.values()
        ]

    @property
    def node_name(self) -> str:
        """Returns the unique identifier for this node type."""
        return "PIIRedactorNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes input data to mask sensitive information.
        
        :param data: The input data, expected to be a string, dict, or list.
        :param context: Execution context, can provide 'custom_mask' to override default.
        :return: Data of the same structure with PII redacted.
        """
        mask = context.get("redaction_mask", self._default_mask)
        
        try:
            return self._traverse_and_redact(data, mask)
        except Exception as e:
            logger.error(
                f"Critical failure in {self.node_name} during data transformation: {str(e)}",
                exc_info=True
            )
            # In a pipeline, we fail-safe by returning the error or raising to halt execution
            raise RuntimeError(f"Node {self.node_name} failed to process payload safely.") from e

    def _redact_text(self, text: str, mask: str) -> str:
        """Applies all compiled regex patterns to a single string."""
        if not isinstance(text, str):
            return text
            
        redacted = text
        for pattern in self._compiled_patterns:
            redacted = pattern.sub(mask, redacted)
        return redacted

    def _traverse_and_redact(self, data: Any, mask: str) -> Any:
        """
        Recursively traverses nested structures to find and redact strings.
        """
        if isinstance(data, str):
            return self._redact_text(data, mask)
        
        if isinstance(data, dict):
            return {key: self._traverse_and_redact(value, mask) for key, value in data.items()}
        
        if isinstance(data, list):
            return [self._traverse_and_redact(item, mask) for item in data]
        
        return data