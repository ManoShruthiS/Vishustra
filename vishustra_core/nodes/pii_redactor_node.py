import re
import logging
from typing import Any, Dict, List, Pattern
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class PIIRedactorNode(BaseNode):
    """
    A node designed to identify and mask Personally Identifiable Information (PII)
    within text data using configurable regex patterns.
    """

    DEFAULT_PATTERNS = {
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "phone": r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b(?:\d[ -]*?){13,16}\b",
        "ipv4": r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    }

    def __init__(self, custom_patterns: Dict[str, str] = None):
        """
        Initializes the redactor with a set of regex patterns.
        
        Args:
            custom_patterns: Optional dictionary of PII labels and their regex strings.
        """
        self._compiled_patterns: Dict[str, Pattern] = {}
        patterns_to_compile = self.DEFAULT_PATTERNS.copy()
        
        if custom_patterns:
            patterns_to_compile.update(custom_patterns)

        for label, regex_str in patterns_to_compile.items():
            try:
                self._compiled_patterns[label] = re.compile(regex_str)
            except re.error as e:
                logger.error(f"Failed to compile regex for '{label}': {e}")

    @property
    def node_name(self) -> str:
        """Returns the canonical name of the node."""
        return "PIIRedactorNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to redact sensitive information.
        
        Args:
            data: The input data, typically a string, list of strings, or nested dict.
            context: Orchestration context (unused in this node but required by interface).
            
        Returns:
            The input data with PII masked by placeholders.
        """
        if data is None:
            return None

        try:
            return self._apply_redaction(data)
        except Exception as e:
            logger.error(f"Critical error during PII redaction in {self.node_name}: {str(e)}")
            raise RuntimeError(f"Redaction process failed: {e}") from e

    def _apply_redaction(self, data: Any) -> Any:
        """
        Recursively traverses data structures to redact strings.
        """
        if isinstance(data, str):
            return self._redact_text(data)
        elif isinstance(data, list):
            return [self._apply_redaction(item) for item in data]
        elif isinstance(data, dict):
            return {k: self._apply_redaction(v) for k, v in data.items()}
        
        # Return non-string/non-container types as is
        return data

    def _redact_text(self, text: str) -> str:
        """
        Iterates through compiled patterns and replaces matches with labels.
        """
        redacted_text = text
        for label, pattern in self._compiled_patterns.items():
            replacement = f"<{label.upper()}>"
            redacted_text = pattern.sub(replacement, redacted_text)
        
        return redacted_text

    def __repr__(self) -> str:
        return f"<{self.node_name}(patterns={list(self._compiled_patterns.keys())})>"

# Internal note: Consider adding context-aware redaction (e.g., via Spacy/Presidio) 
# in future iterations if simple regex proves insufficient for specific locales.