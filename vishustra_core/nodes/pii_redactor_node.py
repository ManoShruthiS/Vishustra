import re
import logging
from typing import Any, Dict, List, Optional
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class PIIRedactorNode(BaseNode):
    """
    A node responsible for identifying and redacting Personally Identifiable Information (PII)
    from text data. Uses regex-based patterns to mask sensitive entities like emails, 
    phone numbers, and credit card patterns to ensure data privacy before further processing.
    """

    DEFAULT_PATTERNS = {
        "EMAIL": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
        "PHONE": r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
        "CREDIT_CARD": r"\b(?:\d[ -]*?){13,16}\b",
        "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
        "IPV4": r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    }

    def __init__(self, custom_patterns: Optional[Dict[str, str]] = None):
        """
        Initializes the redactor with default patterns or extends them with custom ones.
        """
        self.patterns = self.DEFAULT_PATTERNS.copy()
        if custom_patterns:
            self.patterns.update(custom_patterns)
        
        # Pre-compile patterns for performance
        self._compiled_regex = {
            label: re.compile(pattern) for label, pattern in self.patterns.items()
        }

    @property
    def node_name(self) -> str:
        """
        Returns the unique identifier for this node type.
        """
        return "PIIRedactorNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Scans the input data for PII and replaces matches with a redacted placeholder.
        
        Args:
            data: The input to process. Expected to be a string.
            context: Execution context containing metadata or configurations.
            
        Returns:
            The redacted string if input was a string, otherwise the original data.
        """
        if not isinstance(data, str):
            logger.warning(
                "Node [%s] received non-string data of type %s. Skipping redaction.",
                self.node_name, 
                type(data).__name__
            )
            return data

        try:
            redacted_text = data
            for label, regex in self._compiled_regex.items():
                replacement = f"<{label}_REDACTED>"
                redacted_text = regex.sub(replacement, redacted_text)

            logger.debug("Successfully performed PII redaction on input string.")
            return redacted_text

        except Exception as e:
            logger.error(
                "Node [%s] failed to process data: %s", 
                self.node_name, 
                str(e), 
                exc_info=True
            )
            # In a pipeline, we might prefer to fail-safe by returning original or raising
            # depending on the orchestration policy. Here we raise to notify the orchestrator.
            raise RuntimeError(f"PII Redaction failed: {e}") from e

    def __repr__(self) -> str:
        return f"<{self.node_name}(patterns={list(self.patterns.keys())})>"