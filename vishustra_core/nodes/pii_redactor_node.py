import logging
import re
from typing import Any, Dict, List, Tuple

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class PIIRedactorNode(BaseNode):
    """
    A Vishustra processing node designed to identify and redact personally
    identifiable information (PII) from string data.

    This node simulates PII detection and replacement with a specified
    redaction string. It's configurable with custom PII patterns and
    provides robust logging for detected and redacted items.
    """

    def __init__(self, redaction_char: str = "[REDACTED]", pii_patterns: List[Tuple[str, str]] = None):
        """
        Initializes the PIIRedactorNode.

        Args:
            redaction_char (str): The string used to replace detected PII.
                                  Defaults to "[REDACTED]".
            pii_patterns (List[Tuple[str, str]], optional): A list of tuples, where each tuple
                                                            contains (regex_pattern_string, pii_type_name).
                                                            If None, a default set of common PII patterns is used.
        """
        self._redaction_char = redaction_char
        if pii_patterns is None:
            # Default, simple patterns for simulating PII detection.
            # In a production system, these patterns would be much more sophisticated,
            # potentially leveraging NLP libraries (e.g., spaCy, Presidio) for
            # named entity recognition and advanced PII detection.
            self._pii_patterns = [
                (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'Email Address'),
                (r'\b(?:\d{3}[-.\s]?\d{3}[-.\s]?\d{4}|\(\d{3}\)\s*\d{3}-\d{4})\b', 'Phone Number'),
                (r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b', 'Social Security Number (SSN)'),
                # More complex patterns for credit cards or names might be added here,
                # but require careful balancing to avoid false positives.
            ]
        else:
            if not all(isinstance(p, tuple) and len(p) == 2 and isinstance(p[0], str) and isinstance(p[1], str) for p in pii_patterns):
                raise ValueError("pii_patterns must be a list of (regex_pattern_string, pii_type_name) tuples.")
            self._pii_patterns = pii_patterns
        
        # Compile regex patterns for efficiency during processing
        self._compiled_patterns = [(re.compile(pattern), pii_type) for pattern, pii_type in self._pii_patterns]
        logger.info("Node '%s' initialized with %d PII patterns.", self.node_name, len(self._compiled_patterns))


    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "PIIRedactor"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to identify and redact PII.

        If the input `data` is a string, it iterates through configured PII patterns,
        detects matches, and replaces them with the `redaction_char`.
        For other data types, it logs a warning and returns the data unchanged,
        as PII redaction is primarily applicable to textual content.

        Args:
            data (Any): The input data to be processed. Expected to be a string
                        containing text potentially with PII.
            context (Dict[str, Any]): A dictionary containing contextual information
                                     for the processing pipeline. This node does
                                     not currently use the context for its core logic.

        Returns:
            Any: The data with PII redacted if it was a string, otherwise the
                 original data is returned untouched.
        """
        if not isinstance(data, str):
            logger.warning(
                "Node '%s': Input data is not a string (type: %s). PII redaction will be skipped.",
                self.node_name,
                type(data).__name__
            )
            return data

        redacted_data = data
        total_redaction_count = 0
        
        logger.info("Node '%s': Starting PII redaction process.", self.node_name)

        for pattern, pii_type in self._compiled_patterns:
            try:
                # Find all occurrences to log them for debugging/auditing before redaction
                found_matches = pattern.findall(redacted_data)
                if found_matches:
                    logger.debug(
                        "Node '%s': Detected %d occurrence(s) of '%s' PII type. Examples: %s",
                        self.node_name,
                        len(found_matches),
                        pii_type,
                        ', '.join([repr(m) for m in found_matches[:3]]) + ('...' if len(found_matches) > 3 else '')
                    )
                    
                    # Replace with the configured redaction character
                    redacted_data = pattern.sub(self._redaction_char, redacted_data)
                    total_redaction_count += len(found_matches)
            except Exception as e:
                logger.error(
                    "Node '%s': An error occurred while processing PII type '%s' with pattern '%s': %s",
                    self.node_name, pii_type, pattern.pattern, e
                )
                # Decide whether to re-raise or continue. For PII, continuing with other patterns is generally safer.
                
        if total_redaction_count > 0:
            logger.info(
                "Node '%s': Completed PII redaction. %d PII item(s) redacted.",
                self.node_name,
                total_redaction_count
            )
        else:
            logger.info("Node '%s': No PII detected or redacted.", self.node_name)

        return redacted_data