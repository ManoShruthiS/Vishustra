import re
import logging
from typing import Any, Dict

# Assuming BaseNode is located at vishustra_core/nodes/base_node.py
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class PIIRedactorNode(BaseNode):
    """
    A Vishustra processing node responsible for redacting Personally Identifiable
    Information (PII) from input text data.

    This node identifies common PII patterns such as email addresses, phone numbers,
    US Social Security Numbers (SSN), and credit card numbers using regular expressions
    and replaces them with generic redacted placeholders.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "PII Redactor"

    def __init__(self):
        """
        Initializes the PIIRedactorNode with predefined PII patterns.
        """
        self._pii_patterns = [
            # Email Address
            (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'), '[REDACTED_EMAIL]'),
            # Phone Numbers (various formats: (123) 456-7890, 123-456-7890, +1 123 456 7890)
            (re.compile(r'\b(?:\+\d{1,3}[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b'), '[REDACTED_PHONE]'),
            # US Social Security Number (XXX-XX-XXXX)
            (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), '[REDACTED_SSN]'),
            # Credit Card Numbers (13-16 digits, with or without spaces/hyphens)
            # This is a simplified pattern and might falsely identify other long numbers.
            # For production, consider Luhn algorithm validation or dedicated libraries.
            (re.compile(r'\b(?:4\d{3}|5[1-5]\d{2}|6011|3[47]\d{2})[- ]?(?:\d{4}[- ]?){2}\d{3,4}\b'), '[REDACTED_CREDIT_CARD]'),
        ]
        logger.info(f"Initialized {self.node_name} with {len(self._pii_patterns)} PII detection patterns.")

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to identify and redact PII.

        Args:
            data: The input data, expected to be a string containing text.
            context: A dictionary containing context-specific information for processing.

        Returns:
            The processed data with PII redacted. If the input data is not
            a string, it is returned as is with a warning.
        """
        if not isinstance(data, str):
            logger.warning(
                f"{self.node_name}: Input data is not a string (type: {type(data).__name__}). "
                "PII redaction will be skipped. Returning data as-is."
            )
            return data

        redacted_data = data
        redaction_count = 0

        for pattern, replacement_text in self._pii_patterns:
            original_data_before_sub = redacted_data
            redacted_data = pattern.sub(replacement_text, redacted_data)
            
            # Check if any substitutions were made by this pattern
            if original_data_before_sub != redacted_data:
                # Count occurrences replaced by this specific pattern in this pass
                # Note: this is an approximation if previous patterns already redacted overlapping text
                current_pattern_matches = len(pattern.findall(original_data_before_sub))
                redaction_count += current_pattern_matches
                logger.debug(
                    f"{self.node_name}: Redacted {current_pattern_matches} instances "
                    f"using pattern for '{replacement_text.replace('[REDACTED_','').replace(']','')}'."
                )
        
        if redaction_count > 0:
            logger.info(
                f"{self.node_name}: Successfully redacted {redaction_count} potential PII instances."
            )
        else:
            logger.debug(f"{self.node_name}: No PII found or redacted in the provided data.")

        return redacted_data

if __name__ == '__main__':
    # Example usage for testing
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    pii_redactor = PIIRedactorNode()

    test_data_1 = (
        "Hello, my name is John Doe. You can reach me at john.doe@example.com "
        "or call me at +1 (555) 123-4567. My SSN is 123-45-6789. "
        "Also, don't use credit card 1111-2222-3333-4444."
    )
    test_data_2 = "No PII here, just a regular sentence."
    test_data_3 = 12345 # Non-string data
    test_data_4 = (
        "Please contact me via email at test@mail.co.uk or on my mobile "
        "07700 900333. My MasterCard is 5432 1098 7654 3210. Another phone number: 012-345-6789."
    )

    print("\n--- Test Case 1 ---")
    print("Original:", test_data_1)
    redacted_1 = pii_redactor.process(test_data_1, {})
    print("Redacted:", redacted_1)

    print("\n--- Test Case 2 ---")
    print("Original:", test_data_2)
    redacted_2 = pii_redactor.process(test_data_2, {})
    print("Redacted:", redacted_2)

    print("\n--- Test Case 3 (Non-string) ---")
    print("Original:", test_data_3)
    redacted_3 = pii_redactor.process(test_data_3, {})
    print("Redacted:", redacted_3)

    print("\n--- Test Case 4 (More PII) ---")
    print("Original:", test_data_4)
    redacted_4 = pii_redactor.process(test_data_4, {})
    print("Redacted:", redacted_4)
