import logging
import re
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class PIIRedactorNode(BaseNode):
    """
    A processing node that redacts personally identifiable information (PII)
    from text data.

    This node identifies common PII patterns such as email addresses and
    phone numbers using regular expressions and replaces them with a
    configurable placeholder.

    Supported data types for redaction:
    -   str: The string itself will be processed.
    -   dict: String values within the dictionary will be processed recursively.
    -   list (of strings or dicts): Each element will be processed.
    -   Other types: Returned as-is with a warning.
    """

    def __init__(self,
                 redaction_placeholder: str = "[REDACTED_PII]",
                 email_pattern: str = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
                 phone_pattern: str = r"(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})|(\+?\d{1,3}[-.\s]?\(?\d{2,3}\)?[-.\s]?\d{3,4}[-.\s]?\d{4})"):
        """
        Initializes the PIIRedactorNode with redaction patterns and placeholder.

        Args:
            redaction_placeholder (str): The string used to replace identified PII.
            email_pattern (str): Regular expression pattern for email addresses.
            phone_pattern (str): Regular expression pattern for phone numbers.
        """
        self._redaction_placeholder = redaaction_placeholder
        self._email_regex = re.compile(email_pattern)
        self._phone_regex = re.compile(phone_pattern)
        logger.debug(f"PIIRedactorNode initialized with placeholder: '{redaction_placeholder}'")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "PIIRedactorNode"

    def _redact_text(self, text: str) -> str:
        """Applies redaction to a single string."""
        original_text = text
        text = self._email_regex.sub(self._redaction_placeholder, text)
        text = self._phone_regex.sub(self._redaction_placeholder, text)
        if text != original_text:
            logger.debug("PII redacted from text string.")
        return text

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to redact PII.

        Args:
            data (Any): The input data, expected to be a string, dict, or list.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the processing. Not directly used for
                                       configuration in this node, but available.

        Returns:
            Any: The data with PII redacted, or the original data if not a
                 supported type.
        """
        logger.info(f"[{self.node_name}] Starting PII redaction for incoming data.")

        # Configuration could potentially be overridden by context here, e.g.:
        # if 'redaction_placeholder' in context:
        #     self._redaction_placeholder = context['redaction_placeholder']

        try:
            if isinstance(data, str):
                redacted_data = self._redact_text(data)
                logger.debug(f"[{self.node_name}] Redacted a string successfully.")
                return redacted_data
            elif isinstance(data, dict):
                redacted_data = {}
                for key, value in data.items():
                    redacted_data[key] = self.process(value, context)  # Recursive call
                logger.debug(f"[{self.node_name}] Redacted values within a dictionary.")
                return redacted_data
            elif isinstance(data, list):
                redacted_data = [self.process(item, context) for item in data]  # Recursive call
                logger.debug(f"[{self.node_name}] Redacted items within a list.")
                return redacted_data
            else:
                logger.warning(
                    f"[{self.node_name}] Unsupported data type '{type(data)}' for PII redaction. "
                    "Returning data as-is."
                )
                return data
        except Exception as e:
            logger.error(f"[{self.node_name}] An error occurred during PII redaction: {e}", exc_info=True)
            # Depending on policy, might re-raise, return original, or return a failure object
            raise # Re-raising for critical operational failure within the node
        finally:
            logger.info(f"[{self.node_name}] Finished PII redaction processing.")

# Example of how to use this node (for testing/demonstration, not part of the class)
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    redactor = PIIRedactorNode()

    test_data_str = "Please contact me at john.doe@example.com or call 123-456-7890 for details. My other number is +1 (987) 654-3210."
    test_data_dict = {
        "user_message": "My email is test@domain.org and my work phone is (111) 222-3333.",
        "incident_id": "INC-001",
        "nested_info": {
            "contact": "alice@company.net",
            "support_line": "555.123.4567"
        },
        "metadata": ["Some text here", "No PII here either"]
    }
    test_data_list = [
        "Reach out to bob@mail.com",
        {"id": 1, "tel": "000-000-0000"}
    ]
    test_data_unsupported = 12345

    print("\n--- Processing String ---")
    redacted_str = redactor.process(test_data_str, {})
    print(f"Original: {test_data_str}")
    print(f"Redacted: {redacted_str}")

    print("\n--- Processing Dictionary ---")
    redacted_dict = redactor.process(test_data_dict, {})
    print(f"Original: {test_data_dict}")
    print(f"Redacted: {redacted_dict}")

    print("\n--- Processing List ---")
    redacted_list = redactor.process(test_data_list, {})
    print(f"Original: {test_data_list}")
    print(f"Redacted: {redacted_list}")

    print("\n--- Processing Unsupported Type ---")
    redacted_unsupported = redactor.process(test_data_unsupported, {})
    print(f"Original: {test_data_unsupported}")
    print(f"Redacted: {redacted_unsupported}")

    # Example with custom placeholder
    custom_redactor = PIIRedactorNode(redaction_placeholder="[MASKED]")
    print("\n--- Processing String with Custom Placeholder ---")
    redacted_str_custom = custom_redactor.process(test_data_str, {})
    print(f"Original: {test_data_str}")
    print(f"Redacted: {redacted_str_custom}")