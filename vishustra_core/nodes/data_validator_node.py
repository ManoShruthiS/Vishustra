import logging
import re
from typing import Any, Dict, List

# Assuming vishustra_core is a package and nodes is a subpackage
# and base_node is a module within nodes.
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class DataValidatorNode(BaseNode):
    """
    A Vishustra processing node that validates input data against a defined schema.

    This node expects 'data' to be a dictionary and applies various validation
    rules (type, required, min/max values, min/max lengths, regex patterns)
    based on the schema provided during its instantiation.

    If validation fails, a ValueError is raised detailing all identified issues.
    If validation passes, the original data is returned, allowing the pipeline
    to proceed with verified data.
    """

    def __init__(self, validation_schema: Dict[str, Dict[str, Any]]):
        """
        Initializes the DataValidatorNode with a validation schema.

        The validation schema is a dictionary where keys are field names
        and values are dictionaries defining validation rules for that field.

        Example schema structure:
        {
            "user_id": {"type": str, "required": True},
            "age": {"type": int, "required": True, "min": 0, "max": 150},
            "email": {"type": str, "required": False, "regex": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"},
            "tags": {"type": list, "required": False, "min_length": 1, "max_length": 10},
            "description": {"type": str, "max_length": 500}
        }

        Supported rules for each field:
        - "type": Expected Python type (e.g., str, int, float, list, dict).
        - "required": Boolean; if True, the field must be present in the data.
        - "min": Minimum value for numeric types (int, float).
        - "max": Maximum value for numeric types (int, float).
        - "min_length": Minimum length for sequence/mapping types (str, list, dict).
        - "max_length": Maximum length for sequence/mapping types (str, list, dict).
        - "regex": Regular expression pattern for string types. Uses `re.fullmatch`.
        """
        if not isinstance(validation_schema, dict):
            logger.error(f"Invalid type for validation_schema: Expected dict, got {type(validation_schema).__name__}")
            raise TypeError("Validation schema must be a dictionary.")

        self._validation_schema = validation_schema
        logger.debug(f"DataValidatorNode initialized with schema: {self._validation_schema}")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "DataValidatorNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, validating it against the configured schema.

        Args:
            data (Any): The data to be validated. Expected to be a dictionary.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       relevant to the current execution flow.
                                       (Not used by this node but part of the signature).

        Returns:
            Any: The original data if it passes all validation checks.

        Raises:
            TypeError: If the input 'data' is not a dictionary.
            ValueError: If the input 'data' fails any validation rule defined
                        in the schema.
        """
        if not isinstance(data, dict):
            logger.error(f"Input data for '{self.node_name}' must be a dictionary. Received: {type(data).__name__}")
            raise TypeError(
                f"Invalid input data type for '{self.node_name}'. Expected a dictionary, got {type(data).__name__}."
            )

        validation_errors: List[str] = []

        for field_name, rules in self._validation_schema.items():
            field_value = data.get(field_name)
            is_present = field_name in data

            # 1. Required check
            if rules.get("required") is True and not is_present:
                validation_errors.append(f"Field '{field_name}' is required but missing.")
                continue  # Skip further checks for a missing required field

            if is_present:  # Only apply further checks if the field is present
                # 2. Type check
                expected_type = rules.get("type")
                if expected_type and not isinstance(field_value, expected_type):
                    validation_errors.append(
                        f"Field '{field_name}' has incorrect type. Expected {expected_type.__name__}, "
                        f"got {type(field_value).__name__} with value '{field_value}'."
                    )

                # 3. Numeric range checks (min/max)
                if isinstance(field_value, (int, float)):
                    min_value = rules.get("min")
                    if min_value is not None and field_value < min_value:
                        validation_errors.append(
                            f"Field '{field_name}' value {field_value} is less than minimum allowed {min_value}."
                        )
                    max_value = rules.get("max")
                    if max_value is not None and field_value > max_value:
                        validation_errors.append(
                            f"Field '{field_name}' value {field_value} is greater than maximum allowed {max_value}."
                        )

                # 4. Length checks (min_length/max_length) for strings, lists, dicts
                if isinstance(field_value, (str, list, dict)):
                    field_length = len(field_value)
                    min_length = rules.get("min_length")
                    if min_length is not None and field_length < min_length:
                        validation_errors.append(
                            f"Field '{field_name}' length {field_length} is less than minimum allowed {min_length}."
                        )
                    max_length = rules.get("max_length")
                    if max_length is not None and field_length > max_length:
                        validation_errors.append(
                            f"Field '{field_name}' length {field_length} is greater than maximum allowed {max_length}."
                        )

                # 5. Regex check for strings
                regex_pattern = rules.get("regex")
                if regex_pattern and isinstance(field_value, str):
                    if not re.fullmatch(regex_pattern, field_value):
                        validation_errors.append(
                            f"Field '{field_name}' value '{field_value}' does not match required pattern '{regex_pattern}'."
                        )

        if validation_errors:
            error_message = f"Data validation failed for '{self.node_name}':\n" + "\n".join(
                [f"- {e}" for e in validation_errors]
            )
            logger.error(error_message)
            raise ValueError(error_message)
        else:
            logger.info(f"Data successfully validated by '{self.node_name}'.")
            return data