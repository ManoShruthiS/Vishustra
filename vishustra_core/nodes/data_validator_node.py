import logging
import re
from typing import Any, Dict, List, Callable, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class DataValidatorNode(BaseNode):
    """
    A processing node that validates input data against a defined schema.

    This node is crucial for ensuring data quality and consistency within the
    orchestration framework. It allows for specification of expected types,
    string lengths, numeric ranges, regex patterns, and more for various fields.
    """

    def __init__(self, validation_schema: Dict[str, Dict[str, Any]]):
        """
        Initializes the DataValidatorNode with a validation schema.

        The validation_schema is a dictionary where keys are expected data fields
        and values are dictionaries defining validation rules for that field.

        Example schema structure:
        {
            "user_id": {"type": int, "min_value": 1},
            "username": {"type": str, "min_length": 3, "max_length": 50},
            "email": {"type": str, "regex": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"},
            "is_active": {"type": bool, "required": True},
            "tags": {"type": list, "item_type": str, "max_items": 10, "required": False},
            "metadata": {"type": dict, "schema": {"source": {"type": str, "required": True}}},
            "optional_field": {"type": str, "required": False}
        }

        Supported rules for fields:
        - "type": Expected Python type (e.g., int, str, bool, list, dict).
        - "required": bool (default: True). If False, a missing field or None value is not an error.
        - "min_length": int (for str, list, dict - applies to len()).
        - "max_length": int (for str, list, dict - applies to len()).
        - "min_value": Union[int, float] (for int, float).
        - "max_value": Union[int, float] (for int, float).
        - "regex": str (for str). A regex pattern to match.
        - "enum": List[Any] (for any type). A list of allowed values.
        - "custom_validator": Callable[[Any], bool] (for any type). A custom function
                              that returns True for valid, False for invalid.
        - "item_type": Type (for list). Expected type for elements in a list.
        - "schema": Dict (for dict). A nested validation schema for dictionary fields.
        """
        if not isinstance(validation_schema, dict):
            raise TypeError("`validation_schema` must be a dictionary.")
        self._validation_schema = validation_schema
        logger.debug(f"DataValidatorNode initialized with schema: {self._validation_schema}")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "DataValidator"

    def _validate_field(self, field_name: str, value: Any, rules: Dict[str, Any]) -> None:
        """
        Internal method to validate a single field against its rules.
        Raises ValueError if validation fails.
        """
        expected_type = rules.get("type")
        required = rules.get("required", True) # Default to required

        # If the value is None and not required, it's considered valid for this field.
        # No further rules (type, length, value) apply to a non-required None.
        if value is None and not required:
            return

        # If the value is None but required, it's an error.
        if value is None and required:
            raise ValueError(f"Field '{field_name}' is required but its value is None.")

        # Type check
        if expected_type and not isinstance(value, expected_type):
            raise ValueError(
                f"Field '{field_name}' has invalid type. Expected '{expected_type.__name__}', "
                f"got '{type(value).__name__}' with value '{value}'."
            )

        # Enum check
        if "enum" in rules and value not in rules["enum"]:
            raise ValueError(f"Field '{field_name}' value '{value}' is not in allowed enum: {rules['enum']}.")

        # Length checks (for str, list, dict)
        if isinstance(value, (str, list, dict)):
            current_length = len(value)
            min_length = rules.get("min_length")
            max_length = rules.get("max_length")

            if min_length is not None and current_length < min_length:
                raise ValueError(
                    f"Field '{field_name}' length ({current_length}) is less than minimum required ({min_length})."
                )
            if max_length is not None and current_length > max_length:
                raise ValueError(
                    f"Field '{field_name}' length ({current_length}) exceeds maximum allowed ({max_length})."
                )
        
        # Numeric range checks (for int, float)
        if isinstance(value, (int, float)):
            min_value = rules.get("min_value")
            max_value = rules.get("max_value")

            if min_value is not None and value < min_value:
                raise ValueError(f"Field '{field_name}' value ({value}) is less than minimum allowed ({min_value}).")
            if max_value is not None and value > max_value:
                raise ValueError(f"Field '{field_name}' value ({value}) exceeds maximum allowed ({max_value}).")

        # Regex check (for str)
        if isinstance(value, str) and "regex" in rules:
            pattern = rules["regex"]
            if not re.fullmatch(pattern, value):
                raise ValueError(f"Field '{field_name}' value '{value}' does not match regex pattern '{pattern}'.")

        # List item type check
        if isinstance(value, list) and "item_type" in rules:
            item_type = rules["item_type"]
            for i, item in enumerate(value):
                if not isinstance(item, item_type):
                    raise ValueError(
                        f"Field '{field_name}' list item at index {i} has invalid type. "
                        f"Expected '{item_type.__name__}', got '{type(item).__name__}'."
                    )
        
        # Nested dictionary schema validation
        if isinstance(value, dict) and "schema" in rules:
            nested_schema = rules["schema"]
            try:
                # Recursively call process on nested data with nested schema
                # Create a temporary validator for the nested schema
                nested_validator = DataValidatorNode(nested_schema)
                # Pass an empty context to the nested validator, as its context is distinct
                nested_validator.process(value, context={}) 
            except (ValueError, TypeError) as e:
                raise ValueError(f"Field '{field_name}' nested schema validation failed: {e}") from e

        # Custom validator check
        custom_validator = rules.get("custom_validator")
        if isinstance(custom_validator, Callable):
            if not custom_validator(value):
                raise ValueError(f"Field '{field_name}' failed custom validation with value '{value}'.")

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Validates the input data against the configured validation schema.

        Args:
            data: The input data to be validated. Expected to be a dictionary
                  if a validation_schema is provided.
            context: A dictionary containing contextual information,
                     not directly used for validation in this node but passed along.

        Returns:
            The original data if all validations pass.

        Raises:
            ValueError: If the data fails any validation rule defined in the schema.
            TypeError: If the input data is not a dictionary when a schema is present.
        """
        logger.info("Starting data validation process.")

        if not self._validation_schema:
            logger.info("No validation schema provided. Data passed through without validation.")
            return data

        if not isinstance(data, dict):
            raise TypeError(
                f"DataValidatorNode expects dictionary data when a schema is defined. "
                f"Received type: {type(data).__name__}."
            )

        # First, iterate through the schema to check for required fields that are entirely missing.
        for field_name, rules in self._validation_schema.items():
            required = rules.get("required", True)
            if required and field_name not in data:
                raise ValueError(f"Required field '{field_name}' is missing from data.")

        # Now, iterate through the schema again to validate each field present or specified.
        for field_name, rules in self._validation_schema.items():
            value = data.get(field_name) # Use .get() to retrieve value or None if field is missing
            try:
                self._validate_field(field_name, value, rules)
            except (ValueError, TypeError) as e:
                logger.error(f"Validation failed for field '{field_name}': {e}")
                # Re-raise the validation error, possibly wrapped for better context
                raise ValueError(f"Data validation failed for field '{field_name}'. Details: {e}") from e

        logger.info("Data validation completed successfully.")
        return data