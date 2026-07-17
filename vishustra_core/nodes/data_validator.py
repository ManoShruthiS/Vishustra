import logging
from typing import Any, Dict, List, Callable, Union

# Assuming BaseNode is located here as per project context instruction
from vishustra_core.nodes.base_node import BaseNode

# Configure logging for this module
logger = logging.getLogger(__name__)

class DataValidationError(ValueError):
    """Custom exception raised when data fails validation checks within DataValidatorNode."""
    pass

class DataValidatorNode(BaseNode):
    """
    A Vishustra processing node that validates input data against a defined schema.

    The node ensures that incoming data adheres to specified structure, types,
    and constraints, raising a `DataValidationError` if any rule is violated.

    Schema structure example:
    ```python
    schema = {
        "user_id": {
            "type": int,               # Expected Python type (e.g., str, int, float, list, dict, bool)
            "required": True,          # Boolean: True if the field must be present
            "min_value": 1             # Optional: Minimum value for numbers
        },
        "username": {
            "type": str,
            "required": True,
            "min_length": 3,           # Optional: Minimum length for strings/lists/dicts
            "max_length": 50,          # Optional: Maximum length for strings/lists/dicts
            "validator": lambda x: x.isalnum() # Optional: Custom callable validator (returns bool)
        },
        "email_addresses": {
            "type": list,              # For sequences, specify `list` or `tuple`
            "required": False,
            "min_items": 1,            # Optional: Minimum number of items for lists/tuples
            "item_type": str,          # Optional: Expected type of items within the list/tuple
            "max_items": 5
        },
        "is_active": {
            "type": bool,
            "required": True
        }
    }
    ```
    """

    def __init__(self, schema: Dict[str, Any]):
        """
        Initializes the DataValidatorNode with a validation schema.

        Args:
            schema (Dict[str, Any]): A dictionary defining the validation rules.
                                     Refer to the class docstring for the detailed
                                     schema structure and supported rules.
        """
        self._schema = schema
        self._logger = logging.getLogger(self.node_name)
        self._logger.debug(f"DataValidatorNode initialized with schema for fields: {list(schema.keys())}")

    @property
    def node_name(self) -> str:
        """Returns the programmatic name of the node."""
        return "DataValidator"

    def _validate_field(self, field_name: str, value: Any, rules: Dict[str, Any]) -> None:
        """
        Applies a set of validation rules to a single field's value.

        Args:
            field_name (str): The name of the field being validated.
            value (Any): The value of the field.
            rules (Dict[str, Any]): The validation rules dictionary for this specific field.

        Raises:
            DataValidationError: If any specified validation rule fails for the field.
            Exception: If a custom validator callable raises an unexpected error.
        """
        # Type check
        if "type" in rules:
            expected_type = rules["type"]
            # isinstance works for concrete types (str, int, list, dict, etc.)
            if not isinstance(value, expected_type):
                raise DataValidationError(
                    f"Field '{field_name}' type mismatch. Expected `{expected_type.__name__}`, "
                    f"but received `{type(value).__name__}`."
                )

        # Length checks for sequences (str, list, tuple, dict)
        if isinstance(value, (str, list, tuple, dict)):
            if "min_length" in rules and len(value) < rules["min_length"]:
                raise DataValidationError(
                    f"Field '{field_name}' length is {len(value)}, which is less than "
                    f"the required minimum of {rules['min_length']}."
                )
            if "max_length" in rules and len(value) > rules["max_length"]:
                raise DataValidationError(
                    f"Field '{field_name}' length is {len(value)}, which exceeds "
                    f"the allowed maximum of {rules['max_length']}."
                )

        # Value range checks for numbers (int, float)
        if isinstance(value, (int, float)):
            if "min_value" in rules and value < rules["min_value"]:
                raise DataValidationError(
                    f"Field '{field_name}' value is {value}, which is less than "
                    f"the required minimum of {rules['min_value']}."
                )
            if "max_value" in rules and value > rules["max_value"]:
                raise DataValidationError(
                    f"Field '{field_name}' value is {value}, which exceeds "
                    f"the allowed maximum of {rules['max_value']}."
                )

        # Item type and count checks for sequences (lists, tuples)
        if isinstance(value, (list, tuple)):
            if "item_type" in rules:
                expected_item_type = rules["item_type"]
                for i, item in enumerate(value):
                    if not isinstance(item, expected_item_type):
                        raise DataValidationError(
                            f"Field '{field_name}' item at index {i} type mismatch. "
                            f"Expected `{expected_item_type.__name__}`, but received `{type(item).__name__}`."
                        )
            if "min_items" in rules and len(value) < rules["min_items"]:
                raise DataValidationError(
                    f"Field '{field_name}' has {len(value)} items, which is less than "
                    f"the required minimum of {rules['min_items']}."
                )
            if "max_items" in rules and len(value) > rules["max_items"]:
                raise DataValidationError(
                    f"Field '{field_name}' has {len(value)} items, which exceeds "
                    f"the allowed maximum of {rules['max_items']}."
                )

        # Custom callable validator
        if "validator" in rules and callable(rules["validator"]):
            try:
                if not rules["validator"](value):
                    raise DataValidationError(f"Field '{field_name}' failed custom validation check.")
            except Exception as e:
                self._logger.error(
                    f"Custom validator for field '{field_name}' raised an unexpected exception: {e}",
                    exc_info=True
                )
                raise DataValidationError(
                    f"Custom validator for field '{field_name}' encountered an internal error. "
                    f"Details: {e}"
                ) from e

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by validating it against the configured schema.

        The `DataValidatorNode` expects the input `data` to be a dictionary,
        which it then validates against the schema provided during initialization.

        Args:
            data (Any): The input data to be validated. Expected to be a dictionary.
            context (Dict[str, Any]): A dictionary providing runtime context,
                                      not directly used for validation rules in this node.

        Returns:
            Any: The original input `data` if all validation checks pass successfully.

        Raises:
            DataValidationError: If the input data is not a dictionary, or if any
                                 validation rule fails for any field in the data.
            Exception: For any other unexpected errors during the validation process.
        """
        self._logger.debug(f"DataValidatorNode: Starting validation process for incoming data.")

        if not isinstance(data, dict):
            error_msg = (
                f"Input data to {self.node_name} must be a dictionary for schema-based validation. "
                f"Received type: `{type(data).__name__}`."
            )
            self._logger.error(error_msg)
            raise DataValidationError(error_msg)

        validation_errors: List[str] = []

        # Iterate through the schema to apply rules and check for required fields
        for field_name, rules in self._schema.items():
            is_required = rules.get("required", False)
            field_present = field_name in data

            if is_required and not field_present:
                validation_errors.append(f"Missing required field: '{field_name}'.")
                continue  # Cannot validate further rules if the field itself is missing

            if field_present:
                try:
                    self._validate_field(field_name, data[field_name], rules)
                except DataValidationError as dve:
                    # Collect specific validation errors
                    validation_errors.append(str(dve))
                except Exception as e:
                    # Catch unexpected errors during field validation
                    self._logger.exception(
                        f"An unexpected error occurred during validation of field '{field_name}'."
                    )
                    validation_errors.append(
                        f"An unexpected internal error occurred validating field '{field_name}': {type(e).__name__} - {e}."
                    )
            # If the field is not required and not present, it's considered valid
            # (i.e., no validation rules need to be applied).

        if validation_errors:
            # Aggregate all validation errors into a single, comprehensive message
            error_summary = (
                f"Data validation failed for input data with {len(validation_errors)} error(s):\n" +
                "\n".join(f"- {err}" for err in validation_errors)
            )
            self._logger.warning(error_summary)
            raise DataValidationError(error_summary)

        self._logger.info("Data successfully passed all validation checks and is considered valid.")
        return data