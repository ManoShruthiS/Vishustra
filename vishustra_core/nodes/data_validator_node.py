import logging
import re
from typing import Any, Dict, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class DataValidationException(Exception):
    """Custom exception for data validation failures."""
    def __init__(self, message: str, errors: Dict[str, str] = None):
        super().__init__(message)
        self.errors = errors if errors is not None else {}

class DataValidatorNode(BaseNode):
    """
    A processing node designed to validate input data against a defined schema.

    This node expects a validation schema to be provided within the `context`
    dictionary, under the key 'validation_schema'. The schema is a dictionary
    where keys correspond to expected fields in the `data` and values define
    validation rules for each field.

    Example `validation_schema` structure in `context`:
    {
        "user_id": {"type": int, "required": True, "min_value": 1},
        "username": {"type": str, "required": True, "min_length": 3, "max_length": 50},
        "email": {"type": str, "regex": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", "required": False},
        "age": {"type": int, "min_value": 0, "max_value": 120, "required": False},
        "is_active": {"type": bool, "required": True}
    }
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "DataValidatorNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Validates the input data against the specified schema from the context.

        Args:
            data: The input data to be validated. This node primarily supports
                  dictionary-like data structures for schema validation.
            context: A dictionary containing runtime information, expected to
                     include a 'validation_schema' key with validation rules.

        Returns:
            The original, validated data if all checks pass.

        Raises:
            DataValidationException: If validation fails for any reason
                                     (e.g., data mismatch, schema misconfiguration).
            TypeError: If the input data is not a dictionary when a schema is present,
                       or if the schema is malformed.
        """
        validation_schema = context.get("validation_schema")

        if not validation_schema:
            msg = "Validation schema not found in context. DataValidatorNode requires 'validation_schema' to operate."
            logger.error(msg)
            raise DataValidationException(msg)

        if not isinstance(validation_schema, dict):
            msg = f"Provided 'validation_schema' in context is not a dictionary but type '{type(validation_schema).__name__}'. Malformed schema."
            logger.error(msg)
            raise TypeError(msg)

        if not isinstance(data, dict):
            msg = f"DataValidatorNode received non-dictionary data (type: '{type(data).__name__}') but a dictionary is required for schema-based validation."
            logger.error(msg)
            raise TypeError(msg)
            
        errors: Dict[str, str] = {}

        for field_name, rules in validation_schema.items():
            if not isinstance(rules, dict):
                errors[field_name] = f"Validation rules for field '{field_name}' are not a dictionary."
                continue

            is_required = rules.get("required", False)
            field_value = data.get(field_name)

            # Check for required fields
            if field_value is None:
                if is_required:
                    errors[field_name] = f"Field '{field_name}' is required but missing."
                continue # If not required and missing, no further validation needed for this field.

            # Type validation
            expected_type = rules.get("type")
            if expected_type:
                # Handle Union types for validation if needed, for simplicity we check exact type match or direct parent for now
                if not isinstance(field_value, expected_type):
                    errors[field_name] = (
                        f"Field '{field_name}' must be of type {expected_type.__name__}, "
                        f"but received {type(field_value).__name__}."
                    )
                    continue # Skip further checks for this field if type is already wrong

            # Type-specific validations
            if expected_type is str:
                min_length = rules.get("min_length")
                max_length = rules.get("max_length")
                regex = rules.get("regex")

                if min_length is not None and len(field_value) < min_length:
                    errors[field_name] = f"Field '{field_name}' must be at least {min_length} characters long."
                if max_length is not None and len(field_value) > max_length:
                    errors[field_name] = f"Field '{field_name}' must be at most {max_length} characters long."
                if regex:
                    try:
                        if not re.fullmatch(regex, field_value):
                            errors[field_name] = f"Field '{field_name}' does not match the required pattern '{regex}'."
                    except re.error:
                        logger.warning(f"Invalid regex pattern '{regex}' for field '{field_name}'. Skipping regex validation.")
                        errors[field_name] = f"Invalid regex pattern provided for field '{field_name}'."


            elif expected_type in (int, float):
                min_value = rules.get("min_value")
                max_value = rules.get("max_value")

                if min_value is not None and field_value < min_value:
                    errors[field_name] = f"Field '{field_name}' must be at least {min_value}."
                if max_value is not None and field_value > max_value:
                    errors[field_name] = f"Field '{field_name}' must be at most {max_value}."
            
            # Future extensions could include:
            # - Enum validation: 'enum': ['val1', 'val2']
            # - List item validation: 'items': {'type': str, 'min_length': 1}
            # - Custom validation functions: 'validator': my_custom_func

        if errors:
            error_summary = f"Data validation failed for node '{self.node_name}'. Found {len(errors)} error(s)."
            logger.warning(f"{error_summary} Details: {errors}")
            raise DataValidationException(error_summary, errors=errors)

        logger.info(f"Data successfully validated by '{self.node_name}'.")
        return data
