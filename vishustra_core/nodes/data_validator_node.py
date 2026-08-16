import logging
import re
from typing import Any, Dict, Type, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class VishustraValidationError(Exception):
    """Custom exception raised when data validation fails within Vishustra nodes."""
    pass

class DataValidatorNode(BaseNode):
    """
    A Vishustra processing node that validates input data against a defined schema.

    This node expects the input `data` to be a dictionary. The validation schema
    is retrieved from the `context` dictionary under the key `validation_schema`.

    The `validation_schema` should be a dictionary where keys are the expected
    field names in the input `data`, and values are dictionaries specifying
    validation rules for that specific field.

    Supported validation rules within a field's rule dictionary:
    - 'type': The expected Python type (e.g., `str`, `int`, `float`, `bool`, `list`, `dict`).
    - 'required': A boolean indicating if the field must be present (default: `False`).
    - 'min_length': Minimum length for strings, lists, or dictionaries.
    - 'max_length': Maximum length for strings, lists, or dictionaries.
    - 'min_value': Minimum numerical value for integers or floats.
    - 'max_value': Maximum numerical value for integers or floats.
    - 'pattern': A regular expression string for validating string fields.

    Example `validation_schema` in the `context`:
    ```python
    context = {
        "validation_schema": {
            "user_id": {"type": int, "required": True, "min_value": 1},
            "username": {"type": str, "required": True, "min_length": 3, "max_length": 50},
            "email": {"type": str, "required": False, "pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"},
            "age": {"type": int, "min_value": 0, "max_value": 120},
            "tags": {"type": list, "max_length": 10},
            "preferences": {"type": dict}
        }
    }
    ```
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "DataValidator"

    def _validate_field(self, field_name: str, field_value: Any, rules: Dict[str, Any]) -> None:
        """
        Helper method to validate a single field against its defined rules.

        Args:
            field_name: The name of the field being validated.
            field_value: The value of the field from the input data.
            rules: A dictionary of validation rules for this field.

        Raises:
            VishustraValidationError: If the field fails any validation rule.
        """
        # Check 'required' rule
        if rules.get("required", False) and field_value is None:
            raise VishustraValidationError(f"Field '{field_name}' is required but was not provided or is None.")

        # If not required and value is None, no further validation is needed for this field.
        if field_value is None:
            return

        # Check 'type' rule
        expected_type = rules.get("type")
        if expected_type is not None:
            if not isinstance(field_value, expected_type):
                raise VishustraValidationError(
                    f"Field '{field_name}' expected type '{expected_type.__name__}', "
                    f"but received '{type(field_value).__name__}'."
                )

        # Check 'min_length' and 'max_length' for sized types
        if isinstance(field_value, (str, list, dict)):
            length = len(field_value)
            min_length = rules.get("min_length")
            max_length = rules.get("max_length")

            if min_length is not None and not isinstance(min_length, int):
                raise TypeError(f"Validation rule 'min_length' for field '{field_name}' must be an integer.")
            if max_length is not None and not isinstance(max_length, int):
                raise TypeError(f"Validation rule 'max_length' for field '{field_name}' must be an integer.")

            if min_length is not None and length < min_length:
                raise VishustraValidationError(
                    f"Field '{field_name}' (length {length}) must have a minimum length of {min_length}."
                )
            if max_length is not None and length > max_length:
                raise VishustraValidationError(
                    f"Field '{field_name}' (length {length}) must have a maximum length of {max_length}."
                )

        # Check 'min_value' and 'max_value' for numeric types
        if isinstance(field_value, (int, float)):
            min_value = rules.get("min_value")
            max_value = rules.get("max_value")

            if min_value is not None and not isinstance(min_value, (int, float)):
                raise TypeError(f"Validation rule 'min_value' for field '{field_name}' must be a number.")
            if max_value is not None and not isinstance(max_value, (int, float)):
                raise TypeError(f"Validation rule 'max_value' for field '{field_name}' must be a number.")

            if min_value is not None and field_value < min_value:
                raise VishustraValidationError(
                    f"Field '{field_name}' (value {field_value}) must be greater than or equal to {min_value}."
                )
            if max_value is not None and field_value > max_value:
                raise VishustraValidationError(
                    f"Field '{field_name}' (value {field_value}) must be less than or equal to {max_value}."
                )

        # Check 'pattern' for string types
        if isinstance(field_value, str):
            pattern = rules.get("pattern")
            if pattern is not None:
                if not isinstance(pattern, str):
                    raise TypeError(f"Validation rule 'pattern' for field '{field_name}' must be a string (regex).")
                try:
                    if not re.match(pattern, field_value):
                        raise VishustraValidationError(
                            f"Field '{field_name}' (value '{field_value}') does not match required pattern '{pattern}'."
                        )
                except re.error as e:
                    logger.error(f"[{self.node_name}] Invalid regex pattern '{pattern}' for field '{field_name}': {e}")
                    raise TypeError(f"Invalid regex pattern provided for field '{field_name}': {pattern}") from e

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Validates the input data against a schema provided in the context.

        Args:
            data: The data to be validated. Expected to be a dictionary for schema validation.
            context: A dictionary containing operational context, including
                     'validation_schema' with validation rules.

        Returns:
            The original data if validation passes.

        Raises:
            VishustraValidationError: If the data fails any validation rule.
            TypeError: If the 'validation_schema' in context is malformed or invalid rule types are used.
            ValueError: If 'data' is not a dictionary when a validation schema is provided.
        """
        logger.info(f"[{self.node_name}] Starting data validation process.")

        validation_schema = context.get("validation_schema")

        if not validation_schema:
            logger.warning(
                f"[{self.node_name}] No 'validation_schema' found in context. "
                "Data will be passed through without validation, which might indicate a misconfiguration."
            )
            return data

        if not isinstance(validation_schema, dict):
            logger.error(
                f"[{self.node_name}] Configuration error: 'validation_schema' in context "
                f"is not a dictionary. Type: {type(validation_schema).__name__}."
            )
            raise TypeError("Validation schema must be a dictionary.")

        if not isinstance(data, dict):
            logger.error(
                f"[{self.node_name}] Input 'data' for schema validation is not a dictionary. "
                f"Type: {type(data).__name__}. Expected dict for field-level validation."
            )
            raise ValueError("Data to be validated against a schema must be a dictionary.")

        for field_name, rules in validation_schema.items():
            if not isinstance(rules, dict):
                logger.error(
                    f"[{self.node_name}] Configuration error: Validation rules for field '{field_name}' "
                    f"are not a dictionary. Type: {type(rules).__name__}."
                )
                raise TypeError(f"Validation rules for field '{field_name}' must be a dictionary.")

            # Retrieve field value, defaults to None if not present.
            # This allows 'required' rule to correctly identify missing fields.
            field_value = data.get(field_name)

            try:
                self._validate_field(str(field_name), field_value, rules)
            except VishustraValidationError as e:
                logger.error(f"[{self.node_name}] Data validation failed for field '{field_name}': {e}")
                raise
            except TypeError as e:
                logger.error(f"[{self.node_name}] Configuration error in validation rules for field '{field_name}': {e}")
                raise

        logger.info(f"[{self.node_name}] Data validation successful. All checks passed.")
        return data