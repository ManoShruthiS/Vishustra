
import logging
from typing import Any, Dict, Union, Callable, Tuple

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class DataValidationException(ValueError):
    """Custom exception raised when data fails validation against a defined schema."""
    pass


class DataValidator(BaseNode):
    """
    A Vishustra node designed to validate input data against a pre-defined schema.

    This node ensures data integrity and conformity by checking for:
    - Overall data type (e.g., ensuring the root data structure is a dictionary).
    - Presence of required fields within dictionary data.
    - Correct data types for specified fields (supporting single types or unions/tuples).
    - Custom validation logic via callable functions for specific fields.

    Configuration of validation rules is done via the `schema` provided during initialization.
    """

    def __init__(self, schema: Dict[str, Any]):
        """
        Initializes the DataValidator with a validation schema.

        The schema defines the rules for validation and can include:
        - 'expected_data_type': (Optional) The expected type for the top-level 'data' object.
                                E.g., `dict`, `list`, `str`.
        - 'required_fields': (Optional) A list of strings, specifying keys that must be present
                             in the input data (if 'data' is a dictionary).
        - 'field_types': (Optional) A dictionary mapping field names to their expected types.
                         Values can be single types (e.g., `int`), or a tuple of types
                         (e.g., `(str, type(None))` for an optional string).
        - 'custom_field_validators': (Optional) A dictionary mapping field names to callable
                                     validation functions. Each function takes the field's value
                                     as an argument and must return `True` for valid, `False` for invalid.

        Example Schema:
        {
            "expected_data_type": dict,
            "required_fields": ["id", "name"],
            "field_types": {
                "id": int,
                "name": str,
                "age": (int, type(None)), # Age can be an int or None
                "email": str
            },
            "custom_field_validators": {
                "age": lambda x: x is None or x >= 0, # Age must be non-negative or None
                "name": lambda x: len(x.strip()) > 0 # Name must not be empty or just whitespace
            }
        }

        Args:
            schema: A dictionary defining the validation rules.

        Raises:
            TypeError: If the provided schema is not a dictionary.
        """
        if not isinstance(schema, dict):
            raise TypeError("Schema for DataValidator must be a dictionary.")
        self._schema = schema
        logger.debug(f"[{self.node_name}] Initialized with schema: {self._schema}")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "DataValidator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Validates the input data against the configured schema.

        Args:
            data: The input data to be validated.
            context: A dictionary containing contextual information for the processing flow.
                     (Not directly used for schema validation in this implementation, but available).

        Returns:
            The original data if validation is successful.

        Raises:
            DataValidationException: If the data fails any validation rule defined in the schema.
        """
        logger.debug(f"[{self.node_name}] Starting validation for input data.")

        # 1. Overall data type check
        expected_data_type = self._schema.get("expected_data_type")
        if expected_data_type and not isinstance(data, expected_data_type):
            msg = (f"Data validation failed: Expected overall data type "
                   f"{getattr(expected_data_type, '__name__', str(expected_data_type))}, "
                   f"but received {type(data).__name__}.")
            logger.error(f"[{self.node_name}] {msg}")
            raise DataValidationException(msg)

        # Proceed with field-level validation only if data is a dictionary
        if isinstance(data, dict):
            # 2. Check for required fields
            required_fields = self._schema.get("required_fields", [])
            for field in required_fields:
                if field not in data:
                    msg = f"Data validation failed: Missing required field '{field}'."
                    logger.error(f"[{self.node_name}] {msg}")
                    raise DataValidationException(msg)

            # 3. Check field types
            field_types: Dict[str, Union[type, Tuple[type, ...]]] = self._schema.get("field_types", {})
            for field, expected_type_or_tuple in field_types.items():
                if field not in data:
                    # If field is not present, it's not a type violation unless it's also a required_field.
                    # Required fields are checked separately. If optional and not present, it's fine.
                    continue

                if not isinstance(data[field], expected_type_or_tuple):
                    expected_type_str = (
                        expected_type_or_tuple.__name__ if not isinstance(expected_type_or_tuple, tuple)
                        else ' or '.join(t.__name__ for t in expected_type_or_tuple)
                    )
                    msg = (f"Data validation failed for field '{field}': "
                           f"Expected type(s) {expected_type_str}, "
                           f"but received {type(data[field]).__name__} (value: {data[field]}).")
                    logger.error(f"[{self.node_name}] {msg}")
                    raise DataValidationException(msg)

            # 4. Apply custom field validators
            custom_field_validators: Dict[str, Callable[[Any], bool]] = self._schema.get("custom_field_validators", {})
            for field, validator_func in custom_field_validators.items():
                if field in data:
                    try:
                        if not validator_func(data[field]):
                            msg = (f"Data validation failed for field '{field}': "
                                   f"Custom validator returned False for value '{data[field]}'.")
                            logger.error(f"[{self.node_name}] {msg}")
                            raise DataValidationException(msg)
                    except Exception as e:
                        msg = (f"Data validation failed for field '{field}': "
                               f"Custom validator raised an exception for value '{data[field]}': {e}")
                        logger.error(f"[{self.node_name}] {msg}")
                        raise DataValidationException(msg) from e
        elif self._schema.get("required_fields") or self._schema.get("field_types") or self._schema.get("custom_field_validators"):
            # If schema has dict-specific rules but data is not a dict
            msg = (f"Data validation failed: Schema contains field-specific rules, "
                   f"but input data is not a dictionary (received {type(data).__name__}).")
            logger.error(f"[{self.node_name}] {msg}")
            raise DataValidationException(msg)


        logger.info(f"[{self.node_name}] Data successfully validated.")
        return data

