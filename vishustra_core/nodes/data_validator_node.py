import logging
from typing import Any, Dict, List, Type, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ValidationError(ValueError):
    """Custom exception raised when data fails validation against a schema."""
    pass

class DataValidator(BaseNode):
    """
    A Vishustra processing node that validates input data against a predefined schema.

    This node enforces data integrity and structure, raising a ValidationError if
    the input data does not conform to the specified rules. It supports checks for:
    - Overall data type (e.g., dict, list, str, int).
    - Presence of required keys (for dictionaries).
    - Types of specific keys within a dictionary.
    - Length constraints (minimum and maximum) for strings, lists, and dictionaries.

    Configuration Schema Example:
    A schema dictionary passed during initialization configures the validation rules.
    {
        "expected_type": dict,  # Required: The overall expected type of the 'data'
        "required_keys": ["id", "name"], # Optional: List of keys that must exist if data is a dict
        "key_types": { # Optional: Dictionary mapping keys to their expected types if data is a dict
            "id": int,
            "name": str,
            "metadata": dict
        },
        "min_length": 1, # Optional: Minimum length for string, list, or dict data
        "max_length": 100 # Optional: Maximum length for string, list, or dict data
    }
    """

    def __init__(self, schema: Dict[str, Any]):
        """
        Initializes the DataValidator node with a validation schema.

        Args:
            schema: A dictionary defining the validation rules.
                    It must contain at least 'expected_type'.
                    Example:
                    {
                        "expected_type": dict,
                        "required_keys": ["field1"],
                        "key_types": {"field1": str},
                        "min_length": 1,
                        "max_length": 50
                    }
        Raises:
            ValueError: If the provided schema is invalid or missing 'expected_type'.
        """
        if not isinstance(schema, dict):
            raise ValueError("Validation schema must be a dictionary.")
        if "expected_type" not in schema:
            raise ValueError("Validation schema must define an 'expected_type'.")
        if not isinstance(schema["expected_type"], type):
            raise ValueError("'expected_type' in schema must be a type (e.g., dict, str, int).")

        self._schema = schema
        logger.info(f"DataValidator node initialized with schema: {schema}")

    @property
    def node_name(self) -> str:
        """Returns the programmatic name of the node."""
        return "DataValidator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Validates the input data against the configured schema.

        Args:
            data: The data to be validated.
            context: A dictionary containing context-specific information,
                     such as a 'node_id' for logging.

        Returns:
            The original data if validation is successful.

        Raises:
            ValidationError: If the data does not conform to the schema.
            RuntimeError: If an unexpected internal error occurs during validation.
        """
        node_id = context.get('node_id', self.node_name)
        logger.debug(f"[{node_id}] Starting data validation for incoming data.")

        try:
            # 1. Validate overall type
            expected_type = self._schema["expected_type"] # Guaranteed to exist by __init__
            if not isinstance(data, expected_type):
                raise ValidationError(
                    f"[{node_id}] Data type mismatch. Expected '{expected_type.__name__}', "
                    f"got '{type(data).__name__}'."
                )

            # 2. Validate length (for string, list, dict)
            if isinstance(data, (str, list, dict)):
                min_length = self._schema.get("min_length")
                max_length = self._schema.get("max_length")
                data_length = len(data)

                if min_length is not None and data_length < min_length:
                    raise ValidationError(
                        f"[{node_id}] Data length ({data_length}) is less than minimum required length ({min_length})."
                    )
                if max_length is not None and data_length > max_length:
                    raise ValidationError(
                        f"[{node_id}] Data length ({data_length}) exceeds maximum allowed length ({max_length})."
                    )

            # 3. Validate dictionary-specific rules
            # Apply these only if the data is expected to be or is a dictionary.
            if expected_type is dict or (expected_type is None and isinstance(data, dict)):
                if not isinstance(data, dict):
                    # This case should ideally be caught by expected_type check above,
                    # but provides a fallback for robustness if schema implies dict checks on Any type.
                    raise ValidationError(
                        f"[{node_id}] Expected dictionary for key/type validation, but received '{type(data).__name__}'."
                    )

                # Required keys
                required_keys = self._schema.get("required_keys", [])
                for key in required_keys:
                    if key not in data:
                        raise ValidationError(f"[{node_id}] Required key '{key}' is missing from data.")

                # Key types
                key_types = self._schema.get("key_types", {})
                for key, expected_key_type in key_types.items():
                    if key in data: # Only validate type if the key is present
                        if not isinstance(data[key], expected_key_type):
                            raise ValidationError(
                                f"[{node_id}] Type mismatch for key '{key}'. Expected '{expected_key_type.__name__}', "
                                f"got '{type(data[key]).__name__}'."
                            )

            logger.info(f"[{node_id}] Data successfully validated against schema.")
            return data

        except ValidationError as e:
            logger.error(f"[{node_id}] Data validation failed: {e}")
            raise # Re-raise the validation error as it's an expected failure type
        except Exception as e:
            logger.critical(f"[{node_id}] An unexpected error occurred during validation: {e}", exc_info=True)
            # Wrap unexpected errors in a RuntimeError for consistency in higher-level error handling
            raise RuntimeError(f"[{node_id}] Failed to process data due to an internal error.") from e