import logging
from typing import Any, Dict, List, Callable, Union, Tuple, Type

# BaseNode is expected to be available from the vishustra_core package
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

# --- Custom Exceptions for Vishustra Data Validation ---
class VishustraValidationError(ValueError):
    """Base exception for Vishustra data validation errors."""
    pass

class InvalidDataFormatError(VishustraValidationError):
    """Raised when data format is unexpected (e.g., not a dict when expected)."""
    pass

class MissingRequiredKeysError(VishustraValidationError):
    """Raised when required keys are missing from the data."""
    pass

class InvalidDataTypeError(VishustraValidationError):
    """Raised when a data field has an incorrect type."""
    pass

class CustomValidationError(VishustraValidationError):
    """Raised when a custom validation function fails."""
    pass

# --- DataValidatorNode Implementation ---
class DataValidatorNode(BaseNode):
    """
    A Vishustra node designed for robust data validation based on
    pre-defined rules. It supports checking for required keys, verifying data types
    against a schema, and applying custom validation functions.

    Validation rules are configured during node initialization via a dictionary.

    Configuration options for `validation_rules` dictionary:
    - `required_keys`: Optional. A `List[str]` specifying keys that must be present
                       in the input dictionary.
    - `schema`: Optional. A `Dict[str, Union[Type, Tuple[Type, ...]]]` where keys are
                data fields and values are their expected types. `isinstance()` is
                used for checking, allowing tuples for multiple acceptable types
                (e.g., `(int, float)`) or optional types (`(str, type(None))`).
    - `custom_validators`: Optional. A `List[Callable[[Any], bool]]` of functions.
                           Each function takes the entire `data` dictionary as input
                           and must return `True` for valid data or `False` for invalid.
                           If a custom validator raises an exception, it will be caught
                           and re-raised as a `CustomValidationError`.
    """

    def __init__(self, validation_rules: Dict[str, Any]):
        """
        Initializes the DataValidatorNode with specific validation rules.

        Args:
            validation_rules: A dictionary containing rules for validation.
                              See the class docstring for configuration details.
        """
        self._validation_rules = validation_rules
        logger.debug(f"[{self.node_name}] Initialized with rules: {validation_rules}")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "DataValidator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by applying all configured validation rules.

        Args:
            data: The input data to be validated. Expected typically to be a dictionary,
                  depending on the validation rules configured.
            context: A dictionary containing contextual information relevant to the
                     current orchestration run. (Currently not explicitly used by
                     this node for validation logic, but passed through as per
                     BaseNode signature).

        Returns:
            The original `data` object if all validation checks pass successfully.

        Raises:
            InvalidDataFormatError: If `data` is not a dictionary when `required_keys`
                                    or `schema` rules are configured.
            MissingRequiredKeysError: If any keys specified in `required_keys` are
                                      not found in `data`.
            InvalidDataTypeError: If a field's type in `data` does not match the
                                  expected type defined in the `schema`.
            CustomValidationError: If any function within `custom_validators` returns
                                   `False` or raises an exception during execution.
            TypeError: If a configured custom validator is not a callable function.
        """
        logger.info(f"[{self.node_name}] Starting data validation for incoming data.")

        # If no validation rules are configured, log a warning and pass data through.
        if not self._validation_rules:
            logger.warning(f"[{self.node_name}] No validation rules configured. Data will pass through unchecked.")
            return data

        # Check if data is a dictionary if dictionary-specific rules are present
        if any(key in self._validation_rules for key in ['required_keys', 'schema']):
            if not isinstance(data, dict):
                logger.error(f"[{self.node_name}] Validation failed: Expected data to be a dictionary, but received type: {type(data)}.")
                raise InvalidDataFormatError(
                    f"[{self.node_name}] Data must be a dictionary for the configured validation rules."
                )

        # 1. Validate required keys
        required_keys: List[str] = self._validation_rules.get('required_keys', [])
        if required_keys:
            missing_keys = [key for key in required_keys if key not in data]
            if missing_keys:
                logger.error(f"[{self.node_name}] Validation failed: Missing required keys: {missing_keys}.")
                raise MissingRequiredKeysError(
                    f"[{self.node_name}] Data missing required keys: {missing_keys}"
                )

        # 2. Validate schema (type checking)
        schema: Dict[str, Union[Type, Tuple[Type, ...]]] = self._validation_rules.get('schema', {})
        if schema:
            for key, expected_type in schema.items():
                if key in data:
                    if not isinstance(data[key], expected_type):
                        logger.error(
                            f"[{self.node_name}] Validation failed: Key '{key}' expected type {expected_type}, "
                            f"but got {type(data[key])} with value '{data[key]}'."
                        )
                        raise InvalidDataTypeError(
                            f"[{self.node_name}] Key '{key}' has incorrect type. Expected {expected_type}, "
                            f"got {type(data[key])}."
                        )

        # 3. Validate with custom functions
        custom_validators: List[Callable[[Any], bool]] = self._validation_rules.get('custom_validators', [])
        for validator_func in custom_validators:
            if not callable(validator_func):
                logger.critical(
                    f"[{self.node_name}] Configuration error: Custom validator '{validator_func}' "
                    "is not a callable function."
                )
                raise TypeError(
                    f"[{self.node_name}] Custom validator provided '{validator_func}' is not a callable function."
                )
            try:
                if not validator_func(data):
                    logger.error(f"[{self.node_name}] Validation failed by custom rule: '{validator_func.__name__}'.")
                    raise CustomValidationError(
                        f"[{self.node_name}] Data failed custom validation rule: '{validator_func.__name__}'."
                    )
            except Exception as e:
                # Log the full traceback for unexpected errors in custom validators
                logger.exception(
                    f"[{self.node_name}] An unexpected error occurred while executing custom validator "
                    f"'{validator_func.__name__}'."
                )
                raise CustomValidationError(
                    f"[{self.node_name}] Error during custom validation with '{validator_func.__name__}': {e}"
                ) from e

        logger.info(f"[{self.node_name}] Data successfully validated.")
        return data