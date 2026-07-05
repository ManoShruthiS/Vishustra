import logging
from typing import Any, Dict, Callable, List, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class FieldNotFoundError(KeyError):
    """Raised when a specified field path is not found in the data."""
    pass

class DataValidationFailedError(ValueError):
    """Raised when data fails one or more validation rules."""
    def __init__(self, message: str, errors: List[str]):
        super().__init__(message)
        self.errors = errors

class DataValidatorNode(BaseNode):
    """
    A Vishustra processing node that validates input data against a set of predefined rules.

    This node is designed to ensure the structural integrity and content validity of
    incoming data before it proceeds to subsequent processing stages.

    Rules are defined as a list of dictionaries, where each dictionary specifies:
    - 'field': The dot-separated path to the field in the data (e.g., "user.profile.age").
    - 'validator': A callable that takes the field's value and returns True if valid, False otherwise.
    - 'message': An error message to be logged and included in the exception if validation fails.
    - 'optional': (Optional, default False) If True, the validation rule is skipped if the
                  specified field does not exist in the data. If False, a missing field
                  will result in a validation error.
    """

    def __init__(self, rules: List[Dict[str, Union[str, Callable[[Any], bool], bool]]]):
        """
        Initializes the DataValidatorNode with a list of validation rules.

        Args:
            rules: A list of rule dictionaries. Each dictionary must contain
                   'field' (str), 'validator' (Callable[[Any], bool]), and 'message' (str).
                   An optional 'optional' (bool) flag can be included.

        Raises:
            TypeError: If `rules` is not a list, or if individual rule components have incorrect types.
            ValueError: If individual rule dictionaries are malformed (e.g., missing keys, empty strings).
        """
        if not isinstance(rules, list):
            raise TypeError("Validation rules must be provided as a list.")
        for i, rule in enumerate(rules):
            if not isinstance(rule, dict):
                raise TypeError(f"Rule at index {i} must be a dictionary.")
            if not all(k in rule for k in ['field', 'validator', 'message']):
                raise ValueError(f"Rule at index {i} is missing required keys ('field', 'validator', 'message').")
            if not isinstance(rule['field'], str) or not rule['field']:
                raise ValueError(f"Rule at index {i}: 'field' must be a non-empty string.")
            if not callable(rule['validator']):
                raise TypeError(f"Rule at index {i}: 'validator' must be a callable.")
            if not isinstance(rule['message'], str) or not rule['message']:
                raise ValueError(f"Rule at index {i}: 'message' must be a non-empty string.")
            if 'optional' in rule and not isinstance(rule['optional'], bool):
                raise TypeError(f"Rule at index {i}: 'optional' flag must be a boolean.")

        self._rules = rules
        logger.debug(f"DataValidatorNode initialized with {len(self._rules)} validation rules.")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "DataValidatorNode"

    def _get_nested_value(self, data: Dict[str, Any], field_path: str) -> Any:
        """
        Retrieves a nested value from a dictionary using a dot-separated path.

        Args:
            data: The dictionary to search within.
            field_path: The dot-separated path (e.g., "user.profile.age").

        Returns:
            The value found at the specified path.

        Raises:
            FieldNotFoundError: If any part of the path is not found or if an intermediate
                                segment is not a dictionary.
        """
        keys = field_path.split('.')
        current_value = data
        for i, key in enumerate(keys):
            if not isinstance(current_value, dict):
                raise FieldNotFoundError(
                    f"Expected dictionary at path segment '{'.'.join(keys[:i])}' but found "
                    f"type '{type(current_value).__name__}' while looking for '{key}'."
                )
            if key in current_value:
                current_value = current_value[key]
            else:
                raise FieldNotFoundError(
                    f"Field path '{field_path}' not found. Missing key '{key}' at level {i+1}."
                )
        return current_value

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Validates the input data against the configured rules.

        Args:
            data: The input data to validate. Expected to be a dictionary.
            context: A dictionary of contextual information (available for node operations,
                     though not directly used by this specific validator for validation logic).

        Returns:
            The original `data` dictionary if all validations pass.

        Raises:
            TypeError: If the input `data` is not a dictionary.
            DataValidationFailedError: If one or more validation rules fail.
        """
        if not isinstance(data, dict):
            logger.error(f"[{self.node_name}] Invalid input data type. Expected 'dict', got '{type(data).__name__}'.")
            raise TypeError(f"Input data for '{self.node_name}' must be a dictionary, received '{type(data).__name__}'.")

        validation_errors: List[str] = []

        logger.info(f"[{self.node_name}] Starting data validation for incoming data.")

        for rule in self._rules:
            field_path = rule['field']
            validator_func = rule['validator']
            error_message = rule['message']
            is_optional = rule.get('optional', False)

            try:
                field_value = self._get_nested_value(data, field_path)
                if not validator_func(field_value):
                    validation_errors.append(f"Validation failed for field '{field_path}': {error_message}")
                    logger.warning(
                        f"[{self.node_name}] Rule failed for '{field_path}'. Message: '{error_message}'. "
                        f"Value: '{field_value}' (Type: {type(field_value).__name__})."
                    )
                else:
                    logger.debug(f"[{self.node_name}] Validation passed for field '{field_path}'.")
            except FieldNotFoundError as e:
                if not is_optional:
                    validation_errors.append(f"Required field missing: '{field_path}'. Error: {e}")
                    logger.warning(f"[{self.node_name}] Required field '{field_path}' not found: {e}")
                else:
                    logger.debug(f"[{self.node_name}] Optional field '{field_path}' not found, skipping validation.")
            except Exception as e:
                # Catch any unexpected errors that might occur within the validator function itself
                validation_errors.append(f"An unexpected error occurred during validation of '{field_path}': {type(e).__name__} - {e}")
                logger.error(
                    f"[{self.node_name}] Unexpected error during validation of '{field_path}': {type(e).__name__} - {e}",
                    exc_info=True
                )

        if validation_errors:
            combined_error_msg = f"Data validation failed with {len(validation_errors)} error(s)."
            logger.error(f"[{self.node_name}] {combined_error_msg} Details: {'; '.join(validation_errors)}")
            raise DataValidationFailedError(combined_error_msg, validation_errors)
        else:
            logger.info(f"[{self.node_name}] All data validation rules passed successfully.")
            return data