
import logging
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class DataValidationException(ValueError):
    """
    Custom exception raised when data fails validation rules defined in the context.
    """
    pass


class DataValidatorNode(BaseNode):
    """
    A processing node that validates input data against a set of rules
    provided in the execution context.

    The node expects validation rules to be present in the `context` dictionary
    under the key 'validation_rules'. If no rules are found, the data
    is considered valid and passed through.

    Supported validation rules (as a dict under 'validation_rules'):
    - 'required_keys': A list of strings, enforcing the presence of keys in dict data.
    - 'min_length': A dict where keys are field names (str) and values are
                    minimum required lengths (int) for string fields.
    - 'allowed_values': A dict where keys are field names (str) and values are
                        lists of allowed values for that field.
    - 'min_string_length': An integer representing the minimum length for the
                           entire input if `data` is a string.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "DataValidator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, validating it against rules specified in the context.

        Args:
            data: The input data to be validated. This can be any type.
            context: A dictionary containing execution context, including
                     'validation_rules' for this node.

        Returns:
            The original data if it passes all validation checks.

        Raises:
            DataValidationException: If the data fails any of the specified
                                     validation rules.
            TypeError: If the validation rules themselves are malformed.
        """
        logger.debug(f"[{self.node_name}] Starting data validation for incoming data.")

        validation_rules: Dict[str, Any] = context.get('validation_rules', {})

        if not validation_rules:
            logger.info(f"[{self.node_name}] No validation rules found in context. Data will pass without explicit checks.")
            return data

        try:
            # --- Rule: Validate dictionary data ---
            if isinstance(data, dict):
                # Required keys check
                required_keys: List[str] = validation_rules.get('required_keys', [])
                if not isinstance(required_keys, list):
                    raise TypeError(f"Validation rule 'required_keys' must be a list, got {type(required_keys).__name__}.")
                for key in required_keys:
                    if not isinstance(key, str):
                        raise TypeError(f"Elements in 'required_keys' must be strings, got {type(key).__name__}.")
                    if key not in data:
                        error_msg = f"Validation failed: Missing required key '{key}' in data."
                        logger.error(f"[{self.node_name}] {error_msg}")
                        raise DataValidationException(error_msg)

                # Minimum length for string fields check
                min_length_rules: Dict[str, int] = validation_rules.get('min_length', {})
                if not isinstance(min_length_rules, dict):
                    raise TypeError(f"Validation rule 'min_length' must be a dictionary, got {type(min_length_rules).__name__}.")
                for key, min_len in min_length_rules.items():
                    if not isinstance(key, str) or not isinstance(min_len, int):
                        raise TypeError(f"Keys in 'min_length' must be strings and values integers, got key type {type(key).__name__}, value type {type(min_len).__name__}.")
                    if key in data and isinstance(data[key], str) and len(data[key]) < min_len:
                        error_msg = f"Validation failed: Field '{key}' has length {len(data[key])}, which is less than required minimum {min_len}."
                        logger.error(f"[{self.node_name}] {error_msg}")
                        raise DataValidationException(error_msg)

                # Allowed values for fields check
                allowed_values_rules: Dict[str, List[Any]] = validation_rules.get('allowed_values', {})
                if not isinstance(allowed_values_rules, dict):
                    raise TypeError(f"Validation rule 'allowed_values' must be a dictionary, got {type(allowed_values_rules).__name__}.")
                for key, allowed_list in allowed_values_rules.items():
                    if not isinstance(key, str) or not isinstance(allowed_list, list):
                        raise TypeError(f"Keys in 'allowed_values' must be strings and values lists, got key type {type(key).__name__}, value type {type(allowed_list).__name__}.")
                    if key in data and data[key] not in allowed_list:
                        error_msg = f"Validation failed: Field '{key}' has value '{data[key]}', which is not in allowed values {allowed_list}."
                        logger.error(f"[{self.node_name}] {error_msg}")
                        raise DataValidationException(error_msg)

            # --- Rule: Validate string data ---
            elif isinstance(data, str):
                min_str_length: int = validation_rules.get('min_string_length', 0)
                if not isinstance(min_str_length, int):
                    raise TypeError(f"Validation rule 'min_string_length' must be an integer, got {type(min_str_length).__name__}.")
                if len(data) < min_str_length:
                    error_msg = f"Validation failed: Input string has length {len(data)}, which is less than required minimum {min_str_length}."
                    logger.error(f"[{self.node_name}] {error_msg}")
                    raise DataValidationException(error_msg)

            else:
                # Log that specific validation isn't implemented for this type,
                # but doesn't necessarily mean it's invalid if no general rules apply.
                logger.debug(f"[{self.node_name}] No specific validation rules applied for data type: {type(data).__name__}.")

        except (KeyError, TypeError) as e:
            error_msg = f"Malformed validation rules provided in context: {e}"
            logger.error(f"[{self.node_name}] {error_msg}")
            raise DataValidationException(error_msg) from e
        except Exception as e:
            error_msg = f"An unexpected error occurred during validation: {type(e).__name__}: {e}"
            logger.critical(f"[{self.node_name}] {error_msg}")
            raise DataValidationException(error_msg) from e

        logger.info(f"[{self.node_name}] Data successfully validated and passed through.")
        return data

