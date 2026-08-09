from vishustra_core.nodes.base_node import BaseNode
from typing import Any, Dict, List, Type, Callable
import logging

logger = logging.getLogger(__name__)

class DataValidationError(ValueError):
    """Custom exception for data validation failures within the Vishustra framework."""
    pass

class DataValidatorNode(BaseNode):
    """
    A processing node designed to validate input data against a defined schema
    or a set of rules provided through the execution context.

    This node plays a crucial role in ensuring data integrity and consistency,
    preventing malformed or invalid data from propagating to subsequent processing
    stages in an orchestration flow.
    """

    @property
    def node_name(self) -> str:
        """Returns the programmatic name of this data validator node."""
        return "DataValidatorNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Validates the input `data` based on a `validation_config` retrieved from the `context`.

        The `validation_config` in the context dictionary defines the rules,
        which can include required keys, type checks, value constraints, and custom
        validation functions.

        Expected structure of `context["validation_config"]`:
        ```python
        {
            "requires_dict_input": bool,  # If True, `data` must be a dictionary. Defaults to False.
            "required_keys": List[str],   # List of keys that must be present if `data` is a dict.
            "type_checks": Dict[str, Type], # Mapping of key to expected Python type if `data` is a dict.
            "value_constraints": Dict[str, Dict[str, Any]], # Mapping of key to min/max/length constraints.
                                                             # Example: {"age": {"min": 18, "max": 100},
                                                             #           "text": {"min_length": 1, "max_length": 500}}
            "custom_validators": List[Callable[[Any], None]], # List of callables, each taking the entire `data` object
                                                               # and raising a DataValidationError on failure.
        }
        ```

        If no `validation_config` is found in the context, a warning is logged, and
        the data is passed through without modification or validation.
        If validation fails according to the specified rules, a `DataValidationError`
        is raised.

        Args:
            data: The input data payload to be validated. This can be any Python type.
            context: A dictionary containing operational context, crucially including
                     the `validation_config` under the key "validation_config".

        Returns:
            The original `data` payload if all validation checks pass.

        Raises:
            DataValidationError: If the `data` does not conform to the rules
                                 defined in the `validation_config`.
        """
        validation_config: Dict[str, Any] = context.get("validation_config", {})

        if not validation_config:
            logger.warning(
                "DataValidatorNode received no 'validation_config' in context. "
                "Data will pass through without validation. Consider configuring validation rules."
            )
            return data

        logger.debug(f"Initiating data validation for '{self.node_name}' with config: {validation_config}")

        try:
            # Check for dictionary input requirement
            if validation_config.get("requires_dict_input", False) and not isinstance(data, Dict):
                raise DataValidationError(f"Input data must be a dictionary, but received type: {type(data).__name__}.")

            # Apply dictionary-specific validations only if data is a dictionary
            if isinstance(data, Dict):
                # 1. Required Keys Check
                required_keys: List[str] = validation_config.get("required_keys", [])
                if required_keys:
                    missing_keys = [key for key in required_keys if key not in data]
                    if missing_keys:
                        raise DataValidationError(f"Missing required keys: {', '.join(missing_keys)}.")
                    logger.debug(f"Required keys check passed.")

                # 2. Type Checks
                type_checks: Dict[str, Type] = validation_config.get("type_checks", {})
                if type_checks:
                    for key, expected_type in type_checks.items():
                        if key in data and not isinstance(data[key], expected_type):
                            raise DataValidationError(
                                f"Key '{key}' has incorrect type. Expected {expected_type.__name__}, "
                                f"but got {type(data[key]).__name__} (value: {data[key]!r})."
                            )
                    logger.debug(f"Type checks passed.")

                # 3. Value Constraints
                value_constraints: Dict[str, Dict[str, Any]] = validation_config.get("value_constraints", {})
                if value_constraints:
                    for key, constraints in value_constraints.items():
                        if key in data:
                            value = data[key]
                            # Numeric constraints
                            if isinstance(value, (int, float)):
                                if "min" in constraints and value < constraints["min"]:
                                    raise DataValidationError(
                                        f"Value for '{key}' ({value}) is less than minimum allowed ({constraints['min']})."
                                    )
                                if "max" in constraints and value > constraints["max"]:
                                    raise DataValidationError(
                                        f"Value for '{key}' ({value}) is greater than maximum allowed ({constraints['max']})."
                                    )
                            # Length constraints (for strings, lists, dicts)
                            if isinstance(value, (str, list, dict)):
                                if "min_length" in constraints and len(value) < constraints["min_length"]:
                                    raise DataValidationError(
                                        f"Length of '{key}' ({len(value)}) is less than minimum allowed ({constraints['min_length']})."
                                    )
                                if "max_length" in constraints and len(value) > constraints["max_length"]:
                                    raise DataValidationError(
                                        f"Length of '{key}' ({len(value)}) is greater than maximum allowed ({constraints['max_length']})."
                                    )
                    logger.debug(f"Value constraints checks passed.")

            # 4. Custom Validators (apply to the entire data object, regardless of type)
            custom_validators: List[Callable[[Any], None]] = validation_config.get("custom_validators", [])
            if custom_validators:
                for idx, validator in enumerate(custom_validators):
                    logger.debug(f"Applying custom validator {idx+1} for data type {type(data).__name__}.")
                    if not callable(validator):
                         raise TypeError(f"Custom validator at index {idx} is not a callable object.")
                    validator(data) # Assumes validator raises DataValidationError on failure
                logger.debug(f"Custom validators passed.")

            logger.info(f"Data successfully validated by '{self.node_name}'.")
            return data

        except DataValidationError as e:
            logger.error(
                f"Data validation failed for '{self.node_name}'. Reason: {e}. "
                f"Input data sample: {str(data)[:500]}{'...' if len(str(data)) > 500 else ''}"
            )
            raise  # Re-raise the specific validation error for upstream handling
        except Exception as e:
            logger.critical(
                f"An unexpected internal error occurred during validation in '{self.node_name}'. "
                f"Error: {type(e).__name__}: {e}",
                exc_info=True # Log stack trace for unexpected errors
            )
            # Wrap unexpected errors in a DataValidationError for consistency in error propagation
            raise DataValidationError(f"Unexpected internal validation error: {e}") from e