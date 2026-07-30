import logging
from typing import Any, Dict, List, Callable, Optional, Type

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class DataValidationException(ValueError):
    """
    Custom exception raised when data validation within DataValidatorNode fails.
    """
    pass

class DataValidatorNode(BaseNode):
    """
    A Vishustra node responsible for validating input data against a set of
    defined rules. It supports type checking, required key presence (for dicts),
    and custom callable validators.

    Validation rules can be provided at node initialization or dynamically
    via the `context` during processing.
    """

    def __init__(self, validation_schema: Optional[Dict[str, Any]] = None):
        """
        Initializes the DataValidatorNode with an optional default validation schema.

        The `validation_schema` dictionary can contain:
        - 'expected_type': Type - The expected type of the data (e.g., dict, str, int).
        - 'required_keys': List[str] - For dictionary data, a list of keys that must be present.
        - 'custom_validators': List[Callable[[Any], bool]] - A list of functions. Each function
          takes the data as input and must return True for valid, False for invalid.
          If any validator raises an exception, it's considered a failure.
        """
        self._default_validation_schema = validation_schema if validation_schema is not None else {}
        logger.debug(
            f"DataValidatorNode initialized with default schema: {list(self._default_validation_schema.keys())}"
        )

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "DataValidator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by validating it against the configured rules.
        Rules from `context.get('validation_schema')` will override or extend
        the default schema set during initialization.

        If `data` fails any validation rule, a `DataValidationException` is raised.

        Args:
            data: The input data to be validated.
            context: A dictionary containing operational context, potentially including
                     dynamic validation rules under the key 'validation_schema'.

        Returns:
            The original data if it passes all validations.

        Raises:
            DataValidationException: If the data does not conform to the validation rules.
        """
        current_validation_schema = {
            **self._default_validation_schema,
            **context.get("validation_schema", {}),
        }
        logger.info(
            f"Initiating validation for data. Rules applied: {list(current_validation_schema.keys())}"
        )

        # 1. Validate data type
        expected_type: Optional[Type] = current_validation_schema.get("expected_type")
        if expected_type is not None:
            if not isinstance(data, expected_type):
                error_msg = (
                    f"Data validation failed: Expected type `{expected_type.__name__}`, "
                    f"but received type `{type(data).__name__}`."
                )
                logger.error(error_msg)
                raise DataValidationException(error_msg)
            logger.debug(f"Type validation passed: Data is of expected type `{expected_type.__name__}`.")

        # 2. Validate required keys (if data is a dictionary)
        required_keys: Optional[List[str]] = current_validation_schema.get("required_keys")
        if required_keys is not None:
            if isinstance(data, dict):
                missing_keys = [key for key in required_keys if key not in data]
                if missing_keys:
                    error_msg = (
                        f"Data validation failed: Missing required keys in dictionary data: "
                        f"{', '.join(map(repr, missing_keys))}."
                    )
                    logger.error(error_msg)
                    raise DataValidationException(error_msg)
                logger.debug("Required keys validation passed.")
            else:
                logger.warning(
                    f"Required keys validation rules provided ({required_keys}), "
                    f"but data is not a dictionary (type: {type(data).__name__}). Skipping this validation step."
                )

        # 3. Apply custom validators
        custom_validators: Optional[List[Callable[[Any], bool]]] = current_validation_schema.get(
            "custom_validators"
        )
        if custom_validators is not None:
            if not isinstance(custom_validators, list):
                logger.warning(
                    f"Custom validators rule is not a list (received type: {type(custom_validators).__name__}). "
                    "Skipping custom validation."
                )
            else:
                for idx, validator_func in enumerate(custom_validators):
                    if not callable(validator_func):
                        logger.warning(
                            f"Custom validator at index {idx} is not a callable function. Skipping this validator."
                        )
                        continue
                    try:
                        if not validator_func(data):
                            error_msg = (
                                f"Data validation failed: Custom validator at index {idx} "
                                f"returned `False` for the data."
                            )
                            logger.error(error_msg)
                            raise DataValidationException(error_msg)
                        logger.debug(f"Custom validator at index {idx} passed.")
                    except Exception as e:
                        error_msg = (
                            f"Data validation failed: Custom validator at index {idx} "
                            f"raised an unexpected exception: {e.__class__.__name__}: {e}"
                        )
                        logger.exception(error_msg)  # Log full traceback for debugging
                        raise DataValidationException(error_msg) from e

        logger.info("Data passed all validation checks successfully.")
        return data