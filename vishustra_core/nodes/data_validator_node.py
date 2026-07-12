import logging
from typing import Any, Dict, Callable, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class DataValidationError(ValueError):
    """Custom exception raised when data fails validation against the defined schema."""
    pass

class DataValidatorNode(BaseNode):
    """
    A processing node designed to validate input data against a defined schema.

    This node ensures that incoming data conforms to expected types, structures,
    and conditions before it is processed by subsequent nodes in the orchestration
    pipeline. Validation rules are specified during the node's initialization.
    """

    def __init__(self, validation_schema: Dict[str, Union[type, Callable[[Any], bool], Dict[str, Any]]]):
        """
        Initializes the DataValidatorNode with a set of validation rules.

        The `validation_schema` dictionary defines the expected structure and
        constraints for the input data:
        - Keys: Represent the expected field names in the input data.
        - Values: Can be one of the following:
            - A Python type (e.g., `str`, `int`, `list`, `dict`) to check `isinstance`.
            - A callable `(value) -> bool` function for custom value validation.
              It should return `True` for valid values and `False` otherwise.
            - Another dictionary, representing a nested schema for validating
              sub-dictionaries within the data.

        Args:
            validation_schema: A dictionary specifying the validation rules.

        Raises:
            TypeError: If `validation_schema` is not a dictionary.
        """
        if not isinstance(validation_schema, dict):
            logger.error("DataValidatorNode received a non-dictionary validation_schema.")
            raise TypeError("validation_schema must be a dictionary.")
        self._validation_schema = validation_schema
        logger.debug(f"DataValidatorNode initialized with schema: {self._validation_schema}")

    @property
    def node_name(self) -> str:
        """Returns the programmatic name of this node."""
        return "DataValidator"

    def _validate_field(self, field_name: str, value: Any, rule: Union[type, Callable[[Any], bool], Dict[str, Any]]) -> None:
        """
        Internal helper method to validate a single data field against its specified rule.

        Args:
            field_name: The name of the field being validated.
            value: The actual value of the field from the input data.
            rule: The validation rule (type, callable, or nested schema).

        Raises:
            DataValidationError: If the field's value does not pass validation.
            TypeError: If an invalid validation rule type is provided.
        """
        if isinstance(rule, type):
            if not isinstance(value, rule):
                raise DataValidationError(
                    f"Field '{field_name}' validation failed: Expected type '{rule.__name__}', got '{type(value).__name__}'."
                )
        elif isinstance(rule, dict):
            # Handle nested schema validation
            if not isinstance(value, dict):
                raise DataValidationError(
                    f"Field '{field_name}' validation failed: Expected a dictionary for nested validation, got '{type(value).__name__}'."
                )
            self._validate_data_against_schema(value, rule)
        elif callable(rule):
            try:
                if not rule(value):
                    raise DataValidationError(
                        f"Field '{field_name}' validation failed: Custom rule returned False for value '{value}'."
                    )
            except Exception as e:
                logger.error(
                    f"Custom validation rule for field '{field_name}' raised an exception: {e}",
                    exc_info=True
                )
                raise DataValidationError(
                    f"Field '{field_name}' validation failed: Custom rule execution error."
                ) from e
        else:
            logger.error(
                f"Invalid validation rule type for field '{field_name}': {type(rule).__name__}. "
                "Rule must be a type, callable, or dictionary."
            )
            raise TypeError(
                f"Invalid validation rule for field '{field_name}': Rule must be a type, callable, or dict."
            )

    def _validate_data_against_schema(self, data: Dict[str, Any], schema: Dict[str, Any]) -> None:
        """
        Recursively validates a dictionary of data against a given schema.

        Args:
            data: The dictionary data to validate.
            schema: The schema to validate against.

        Raises:
            DataValidationError: If any part of the data fails validation.
        """
        for field_name, rule in schema.items():
            if field_name not in data:
                raise DataValidationError(f"Required field '{field_name}' is missing from data.")

            value = data[field_name]
            self._validate_field(field_name, value, rule)

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Executes the data validation process.

        The input `data` is checked against the `validation_schema` configured
        during the node's initialization. If all validations pass, the original
        data is returned.

        Args:
            data: The input data to be validated. Expected to be a dictionary.
            context: A dictionary providing contextual information for the node's
                     operation (not directly used for validation rules in this
                     implementation, but available for broader context).

        Returns:
            The original `data` if it successfully passes all validation checks.

        Raises:
            TypeError: If the input `data` is not a dictionary.
            DataValidationError: If the input `data` does not conform to the
                                 validation schema.
            RuntimeError: For any unexpected errors during the validation process.
        """
        logger.info(f"[{self.node_name}] Starting data validation for incoming data.")

        if not isinstance(data, dict):
            logger.error(
                f"[{self.node_name}] Input data is not a dictionary (type: {type(data).__name__}). "
                "This node requires dictionary input for schema validation."
            )
            raise TypeError(
                f"[{self.node_name}] Input data must be a dictionary for schema validation. "
                f"Received type: {type(data).__name__}."
            )

        try:
            self._validate_data_against_schema(data, self._validation_schema)
            logger.info(f"[{self.node_name}] Data successfully validated against schema.")
            return data
        except DataValidationError as e:
            logger.warning(f"[{self.node_name}] Data validation failed: {e}")
            raise e
        except Exception as e:
            # Catch any unexpected errors during validation that weren't caught by specific DataValidationError
            logger.exception(f"[{self.node_name}] An unexpected error occurred during data validation: {e}")
            raise RuntimeError(f"[{self.node_name}] Unexpected error during validation process.") from e