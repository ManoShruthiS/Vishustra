from abc import ABC, abstractmethod
import logging
import re
from typing import Any, Dict, Optional, Type

# Assuming BaseNode is available at this path as per project context
from vishustra_core.nodes.base_node import BaseNode

# Initialize logger for this module
logger = logging.getLogger(__name__)

class DataValidationException(ValueError):
    """
    Custom exception raised when data fails to meet predefined validation rules.
    This provides a specific error type for consumers to catch.
    """
    pass

class DataValidatorNode(BaseNode):
    """
    A Vishustra processing node responsible for validating incoming data against
    a set of predefined rules. This ensures data integrity and adherence to expected
    formats before further processing.

    Validation rules can specify:
    - The overall expected type of the data (e.g., dict, list, str, int).
    - A schema for dictionary data, detailing rules for individual fields:
        - Expected field type.
        - Whether a field is required.
        - Minimum and maximum values for numerical fields.
        - Regular expression patterns for string fields.
        - Element type for list fields.
    """

    def __init__(self, validation_config: Dict[str, Any]):
        """
        Initializes the DataValidatorNode with a set of validation rules.

        Args:
            validation_config (Dict[str, Any]): A dictionary defining the validation rules.
                Example structure:
                {
                    "expected_data_type": dict, # Optional: Overall data type check
                    "schema": { # Optional: Rules for dictionary fields
                        "id": {"type": int, "required": True, "min_value": 1},
                        "name": {"type": str, "required": True, "pattern": r"^[A-Za-z\s]{3,}$"},
                        "description": {"type": str, "required": False},
                        "tags": {"type": list, "element_type": str, "required": False},
                        "priority": {"type": int, "required": False, "max_value": 10}
                    }
                }
        """
        self._validation_config = validation_config
        logger.debug(f"DataValidatorNode initialized with configuration: {validation_config}")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this node."""
        return "DataValidatorNode"

    def _validate_field(self, field_name: str, value: Any, rules: Dict[str, Any], data_description: str) -> None:
        """
        Helper method to validate a single field's value against its specific rules.

        Args:
            field_name (str): The name of the field being validated.
            value (Any): The actual value of the field.
            rules (Dict[str, Any]): The validation rules for this specific field.
            data_description (str): A string describing the context of the data (e.g., "input data").

        Raises:
            DataValidationException: If the field fails any validation rule.
        """
        expected_type: Optional[Type] = rules.get("type")
        min_value: Optional[Any] = rules.get("min_value")
        max_value: Optional[Any] = rules.get("max_value")
        pattern: Optional[str] = rules.get("pattern")
        element_type: Optional[Type] = rules.get("element_type")

        # 1. Type check
        if expected_type and not isinstance(value, expected_type):
            raise DataValidationException(
                f"Field '{field_name}' in {data_description} expected type "
                f"'{expected_type.__name__}', but received '{type(value).__name__}'."
            )

        # 2. Min/Max value check (primarily for numerical types)
        if (min_value is not None) and (isinstance(value, (int, float))):
            if value < min_value:
                raise DataValidationException(
                    f"Field '{field_name}' in {data_description} value '{value}' is less than "
                    f"minimum allowed '{min_value}'."
                )
        if (max_value is not None) and (isinstance(value, (int, float))):
            if value > max_value:
                raise DataValidationException(
                    f"Field '{field_name}' in {data_description} value '{value}' is greater than "
                    f"maximum allowed '{max_value}'."
                )

        # 3. Pattern check (for string types)
        if pattern:
            if isinstance(value, str):
                if not re.fullmatch(pattern, value):
                    raise DataValidationException(
                        f"Field '{field_name}' in {data_description} string '{value}' does not "
                        f"match required pattern '{pattern}'."
                    )
            else:
                logger.warning(
                    f"Pattern rule '{pattern}' was provided for field '{field_name}' in {data_description}, "
                    f"but its value is not a string (type: {type(value).__name__}). Pattern rule ignored."
                )

        # 4. Element type check (for list types)
        if element_type:
            if isinstance(value, list):
                for i, element in enumerate(value):
                    if not isinstance(element, element_type):
                        raise DataValidationException(
                            f"Field '{field_name}' in {data_description} list element at index {i} "
                            f"expected type '{element_type.__name__}', but received '{type(element).__name__}'."
                        )
            else:
                logger.warning(
                    f"Element type rule '{element_type.__name__}' was provided for field '{field_name}' "
                    f"in {data_description}, but its value is not a list (type: {type(value).__name__}). "
                    f"Element type rule ignored."
                )

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by validating it against the configured rules.

        Args:
            data (Any): The input data to be validated.
            context (Dict[str, Any]): The current processing context, potentially containing
                                       'node_id' for logging purposes.

        Returns:
            Any: The original input data, unmodified, if all validation rules pass.

        Raises:
            DataValidationException: If the data fails any validation rule defined in
                                     `validation_config`.
        """
        node_id = context.get("node_id", self.node_name)
        logger.info(f"[{node_id}] Starting data validation process.")

        try:
            expected_data_type: Optional[Type] = self._validation_config.get("expected_data_type")
            schema: Optional[Dict[str, Any]] = self._validation_config.get("schema")

            # 1. Overall data type validation
            if expected_data_type and not isinstance(data, expected_data_type):
                raise DataValidationException(
                    f"Overall data type mismatch: Expected '{expected_data_type.__name__}', "
                    f"but received '{type(data).__name__}' for data processed by '{node_id}'."
                )

            # 2. Schema validation for dictionary data
            if schema:
                if not isinstance(data, dict):
                    # If a schema is provided, the data *must* be a dictionary.
                    raise DataValidationException(
                        f"Schema validation requested, but input data is not a dictionary. "
                        f"Received type '{type(data).__name__}' for node '{node_id}'."
                    )

                for field_name, rules in schema.items():
                    required: bool = rules.get("required", False)

                    if field_name not in data:
                        if required:
                            raise DataValidationException(
                                f"Required field '{field_name}' is missing from input data for node '{node_id}'."
                            )
                        else:
                            # If field is optional and missing, skip its validation
                            logger.debug(
                                f"[{node_id}] Optional field '{field_name}' is missing. Skipping specific validation."
                            )
                            continue # Move to the next field in the schema

                    # Field exists (and is possibly required), proceed with its specific rules
                    self._validate_field(field_name, data[field_name], rules, f"data for node '{node_id}'")

            logger.info(f"[{node_id}] Data validation completed successfully.")
            return data

        except DataValidationException as e:
            # Catch our custom validation exception, log it, and re-raise
            logger.error(f"[{node_id}] Data validation failed: {e}")
            raise # Re-raise the specific validation error
        except Exception as e:
            # Catch any other unexpected errors during the validation process
            logger.critical(
                f"[{node_id}] An unexpected error occurred during data validation: {e}",
                exc_info=True # Log traceback for critical errors
            )
            # Wrap the unexpected error in our custom exception for consistent error handling downstream
            raise DataValidationException(f"Unexpected error during validation for node '{node_id}': {e}") from e
