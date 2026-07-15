import logging
from typing import Any, Dict, List, Optional, Union, Callable

try:
    from vishustra_core.nodes.base_node import BaseNode
except ImportError:
    # Fallback for local development/testing outside the full package structure
    from abc import ABC, abstractmethod

    class BaseNode(ABC):
        """
        Base class for all Vishustra processing nodes.
        Each node must implement the process method.
        """
        
        @abstractmethod
        def process(self, data: Any, context: Dict[str, Any]) -> Any:
            """
            Processes the input data and returns the result.
            """
            pass
            
        @property
        @abstractmethod
        def node_name(self) -> str:
            """Returns the name of the node."""
            pass

logger = logging.getLogger(__name__)

class DataValidationError(ValueError):
    """Custom exception raised when data fails validation rules."""
    pass

class DataValidatorNode(BaseNode):
    """
    A Vishustra node dedicated to validating incoming data against a predefined schema.

    This node ensures data integrity and consistency by checking for required fields,
    enforcing data types, and applying custom validation logic to field values.
    """

    def __init__(
        self,
        schema: Dict[str, Union[type, Callable[[Any], bool]]],
        required_fields: Optional[List[str]] = None,
        allow_extra_fields: bool = False
    ):
        """
        Initializes the DataValidatorNode with specific validation rules.

        Args:
            schema: A dictionary where keys are expected field names and values are
                    the validation rules for that field. Values can be:
                    - A Python type (e.g., `str`, `int`) to check the field's data type.
                    - A `Callable` (function or lambda) that takes the field value
                      as an argument and must return `True` for valid, `False` for invalid,
                      or raise a `DataValidationError` for more specific failure messages.
            required_fields: An optional list of field names that absolutely *must* be
                             present in the input data. If `None`, all fields specified
                             in the `schema` are implicitly considered required. If an
                             empty list `[]`, no fields are enforced as required by default,
                             relying solely on schema validation for presence.
            allow_extra_fields: If `True`, fields present in the input data but not
                                explicitly defined in the `schema` will be ignored.
                                If `False`, the presence of any undeclared field will
                                raise a `DataValidationError`.
        """
        if not isinstance(schema, dict):
            raise TypeError("Schema must be a dictionary.")

        self._schema = schema
        # If required_fields is None, default to all keys in the schema.
        # Otherwise, use the provided list (which can be empty).
        self._required_fields = required_fields if required_fields is not None else list(schema.keys())
        self._allow_extra_fields = allow_extra_fields

        logger.debug(
            f"DataValidatorNode initialized with schema keys: {list(self._schema.keys())}, "
            f"required_fields: {self._required_fields}, "
            f"allow_extra_fields: {self._allow_extra_fields}"
        )

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "DataValidator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by validating it against the configured schema.

        Args:
            data: The input data to be validated. Expected to be a dictionary.
            context: A dictionary containing contextual information for the node.
                     While this implementation primarily uses the init-time schema,
                     dynamic validation parameters could be passed here in future
                     iterations.

        Returns:
            The original data if it passes all validation rules. The data is not
            modified by this node; it serves purely as a gatekeeper.

        Raises:
            TypeError: If the input `data` is not a dictionary.
            DataValidationError: If the data fails any configured validation rule.
            ValueError: If the node's schema configuration itself is invalid (e.g., bad rule type).
        """
        if not isinstance(data, dict):
            error_msg = (f"DataValidatorNode expects dictionary input, but received "
                         f"'{type(data).__name__}'.")
            logger.error(error_msg)
            raise TypeError(error_msg)

        # Create a copy to allow safe modifications in future if required,
        # but for validation, it acts on the original logic.
        processed_data = data.copy()

        # 1. Check for missing required fields
        for field in self._required_fields:
            if field not in processed_data:
                error_msg = f"Required field '{field}' is missing from the data."
                logger.warning(error_msg)
                raise DataValidationError(error_msg)

        # 2. Check for extra fields if not allowed
        if not self._allow_extra_fields:
            for field in processed_data:
                if field not in self._schema:
                    error_msg = (f"Unauthorized field '{field}' found in the data. "
                                 f"Set 'allow_extra_fields=True' or add '{field}' to schema.")
                    logger.warning(error_msg)
                    raise DataValidationError(error_msg)

        # 3. Validate existing fields against the schema rules
        for field, rules in self._schema.items():
            if field in processed_data: # Only validate fields that are actually present
                value = processed_data[field]
                try:
                    if isinstance(rules, type):
                        # Type validation
                        if not isinstance(value, rules):
                            error_msg = (f"Field '{field}' expected type '{rules.__name__}', "
                                         f"but received '{type(value).__name__}' with value '{value}'.")
                            logger.warning(error_msg)
                            raise DataValidationError(error_msg)
                    elif callable(rules):
                        # Custom callable validation
                        validation_result = rules(value)
                        if isinstance(validation_result, bool):
                            if not validation_result:
                                error_msg = f"Field '{field}' failed custom validation for value '{value}'."
                                logger.warning(error_msg)
                                raise DataValidationError(error_msg)
                        else:
                            # Enforce strictness: custom validators must return boolean or raise error
                            error_msg = (f"Custom validator for field '{field}' returned a non-boolean value "
                                         f"({type(validation_result).__name__}). "
                                         "Custom validators must return True/False or raise DataValidationError.")
                            logger.error(error_msg)
                            raise ValueError(error_msg) # Node misconfiguration error
                    else:
                        error_msg = (f"Invalid rule type for field '{field}'. Expected Python type or a callable, "
                                     f"but got '{type(rules).__name__}'.")
                        logger.error(error_msg)
                        raise ValueError(error_msg) # Node misconfiguration error
                except DataValidationError as e:
                    # Re-raise explicit DataValidationError for consistency
                    raise e
                except Exception as e:
                    # Catch any other unexpected exceptions during custom callable execution
                    error_msg = (f"An unexpected error occurred during validation of field '{field}' "
                                 f"with value '{value}': {e}")
                    logger.error(error_msg, exc_info=True)
                    raise DataValidationError(error_msg) from e

        logger.info("Data validated successfully.")
        return data # Return the original, valid data