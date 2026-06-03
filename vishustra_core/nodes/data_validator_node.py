import logging
from typing import Any, Dict, Optional, Callable, List, Type, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class DataValidationError(ValueError):
    """Custom exception raised when data fails validation checks within DataValidatorNode."""
    pass


class DataValidatorNode(BaseNode):
    """
    A Vishustra processing node that validates input data against a predefined or dynamic schema.

    This node is designed to ensure that incoming data adheres to specified requirements,
    such as the presence of required fields, correct data types, value ranges,
    allowed values, and optional custom validation logic. If data fails validation,
    it raises a `DataValidationError`, halting further processing in the flow
    and indicating a data quality issue.
    """

    def __init__(self, validation_schema: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        Initializes the DataValidatorNode with a static validation schema.

        The `validation_schema` is a dictionary where each key represents a field name
        in the input data, and its corresponding value is another dictionary
        defining the validation rules for that field.

        Example schema structure for a field's rules:
        ```python
        {
            "field_name": {
                "type": Type,                 # e.g., str, int, float, bool, list, dict
                "required": bool,             # True if the field must be present
                "min_length": int,            # For strings/lists/dicts, minimum length/size
                "max_length": int,            # For strings/lists/dicts, maximum length/size
                "min_value": Union[int, float], # For numeric types, minimum value
                "max_value": Union[int, float], # For numeric types, maximum value
                "allowed_values": List[Any],  # For any type, value must be one of these
                "custom_validator": Callable[[Any], bool], # A function(value) -> bool
            }
        }
        ```

        Rules are applied in a specific order: `required`, `type`, `min/max_length`,
        `min/max_value`, `allowed_values`, `custom_validator`. Validation for a
        field stops at the first critical failure (e.g., missing required field,
        type mismatch) to avoid subsequent checks on invalid data.

        :param validation_schema: An optional dictionary defining the initial validation rules.
                                  If None, an empty schema is used. This schema can be
                                  overridden or extended by rules provided in the `context`
                                  during the `process` call.
        """
        self._validation_schema = validation_schema if validation_schema is not None else {}
        logger.debug(f"DataValidatorNode initialized with schema: {self._validation_schema}")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "DataValidator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes and validates the input data against the configured schema.

        The validation schema used for a particular `process` call is determined
        by first taking the schema defined during node initialization and then
        merging any additional rules provided in the `context` dictionary under
        the key `'validation_schema'`. Context-provided rules will override
        rules for the same field names defined during initialization.

        If no `validation_schema` (neither initialized nor in context) is present,
        the data is considered valid by default, and no checks are performed.
        Otherwise, if data fails any validation rule, a `DataValidationError` is raised.

        :param data: The input data to be validated. For structured validation
                     with a schema, this is expected to be a dictionary.
        :param context: A dictionary containing contextual information. May include
                        a `'validation_schema'` key to dynamically override or
                        extend the node's configured schema for this operation.
        :raises TypeError: If the input data is not a dictionary when a validation
                           schema is actively defined, or if the schema itself
                           is malformed.
        :raises DataValidationError: If the input data fails any validation rule
                                     defined in the effective schema.
        :return: The original input data, unchanged, if validation passes successfully.
        """
        effective_schema = self._validation_schema.copy()
        if 'validation_schema' in context:
            # Context schema dynamically overrides or extends the initialized schema
            dynamic_schema = context['validation_schema']
            if not isinstance(dynamic_schema, dict):
                error_msg = (f"Context 'validation_schema' must be a dictionary, "
                             f"but got {type(dynamic_schema).__name__}.")
                logger.error(error_msg)
                raise TypeError(error_msg)
            effective_schema.update(dynamic_schema)
            logger.debug(f"Validation schema updated from context for this run: {effective_schema}")

        if not effective_schema:
            logger.info("No validation schema provided or detected. Data passes without specific checks.")
            return data

        if not isinstance(data, dict):
            error_msg = (f"DataValidatorNode expects dictionary input for validation with a schema, "
                         f"but received {type(data).__name__}.")
            logger.error(error_msg)
            raise TypeError(error_msg)

        validation_errors: List[str] = []

        for field_name, rules in effective_schema.items():
            if not isinstance(rules, dict):
                validation_errors.append(f"Validation rules for field '{field_name}' must be a dictionary.")
                continue

            field_value = data.get(field_name)
            is_present = field_name in data

            # 1. Check for required fields
            if rules.get("required", False) and not is_present:
                validation_errors.append(f"Field '{field_name}' is required but missing.")
                continue  # Stop further checks for a missing required field

            # If field is not required and not present, no further checks are needed for it.
            if not rules.get("required", False) and not is_present:
                continue

            # 2. Check data type
            expected_type: Optional[Type] = rules.get("type")
            if expected_type and not isinstance(field_value, expected_type):
                validation_errors.append(
                    f"Field '{field_name}' expected type {getattr(expected_type, '__name__', str(expected_type))}, "
                    f"but got {type(field_value).__name__} (value: {field_value})."
                )
                # Continue, as type mismatch might make other checks on the value meaningless
                continue

            # 3. Check length (for strings, lists, dicts)
            if expected_type in [str, list, dict]:
                current_len = len(field_value)
                min_len: Optional[int] = rules.get("min_length")
                if min_len is not None and current_len < min_len:
                    validation_errors.append(
                        f"Field '{field_name}' requires minimum length of {min_len}, "
                        f"but got {current_len} (value: '{field_value}')."
                    )
                max_len: Optional[int] = rules.get("max_length")
                if max_len is not None and current_len > max_len:
                    validation_errors.append(
                        f"Field '{field_name}' allows maximum length of {max_len}, "
                        f"but got {current_len} (value: '{field_value}')."
                    )

            # 4. Check value range (for numeric types)
            if expected_type in [int, float]:
                min_value: Optional[Union[int, float]] = rules.get("min_value")
                if min_value is not None and field_value < min_value:
                    validation_errors.append(
                        f"Field '{field_name}' requires minimum value of {min_value}, "
                        f"but got {field_value}."
                    )
                max_value: Optional[Union[int, float]] = rules.get("max_value")
                if max_value is not None and field_value > max_value:
                    validation_errors.append(
                        f"Field '{field_name}' allows maximum value of {max_value}, "
                        f"but got {field_value}."
                    )

            # 5. Check allowed values
            allowed_values: Optional[List[Any]] = rules.get("allowed_values")
            if allowed_values is not None and field_value not in allowed_values:
                validation_errors.append(
                    f"Field '{field_name}' value '{field_value}' is not among allowed values: {allowed_values}."
                )

            # 6. Custom validator function
            custom_validator: Optional[Callable[[Any], bool]] = rules.get("custom_validator")
            if custom_validator:
                if not callable(custom_validator):
                    validation_errors.append(f"Custom validator for field '{field_name}' is not a callable.")
                else:
                    try:
                        if not custom_validator(field_value):
                            validation_errors.append(
                                f"Field '{field_name}' failed custom validation for value '{field_value}'."
                            )
                    except Exception as e:
                        validation_errors.append(
                            f"Custom validator for field '{field_name}' raised an unexpected exception: {e}."
                        )
                        logger.exception(f"Exception during custom validation for field '{field_name}'.")

        if validation_errors:
            error_message = f"Data validation failed for input. Errors: {'; '.join(validation_errors)}"
            logger.warning(error_message)
            raise DataValidationError(error_message)

        logger.info("Data validated successfully against schema.")
        return data