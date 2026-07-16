import logging
from typing import Any, Dict, Callable, List, Union
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class VishustraValidationError(ValueError):
    """Custom exception raised for validation errors within Vishustra nodes."""
    pass

class DataValidatorNode(BaseNode):
    """
    A Vishustra node for validating structured data (dictionaries) against a defined schema.

    This node enforces data integrity by checking field types, applying custom callable
    validation rules, and managing the presence of required fields as well as preventing
    unexpected extra fields.
    """

    def __init__(
        self,
        schema: Dict[str, Union[type, Callable[[Any], bool], Callable[[Any, Dict[str, Any], Dict[str, Any]], bool]]],
        required_fields: Union[List[str], None] = None,
        allow_extra_fields: bool = False
    ):
        """
        Initializes the DataValidatorNode with a validation schema and configuration.

        Args:
            schema: A dictionary where keys are field names and values are validation rules.
                    Validation rules can be:
                    - A Python type (e.g., `str`, `int`) to check `isinstance()`.
                    - A callable `func(field_value)` that returns `True` for valid, `False` for invalid.
                    - A callable `func(field_value, full_data_dict, context_dict)` for validations
                      that require access to other data fields or the processing context.
            required_fields: An optional list of field names that *must* be present in the input data.
                             If `None`, all fields explicitly defined in the `schema` are considered required.
                             If an empty list `[]`, no fields are strictly required for presence,
                             but fields in the schema are still validated if present.
            allow_extra_fields: If `False`, any field in the input data that is not defined in the `schema`
                                will cause a `VishustraValidationError`. If `True`, unknown fields are ignored.
        """
        self._schema = schema
        self._required_fields = required_fields if required_fields is not None else list(schema.keys())
        self._allow_extra_fields = allow_extra_fields
        logger.debug(
            f"DataValidatorNode initialized with schema keys: {list(self._schema.keys())}, "
            f"explicitly required fields: {self._required_fields}, "
            f"allow_extra_fields: {self._allow_extra_fields}"
        )

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "DataValidator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by validating it against the configured schema and rules.

        Args:
            data: The input data, expected to be a dictionary for schema validation.
            context: A dictionary containing contextual information relevant to the processing.

        Returns:
            The original input data if all validation checks pass.

        Raises:
            VishustraValidationError: If the data fails any validation rule, is not a dictionary
                                    (when schema implies dictionary structure), is missing required fields,
                                    or contains disallowed extra fields.
        """
        logger.info(f"Node '{self.node_name}' starting data validation.")

        if not isinstance(data, dict):
            error_msg = (
                f"Input data for '{self.node_name}' must be a dictionary when a schema is provided, "
                f"but received type: {type(data).__name__}."
            )
            logger.error(error_msg)
            raise VishustraValidationError(error_msg)

        # 1. Validate required fields for presence
        for field in self._required_fields:
            if field not in data:
                error_msg = f"Required field '{field}' is missing from the input data."
                logger.error(error_msg)
                raise VishustraValidationError(error_msg)

        # 2. Validate fields against schema rules (type checks and custom callables)
        for field_name, rule in self._schema.items():
            if field_name not in data:
                # Field is in schema but not in data. If it's not explicitly required,
                # we don't need to validate it, as it's optional.
                continue

            field_value = data[field_name]
            try:
                if isinstance(rule, type):
                    if not isinstance(field_value, rule):
                        raise VishustraValidationError(
                            f"Field '{field_name}' expected type {rule.__name__}, but received {type(field_value).__name__}."
                        )
                elif callable(rule):
                    validation_passed = False
                    try:
                        # Attempt to call with the most verbose signature (value, data, context)
                        validation_passed = rule(field_value, data, context)
                    except TypeError:
                        try:
                            # Fallback to calling with just (value, data)
                            validation_passed = rule(field_value, data)
                        except TypeError:
                            # Fallback to calling with just (value)
                            validation_passed = rule(field_value)
                    
                    if not validation_passed:
                        raise VishustraValidationError(f"Field '{field_name}' failed custom validation.")
                else:
                    raise VishustraValidationError(
                        f"Invalid schema rule for field '{field_name}'. Expected a type or a callable."
                    )
            except VishustraValidationError as e:
                logger.error(f"Validation failed for field '{field_name}': {e}")
                raise
            except Exception as e:
                logger.exception(
                    f"An unexpected error occurred during validation of field '{field_name}'. "
                    f"Rule type: {type(rule).__name__}, Error: {e}"
                )
                raise VishustraValidationError(
                    f"An unexpected internal error occurred during validation of field '{field_name}'."
                ) from e

        # 3. Check for disallowed extra fields
        if not self._allow_extra_fields:
            extra_fields = [field for field in data if field not in self._schema]
            if extra_fields:
                error_msg = f"Input data contains unexpected extra fields: {', '.join(extra_fields)}. " \
                            f"To allow these, set 'allow_extra_fields=True' in node configuration."
                logger.error(error_msg)
                raise VishustraValidationError(error_msg)

        logger.info(f"Node '{self.node_name}' successfully validated data.")
        return data