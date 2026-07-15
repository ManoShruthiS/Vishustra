import logging
from typing import Any, Dict, Callable, Optional, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class DataValidatorNode(BaseNode):
    """
    A Vishustra processing node responsible for validating input data.

    This node offers flexibility by allowing validation either through
    a custom callable function or a basic schema definition. It ensures
    that data conforms to expected formats or rules before further processing
    in the orchestration pipeline.
    """

    def __init__(
        self,
        node_id: str,
        validation_fn: Optional[Callable[[Any, Dict[str, Any]], bool]] = None,
        schema: Optional[Dict[str, Any]] = None,
        error_message: str = "Input data failed validation.",
        raise_on_invalid: bool = True,
    ):
        """
        Initializes the DataValidatorNode.

        This node requires either a `validation_fn` or a `schema` to be provided.
        It's designed to be flexible; for complex schema validations (e.g., full
        JSON Schema Draft 7+ features), `validation_fn` allows integration with
        external validation libraries, or the internal basic schema validation
        can be utilized for simpler checks.

        Args:
            node_id (str): A unique identifier for this instance of the node.
                           Used to differentiate multiple validator nodes.
            validation_fn (Optional[Callable[[Any, Dict[str, Any]], bool]]):
                An optional callable that takes `data` and `context` as arguments.
                It should return `True` if the data is valid, `False` otherwise.
                If provided, this function will be used for validation.
            schema (Optional[Dict[str, Any]]):
                An optional dictionary representing a validation schema (e.g., a simplified
                JSON schema structure). If `validation_fn` is not provided,
                a basic internal validation based on this schema will be performed.
                Currently supports top-level 'type' and 'required' checks, and
                'type' checks within 'properties' for dictionary data.
            error_message (str): A custom message to use when validation fails.
            raise_on_invalid (bool): If `True`, a `ValueError` is raised upon
                                     validation failure. If `False`, validation
                                     failures are logged as warnings, but data
                                     is passed through (use with caution).

        Raises:
            ValueError: If both `validation_fn` and `schema` are provided,
                        or if neither is provided.
        """
        if validation_fn is None and schema is None:
            raise ValueError("Either 'validation_fn' or 'schema' must be provided to DataValidatorNode.")
        if validation_fn is not None and schema is not None:
            raise ValueError("Cannot provide both 'validation_fn' and 'schema'. Choose one validation method.")

        self._node_id = node_id
        self._validation_fn = validation_fn
        self._schema = schema
        self._error_message = error_message
        self._raise_on_invalid = raise_on_invalid
        logger.debug(f"DataValidatorNode '{self.node_name}' initialized with ID: '{node_id}'.")

    @property
    def node_name(self) -> str:
        """Returns the unique name of this node instance."""
        return f"DataValidatorNode:{self._node_id}"

    def _validate_with_schema(self, data: Any, schema: Dict[str, Any]) -> bool:
        """
        Performs basic validation based on a provided schema dictionary.

        This method offers a simplified simulation of schema validation and does
        not implement a full JSON Schema validator. It checks for overall data
        'type' and, if the data is a dictionary, performs 'required' field checks
        and 'type' checks for properties.

        Args:
            data (Any): The input data to validate.
            schema (Dict[str, Any]): The schema dictionary defining validation rules.

        Returns:
            bool: `True` if the data conforms to the schema, `False` otherwise.
        """
        expected_overall_type = schema.get('type')

        # Validate overall data type if specified
        if expected_overall_type:
            type_checks = {
                'object': dict,
                'string': str,
                'integer': int,
                'number': (int, float),
                'boolean': bool,
                'array': list,
            }
            if expected_overall_type in type_checks:
                if not isinstance(data, type_checks[expected_overall_type]):
                    logger.warning(
                        f"[{self.node_name}] Schema expects overall type '{expected_overall_type}', "
                        f"but data is type '{type(data).__name__}'."
                    )
                    return False
            else:
                logger.warning(
                    f"[{self.node_name}] Schema specifies unsupported overall type '{expected_overall_type}'. "
                    "Skipping overall type validation."
                )

        # If data is a dictionary, perform property-level checks
        if isinstance(data, dict):
            # Check 'required' fields
            required_fields = schema.get('required', [])
            for field in required_fields:
                if field not in data:
                    logger.warning(f"[{self.node_name}] Required field '{field}' is missing from data.")
                    return False

            # Check 'properties' for types if specified
            properties = schema.get('properties', {})
            for field, field_schema in properties.items():
                if field in data:
                    expected_field_type = field_schema.get('type')
                    current_value = data[field]
                    
                    if expected_field_type:
                        if expected_field_type == 'string' and not isinstance(current_value, str):
                            logger.warning(
                                f"[{self.node_name}] Field '{field}' expected type 'string', "
                                f"got '{type(current_value).__name__}'."
                            )
                            return False
                        elif expected_field_type == 'integer' and not isinstance(current_value, int):
                            logger.warning(
                                f"[{self.node_name}] Field '{field}' expected type 'integer', "
                                f"got '{type(current_value).__name__}'."
                            )
                            return False
                        elif expected_field_type == 'number' and not isinstance(current_value, (int, float)):
                            logger.warning(
                                f"[{self.node_name}] Field '{field}' expected type 'number', "
                                f"got '{type(current_value).__name__}'."
                            )
                            return False
                        elif expected_field_type == 'boolean' and not isinstance(current_value, bool):
                            logger.warning(
                                f"[{self.node_name}] Field '{field}' expected type 'boolean', "
                                f"got '{type(current_value).__name__}'."
                            )
                            return False
                        elif expected_field_type == 'array' and not isinstance(current_value, list):
                            logger.warning(
                                f"[{self.node_name}] Field '{field}' expected type 'array', "
                                f"got '{type(current_value).__name__}'."
                            )
                            return False
                        elif expected_field_type == 'object' and not isinstance(current_value, dict):
                            logger.warning(
                                f"[{self.node_name}] Field '{field}' expected type 'object', "
                                f"got '{type(current_value).__name__}'."
                            )
                            return False
        
        return True

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by validating it against the configured rules or schema.

        Args:
            data (Any): The input data to be validated.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       relevant to the current processing pipeline.

        Returns:
            Any: The original, validated data if validation passes.

        Raises:
            ValueError: If validation fails and the node is configured to raise an error.
            RuntimeError: If no validation method was configured (should be caught by `__init__`).
        """
        is_valid = False
        validation_failure_reason = ""

        try:
            if self._validation_fn:
                logger.debug(f"[{self.node_name}] Executing custom validation function for data.")
                is_valid = self._validation_fn(data, context)
                if not is_valid:
                    validation_failure_reason = "Custom validation function returned False."
            elif self._schema:
                logger.debug(f"[{self.node_name}] Applying schema-based validation to data.")
                is_valid = self._validate_with_schema(data, self._schema)
                if not is_valid:
                    validation_failure_reason = "Data failed schema validation."
            else:
                # This state should ideally be prevented by __init__ checks.
                logger.error(f"[{self.node_name}] No validation method configured, this indicates a node misconfiguration.")
                raise RuntimeError("No validation method configured for DataValidatorNode instance.")

        except Exception as e:
            logger.exception(f"[{self.node_name}] An unexpected error occurred during validation: {e}")
            validation_failure_reason = f"Validation encountered an exception: {type(e).__name__} - {e}"
            is_valid = False # Ensure validation status is explicitly false on error

        if not is_valid:
            full_error_msg = f"[{self.node_name}] {self._error_message} Details: {validation_failure_reason}"
            if self._raise_on_invalid:
                logger.error(full_error_msg)
                raise ValueError(full_error_msg)
            else:
                logger.warning(f"{full_error_msg} (Proceeding with invalid data as configured.)")
        else:
            logger.info(f"[{self.node_name}] Data successfully passed validation.")
        
        return data