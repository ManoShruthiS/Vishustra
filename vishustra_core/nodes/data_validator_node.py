import logging
from typing import Any, Dict, List, Type

# Import BaseNode from the specified project path
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class DataValidatorNode(BaseNode):
    """
    A Vishustra processing node that validates input data against a defined schema
    and required fields provided in the context.

    This node ensures that the input `data` (expected to be a dictionary when rules are present)
    adheres to specified structural and type constraints.

    Expected `context` keys for configuration:
    - 'validation_schema': Dict[str, Type] (optional)
      A mapping of data field names to their expected Python types.
      Example: `{"name": str, "age": int}`.
      Fields not explicitly listed in the schema will be ignored for type validation.
    - 'required_fields': List[str] (optional)
      A list of field names that *must* be present in the input data.
      Example: `["name", "email"]`.

    If validation fails, appropriate `TypeError` or `ValueError` exceptions are raised.
    If no validation rules (`validation_schema` or `required_fields`) are provided
    in the context, the node acts as a passthrough.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "DataValidator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by validating it against the schema and required fields
        defined in the context.

        Args:
            data (Any): The input data to be validated. It is expected to be a dictionary
                        when validation rules are configured.
            context (Dict[str, Any]): A dictionary containing runtime context and
                                     validation configurations.

        Returns:
            Any: The original, validated data if all checks pass.

        Raises:
            TypeError: If the input data is not a dictionary (when schema/required fields
                       are provided), or if a field's type does not match the
                       expected type in the `validation_schema`.
            ValueError: If a field listed in `required_fields` is missing from the data.
        """
        logger.debug(f"[{self.node_name}] Starting data validation for incoming data.")

        validation_schema: Dict[str, Type] = context.get('validation_schema', {})
        required_fields: List[str] = context.get('required_fields', [])

        # If validation rules are present, data must be a dictionary.
        # If no rules are present, this node acts as a passthrough.
        if (validation_schema or required_fields) and not isinstance(data, dict):
            error_msg = (
                f"[{self.node_name}] Validation schema or required fields are defined, "
                f"but input data is not a dictionary. Received type: {type(data).__name__}."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)
        elif not (validation_schema or required_fields):
            logger.warning(
                f"[{self.node_name}] No 'validation_schema' or 'required_fields' "
                f"provided in context. Data validation will be a no-op, passing data through."
            )
            return data # No validation configured, so data is considered valid by default.

        # 1. Validate required fields
        for field in required_fields:
            if field not in data:
                error_msg = f"[{self.node_name}] Required field '{field}' is missing from data."
                logger.error(error_msg)
                raise ValueError(error_msg)
            logger.debug(f"[{self.node_name}] Required field '{field}' found in data.")

        # 2. Validate types based on schema
        for field, expected_type in validation_schema.items():
            if field in data:
                actual_value = data[field]
                if not isinstance(actual_value, expected_type):
                    error_msg = (
                        f"[{self.node_name}] Field '{field}' has an incorrect type. "
                        f"Expected '{expected_type.__name__}', but got '{type(actual_value).__name__}' "
                        f"(value: {actual_value!r})."
                    )
                    logger.error(error_msg)
                    raise TypeError(error_msg)
                logger.debug(f"[{self.node_name}] Field '{field}' type '{expected_type.__name__}' validated successfully.")
            # If a field is in the schema but not in data, it's only an issue if it was
            # explicitly listed in `required_fields`, which has already been checked.
            # Otherwise, an absent field not in `required_fields` is implicitly optional
            # and does not trigger a type error.

        logger.info(f"[{self.node_name}] Data successfully validated against the configured schema and required fields.")
        return data