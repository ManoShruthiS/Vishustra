import logging
from typing import Any, Dict, Union

# Assuming vishustra_core is a package at the project root
# For local development/testing, you might need to adjust sys.path or use relative import if in same directory
# from .base_node import BaseNode
# For the specified project context, it's typically an absolute import from the project root.
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class DataValidatorNode(BaseNode):
    """
    A Vishustra processing node responsible for validating input data against
    a defined set of rules provided in the context.

    The node expects 'validation_rules' in the context dictionary.
    These rules should be a dictionary mapping data field names to their expected
    Python types (e.g., {'name': str, 'age': int, 'is_active': bool}).

    If 'validation_rules' are not provided, the node passes the data through
    without validation, logging a warning.

    If validation fails due to missing fields or type mismatches, it raises
    appropriate errors and logs the failure.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "DataValidatorNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Validates the input data based on rules specified in the context.

        Args:
            data (Any): The data to be validated. Expected to be a dictionary
                        if validation_rules are provided.
            context (Dict[str, Any]): A dictionary containing operational context,
                                      including 'validation_rules'.

        Returns:
            Any: The original data if all validation checks pass.

        Raises:
            TypeError: If `data` is not a dictionary when `validation_rules` are present,
                       or if a field's type does not match the expected type.
            ValueError: If a required field is missing from the data.
        """
        validation_rules: Union[Dict[str, Any], None] = context.get("validation_rules")

        if not validation_rules:
            logger.info(
                f"[{self.node_name}] No 'validation_rules' found in context. "
                "Data will be passed through without validation."
            )
            return data

        logger.debug(f"[{self.node_name}] Starting data validation with rules: {validation_rules}")

        if not isinstance(data, dict):
            error_msg = (
                f"[{self.node_name}] Validation rules are present, but input data "
                f"is not a dictionary. Received type: {type(data).__name__}. Data: {data!r}"
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

        for field_name, expected_type in validation_rules.items():
            # Check for field existence
            if field_name not in data:
                error_msg = (
                    f"[{self.node_name}] Validation failed: Required field "
                    f"'{field_name}' is missing from data. Data keys: {list(data.keys())}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            field_value = data[field_name]

            # Check for type match
            if not isinstance(field_value, expected_type):
                error_msg = (
                    f"[{self.node_name}] Validation failed for field '{field_name}': "
                    f"Expected type '{expected_type.__name__}', but got "
                    f"'{type(field_value).__name__}' with value '{field_value!r}'."
                )
                logger.error(error_msg)
                raise TypeError(error_msg)

            logger.debug(
                f"[{self.node_name}] Field '{field_name}' passed validation "
                f"(type: {type(field_value).__name__})."
            )

        logger.info(f"[{self.node_name}] Data successfully passed all validation checks.")
        return data