import logging
import re
from typing import Any, Dict, Type

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class DataValidatorNode(BaseNode):
    """
    A Vishustra node designed to validate input data against a set of predefined
    rules specified in the processing context.

    This node performs schema validation, type checking, and applies various
    constraints (e.g., min/max values, string lengths, regex patterns)
    to ensure data quality and integrity before further processing.

    Validation rules are expected within the `context` dictionary under the
    key 'validation_rules'. Each rule is a dictionary keyed by the field name
    to be validated, with values specifying validation criteria.

    Example `validation_rules` structure within context:
    ```json
    {
        "user_id": {"type": "int", "required": True, "min_value": 1},
        "username": {"type": "str", "required": True, "min_length": 3, "max_length": 50},
        "email": {
            "type": "str",
            "required": False,
            "pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        },
        "age": {"type": "int", "min_value": 0, "max_value": 120},
        "tags": {"type": "list", "required": False}
    }
    ```
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "DataValidatorNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by validating it against the rules provided in the context.

        The node logs detailed information about validation steps and failures.
        If any validation fails, an appropriate error is raised.

        Args:
            data: The input data to be validated. Expected to be a dictionary
                  for structured validation.
            context: A dictionary containing operational context, including
                     'validation_rules' for this node.

        Returns:
            The original data if all validations pass successfully.

        Raises:
            ValueError: If `data` is not a dictionary, a required field is missing,
                        or a value constraint (like min_value/max_value) is violated.
            TypeError: If a field's type does not match the expected type.
        """
        logger.debug(f"[{self.node_name}] Initiating data validation.")

        if not isinstance(data, dict):
            error_msg = (
                f"[{self.node_name}] Invalid input data: Expected a dictionary "
                f"for structured validation, but received {type(data).__name__}."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        validation_rules = context.get("validation_rules")
        if not validation_rules:
            logger.warning(
                f"[{self.node_name}] No 'validation_rules' found in the context. "
                "Validation process will be skipped, returning original data."
            )
            return data

        for field_name, rules in validation_rules.items():
            logger.debug(f"[{self.node_name}] Validating field: '{field_name}' with rules: {rules}")

            is_required = rules.get("required", False)
            expected_type_str = rules.get("type")

            # 1. Check for required fields
            if is_required and field_name not in data:
                error_msg = (
                    f"[{self.node_name}] Validation failed for field '{field_name}': "
                    "Required field is missing from the data."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Proceed with further checks only if field is present or not required
            if field_name in data:
                field_value = data[field_name]

                # 2. Type validation
                if expected_type_str:
                    try:
                        expected_type = self._get_python_type(expected_type_str)
                        if not isinstance(field_value, expected_type):
                            error_msg = (
                                f"[{self.node_name}] Validation failed for field '{field_name}': "
                                f"Expected type '{expected_type_str}', "
                                f"but received '{type(field_value).__name__}' "
                                f"with value '{field_value}'."
                            )
                            logger.error(error_msg)
                            raise TypeError(error_msg)
                    except ValueError as e:
                        logger.warning(
                            f"[{self.node_name}] Invalid type specification '{expected_type_str}' "
                            f"for field '{field_name}': {e}. Skipping type validation for this field."
                        )

                # 3. Numeric range validation (for int/float)
                if isinstance(field_value, (int, float)):
                    min_value = rules.get("min_value")
                    max_value = rules.get("max_value")

                    if min_value is not None and field_value < min_value:
                        error_msg = (
                            f"[{self.node_name}] Validation failed for field '{field_name}': "
                            f"Value {field_value} is less than the minimum allowed {min_value}."
                        )
                        logger.error(error_msg)
                        raise ValueError(error_msg)
                    if max_value is not None and field_value > max_value:
                        error_msg = (
                            f"[{self.node_name}] Validation failed for field '{field_name}': "
                            f"Value {field_value} is greater than the maximum allowed {max_value}."
                        )
                        logger.error(error_msg)
                        raise ValueError(error_msg)

                # 4. String length and pattern validation
                if isinstance(field_value, str):
                    min_length = rules.get("min_length")
                    max_length = rules.get("max_length")
                    pattern = rules.get("pattern")

                    if min_length is not None and len(field_value) < min_length:
                        error_msg = (
                            f"[{self.node_name}] Validation failed for field '{field_name}': "
                            f"Length {len(field_value)} is less than the minimum allowed {min_length}."
                        )
                        logger.error(error_msg)
                        raise ValueError(error_msg)
                    if max_length is not None and len(field_value) > max_length:
                        error_msg = (
                            f"[{self.node_name}] Validation failed for field '{field_name}': "
                            f"Length {len(field_value)} is greater than the maximum allowed {max_length}."
                        )
                        logger.error(error_msg)
                        raise ValueError(error_msg)
                    if pattern is not None and not re.fullmatch(pattern, field_value):
                        error_msg = (
                            f"[{self.node_name}] Validation failed for field '{field_name}': "
                            f"Value '{field_value}' does not match the required pattern '{pattern}'."
                        )
                        logger.error(error_msg)
                        raise ValueError(error_msg)

        logger.info(f"[{self.node_name}] Data validation completed successfully. Returning original data.")
        return data

    def _get_python_type(self, type_str: str) -> Type:
        """
        Converts a string representation of a type (e.g., "int", "str") to its
        corresponding Python type object (e.g., int, str).

        Args:
            type_str: The string name of the type.

        Returns:
            The Python type object.

        Raises:
            ValueError: If the type string is unknown or unsupported.
        """
        type_map = {
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "Any": Any,  # Allow explicit 'Any' for less strict type checks
        }
        py_type = type_map.get(type_str)
        if py_type is None:
            raise ValueError(f"Unknown or unsupported type string specified: '{type_str}'")
        return py_type