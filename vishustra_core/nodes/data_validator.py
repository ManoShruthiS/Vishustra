"""
DataValidator node for Vishustra framework.

This node performs schema-based validation on incoming data, ensuring it conforms
to predefined types, constraints, and presence requirements before further processing.
"""

import logging
import re
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class DataValidator(BaseNode):
    """
    A Vishustra processing node responsible for validating input data against
    a specified schema.

    The validation schema is expected within the 'context' dictionary under the
    key 'validation_schema'. This schema defines rules for each expected field
    in the input `data` dictionary.

    Example `validation_schema` structure:
    {
        "user_id": {"type": int, "required": True, "min_value": 1000},
        "username": {"type": str, "required": True, "min_length": 3, "max_length": 50},
        "email": {
            "type": str,
            "regex": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
            "required": False
        },
        "age": {"type": int, "min_value": 0, "max_value": 120, "required": False}
    }
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this processing node."""
        return "DataValidator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Validates the input `data` against the schema defined in the `context`.

        If a 'validation_schema' is present in the context, this method
        iterates through the schema rules, applying type checks, presence checks,
        and value/length/regex constraints to the corresponding fields in the `data`.

        Args:
            data: The input data to be validated. Expected to be a dictionary
                  if a `validation_schema` is provided in the context.
            context: A dictionary containing operational context, which *must*
                     include a 'validation_schema' key for validation to occur.

        Returns:
            The original `data` dictionary if all validation rules pass.

        Raises:
            TypeError: If `data` is not a dictionary when a `validation_schema`
                       is provided, or if schema rules are malformed.
            ValueError: If `data` fails any of the defined validation rules
                        (e.g., missing required fields, type mismatch, constraint violation).
        """
        schema = context.get("validation_schema")

        if not schema:
            logger.warning(
                "No 'validation_schema' found in context for DataValidator. "
                "Passing data through without validation."
            )
            return data

        if not isinstance(data, dict):
            logger.error(
                f"Validation schema provided, but input data is not a dictionary. "
                f"Expected dict, got {type(data).__name__} for node '{self.node_name}'."
            )
            raise TypeError(
                f"DataValidator expects 'data' to be a dictionary when a schema is present, "
                f"but received {type(data).__name__}."
            )

        logger.debug(
            f"Starting validation for node '{self.node_name}' with data keys: {list(data.keys())}"
        )

        for field_name, rules in schema.items():
            if not isinstance(rules, dict):
                logger.error(
                    f"Malformed schema rule for field '{field_name}': expected a dict, "
                    f"got {type(rules).__name__}."
                )
                raise TypeError(
                    f"Malformed schema for field '{field_name}'. Rules must be a dictionary."
                )

            field_value = data.get(field_name)
            is_required = rules.get("required", False)
            expected_type = rules.get("type")

            # Handle missing required fields
            if field_value is None:
                if is_required:
                    logger.error(
                        f"Validation failed for field '{field_name}': "
                        f"Required field is missing in data."
                    )
                    raise ValueError(f"Field '{field_name}' is required but missing.")
                else:
                    # If not required and missing, no further validation for this field
                    continue

            # Type validation
            if expected_type and not isinstance(field_value, expected_type):
                logger.error(
                    f"Validation failed for field '{field_name}': "
                    f"Expected type {getattr(expected_type, '__name__', str(expected_type))}, "
                    f"got {type(field_value).__name__} with value '{field_value}'."
                )
                raise ValueError(
                    f"Field '{field_name}' must be of type "
                    f"{getattr(expected_type, '__name__', str(expected_type))}, "
                    f"but received {type(field_value).__name__}."
                )

            # Conditional validations based on expected type
            if expected_type == str:
                min_length = rules.get("min_length")
                max_length = rules.get("max_length")
                regex_pattern = rules.get("regex")

                if min_length is not None and len(field_value) < min_length:
                    logger.error(
                        f"Validation failed for field '{field_name}': "
                        f"String length {len(field_value)} is less than minimum {min_length}."
                    )
                    raise ValueError(
                        f"Field '{field_name}' length must be at least {min_length}."
                    )
                if max_length is not None and len(field_value) > max_length:
                    logger.error(
                        f"Validation failed for field '{field_name}': "
                        f"String length {len(field_value)} is greater than maximum {max_length}."
                    )
                    raise ValueError(
                        f"Field '{field_name}' length must be at most {max_length}."
                    )
                if regex_pattern:
                    try:
                        if not re.fullmatch(regex_pattern, field_value):
                            logger.error(
                                f"Validation failed for field '{field_name}': "
                                f"String '{field_value}' does not match regex pattern '{regex_pattern}'."
                            )
                            raise ValueError(
                                f"Field '{field_name}' does not match the required pattern."
                            )
                    except re.error as e:
                        logger.error(
                            f"Malformed regex pattern '{regex_pattern}' for field '{field_name}': {e}"
                        )
                        raise TypeError(
                            f"Invalid regex pattern for field '{field_name}': {e}"
                        ) from e

            elif expected_type in (int, float):
                min_value = rules.get("min_value")
                max_value = rules.get("max_value")

                if min_value is not None and field_value < min_value:
                    logger.error(
                        f"Validation failed for field '{field_name}': "
                        f"Value {field_value} is less than minimum {min_value}."
                    )
                    raise ValueError(
                        f"Field '{field_name}' value must be at least {min_value}."
                    )
                if max_value is not None and field_value > max_value:
                    logger.error(
                        f"Validation failed for field '{field_name}': "
                        f"Value {field_value} is greater than maximum {max_value}."
                    )
                    raise ValueError(
                        f"Field '{field_name}' value must be at most {max_value}."
                    )
            # Future expansion: Add more type-specific validations here (e.g., list, dict, custom objects)

        logger.info(f"Data successfully validated by node '{self.node_name}'.")
        return data