import logging
import re
from typing import Any, Dict, Type, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class DataValidatorNode(BaseNode):
    """
    A processing node that validates input data against a defined schema.

    This node expects the validation schema to be provided in the 'context'
    dictionary under the key 'validation_schema'.

    Each entry in the 'validation_schema' should be a dictionary representing
    rules for a specific field, which can include:
    - 'type': The expected Python type(s) (e.g., str, int, float, list, dict,
              or a tuple of types like (int, float)).
    - 'required': A boolean indicating if the field is mandatory (default: False).
    - 'min_length': For strings/lists/tuples, the minimum allowed length.
    - 'max_length': For strings/lists/tuples, the maximum allowed length.
    - 'min_value': For numbers (int, float), the minimum allowed value.
    - 'max_value': For numbers (int, float), the maximum allowed value.
    - 'pattern': For strings, a regex pattern that the field's value must fully match.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "DataValidator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Validates the input data against the schema provided in the context.

        Args:
            data: The input data to be validated. Expected to be a dictionary.
            context: A dictionary containing operational context, which must
                     include 'validation_schema' for validation rules.

        Returns:
            The original data if validation is successful.

        Raises:
            TypeError: If the input data is not a dictionary or a field has an
                       incorrect type according to the schema.
            ValueError: If the validation schema is missing or invalid in the
                        context, a required field is missing, or other validation
                        rules (e.g., length, value range, pattern) are violated.
        """
        logger.debug("Entering DataValidatorNode.process. Data type: %s", type(data))

        if not isinstance(data, dict):
            logger.error("Input data for DataValidatorNode must be a dictionary. Got: %s", type(data))
            raise TypeError(f"Input data must be a dictionary. Got {type(data).__name__}.")

        validation_schema = context.get("validation_schema")
        if not isinstance(validation_schema, dict):
            logger.error(
                "Validation schema not found or is invalid in context. "
                "Expected 'validation_schema': Dict[str, Dict[str, Any]]. Got: %s",
                type(validation_schema)
            )
            raise ValueError(
                "Validation schema must be provided in 'context' under "
                "'validation_schema' and be a dictionary of field rules."
            )

        validation_errors = []

        for field_name, rules in validation_schema.items():
            if not isinstance(rules, dict):
                logger.warning(
                    "Invalid rules definition for field '%s' in schema. Expected dict, got %s. Skipping validation for this field.",
                    field_name, type(rules)
                )
                validation_errors.append(f"Invalid schema rules for field '{field_name}'.")
                continue

            field_value = data.get(field_name)
            is_present = field_name in data

            # 1. Check 'required' rule
            if rules.get("required", False) and not is_present:
                validation_errors.append(f"Required field '{field_name}' is missing.")
                logger.warning("Validation failed: Required field '%s' missing.", field_name)
                continue  # Skip further checks for this field if it's missing and required

            if not is_present:  # If field is optional and not present, no further checks are needed
                continue

            # 2. Check 'type' rule
            expected_type: Union[Type, tuple[Type, ...], None] = rules.get("type")
            if expected_type is not None:
                if not isinstance(field_value, expected_type):
                    type_name = getattr(expected_type, '__name__', str(expected_type))
                    validation_errors.append(
                        f"Field '{field_name}' has incorrect type. "
                        f"Expected {type_name}, got {type(field_value).__name__}."
                    )
                    logger.warning(
                        "Validation failed for field '%s': Type mismatch. Expected %s, got %s.",
                        field_name, type_name, type(field_value).__name__
                    )
                    continue  # Skip further checks for this field if type is wrong

            # 3. Check 'min_length' and 'max_length' for sequences (str, list, tuple)
            if isinstance(field_value, (str, list, tuple)):
                current_len = len(field_value)
                min_len = rules.get("min_length")
                max_len = rules.get("max_length")

                if min_len is not None and current_len < min_len:
                    validation_errors.append(
                        f"Field '{field_name}' length ({current_len}) is less than minimum required ({min_len})."
                    )
                    logger.warning(
                        "Validation failed for field '%s': Length (%d) under min_length (%d).",
                        field_name, current_len, min_len
                    )
                if max_len is not None and current_len > max_len:
                    validation_errors.append(
                        f"Field '{field_name}' length ({current_len}) exceeds maximum allowed ({max_len})."
                    )
                    logger.warning(
                        "Validation failed for field '%s': Length (%d) over max_length (%d).",
                        field_name, current_len, max_len
                    )

            # 4. Check 'min_value' and 'max_value' for numbers
            if isinstance(field_value, (int, float)):
                min_val = rules.get("min_value")
                max_val = rules.get("max_value")

                if min_val is not None and field_value < min_val:
                    validation_errors.append(
                        f"Field '{field_name}' value ({field_value}) is less than minimum allowed ({min_val})."
                    )
                    logger.warning(
                        "Validation failed for field '%s': Value (%.2f) under min_value (%.2f).",
                        field_name, field_value, min_val
                    )
                if max_val is not None and field_value > max_val:
                    validation_errors.append(
                        f"Field '{field_name}' value ({field_value}) exceeds maximum allowed ({max_val})."
                    )
                    logger.warning(
                        "Validation failed for field '%s': Value (%.2f) over max_value (%.2f).",
                        field_name, field_value, max_val
                    )

            # 5. Check 'pattern' for strings
            if isinstance(field_value, str) and "pattern" in rules:
                pattern = rules["pattern"]
                if not isinstance(pattern, str):
                    logger.warning("Invalid pattern definition for field '%s'. Expected string, got %s.", field_name, type(pattern))
                    validation_errors.append(f"Invalid pattern definition for field '{field_name}'.")
                    continue

                try:
                    if not re.fullmatch(pattern, field_value):
                        validation_errors.append(
                            f"Field '{field_name}' value does not fully match required pattern '{pattern}'."
                        )
                        logger.warning(
                            "Validation failed for field '%s': Value '%s' does not match pattern '%s'.",
                            field_name, field_value, pattern
                        )
                except re.error as e:
                    logger.error(
                        "Invalid regex pattern '%s' provided for field '%s': %s",
                        pattern, field_name, e
                    )
                    validation_errors.append(
                        f"Internal error: Invalid regex pattern provided for field '{field_name}'."
                    )

        if validation_errors:
            full_error_msg = f"Data validation failed: {'; '.join(validation_errors)}"
            logger.error("Node '%s' failed validation. Errors: %s", self.node_name, full_error_msg)
            raise ValueError(full_error_msg)
        
        logger.info("Data validation successful for node '%s'. Data (first 100 chars): %s", self.node_name, str(data)[:100])
        logger.debug("Exiting DataValidatorNode.process.")
        return data
