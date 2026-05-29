import logging
import re
from typing import Any, Dict

# Assuming BaseNode is available at this path as per project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class DataValidationError(ValueError):
    """Custom exception raised when data fails validation against a schema."""
    pass

class DataValidatorNode(BaseNode):
    """
    A Vishustra processing node responsible for validating input data against a defined schema.

    The validation schema is expected to be provided in the 'context' dictionary
    under the key 'validation_schema'. This schema defines the expected structure,
    types, and constraints for the input data.

    Example Schema Structure:
    {
        "field_name_1": {
            "type": <Python_Type>,  # e.g., int, str, float, bool, list, dict
            "required": True/False, # default False
            "default": Any,         # Value to apply if field is missing and not required
            "min": Any,             # Minimum value for numbers
            "max": Any,             # Maximum value for numbers
            "min_len": int,         # Minimum length for strings/lists/dicts
            "max_len": int,         # Maximum length for strings/lists/dicts
            "pattern": str,         # Regex pattern for strings
            "allow_none": True/False # default False, allows field value to be None
        },
        "field_name_2": { ... }
    }

    The node attempts basic type coercion for numeric types (str to int/float).
    If validation fails, a DataValidationError is raised.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this node."""
        return "data_validator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Validates the input data against the schema provided in the context.

        Args:
            data: The input data to be validated. Expected to be a dictionary
                  for schema-based validation.
            context: A dictionary containing runtime information, including
                     the 'validation_schema' key with the validation rules.

        Returns:
            The validated data, potentially enriched with default values.

        Raises:
            DataValidationError: If validation fails due to missing required fields,
                                 incorrect types, or violated constraints, or if
                                 the validation schema itself is malformed.
        """
        logger.debug(f"[{self.node_name}] Starting data validation for incoming data.")

        validation_schema = context.get("validation_schema")
        if not validation_schema:
            error_msg = (
                f"[{self.node_name}] Validation schema not found in context. "
                "Please provide it under the 'validation_schema' key."
            )
            logger.error(error_msg)
            raise DataValidationError(error_msg)

        if not isinstance(validation_schema, dict):
            error_msg = (
                f"[{self.node_name}] Invalid validation schema: must be a dictionary. "
                f"Got: {type(validation_schema).__name__}"
            )
            logger.error(error_msg)
            raise DataValidationError(error_msg)

        if not isinstance(data, dict):
            error_msg = (
                f"[{self.node_name}] Input data must be a dictionary for schema validation. "
                f"Got: {type(data).__name__}"
            )
            logger.error(error_msg)
            raise DataValidationError(error_msg)
        
        # Create a mutable copy of the data to apply defaults or coerced values
        processed_data = dict(data)

        for field_name, field_rules in validation_schema.items():
            if not isinstance(field_rules, dict):
                error_msg = (
                    f"[{self.node_name}] Invalid rules for field '{field_name}': "
                    f"must be a dictionary. Got: {type(field_rules).__name__}"
                )
                logger.error(error_msg)
                raise DataValidationError(error_msg)

            field_type = field_rules.get("type")
            is_required = field_rules.get("required", False)
            default_value = field_rules.get("default", Ellipsis) # Using Ellipsis as a unique sentinel
            allow_none = field_rules.get("allow_none", False)

            if field_name not in processed_data:
                if is_required:
                    error_msg = f"[{self.node_name}] Required field '{field_name}' is missing."
                    logger.warning(error_msg)
                    raise DataValidationError(error_msg)
                elif default_value is not Ellipsis:
                    processed_data[field_name] = default_value
                    logger.debug(
                        f"[{self.node_name}] Field '{field_name}' missing, "
                        f"applying default value: {default_value!r}"
                    )
                    continue # Skip further validation if default is applied
                else:
                    logger.debug(
                        f"[{self.node_name}] Optional field '{field_name}' is missing "
                        "and no default provided. Skipping further checks for this field."
                    )
                    continue

            field_value = processed_data[field_name]

            # Check for None value if not explicitly allowed
            if field_value is None and not allow_none:
                error_msg = f"[{self.node_name}] Field '{field_name}' cannot be None."
                logger.warning(error_msg)
                raise DataValidationError(error_msg)

            # Type validation (and basic coercion for common types)
            if field_type and field_value is not None:
                if not isinstance(field_value, field_type):
                    try:
                        if field_type is int and isinstance(field_value, str):
                            processed_data[field_name] = int(field_value)
                            logger.debug(f"[{self.node_name}] Coerced field '{field_name}' to int from string.")
                            field_value = processed_data[field_name] # Update value after coercion
                        elif field_type is float and isinstance(field_value, (str, int)):
                            processed_data[field_name] = float(field_value)
                            logger.debug(f"[{self.node_name}] Coerced field '{field_name}' to float.")
                            field_value = processed_data[field_name] # Update value after coercion
                        else:
                            raise TypeError # Re-raise if not coercible or not a common type
                    except (ValueError, TypeError):
                        error_msg = (
                            f"[{self.node_name}] Field '{field_name}' has incorrect type. "
                            f"Expected {field_type.__name__}, got {type(field_value).__name__}."
                        )
                        logger.warning(error_msg)
                        raise DataValidationError(error_msg)

            # Apply specific validations based on the inferred or coerced type
            if field_value is not None:
                # Length validations (for strings, lists, dicts)
                if isinstance(field_value, (str, list, dict)):
                    current_len = len(field_value)
                    if "min_len" in field_rules and current_len < field_rules["min_len"]:
                        error_msg = (
                            f"[{self.node_name}] Field '{field_name}' has insufficient length. "
                            f"Min length {field_rules['min_len']}, got {current_len}."
                        )
                        logger.warning(error_msg)
                        raise DataValidationError(error_msg)
                    if "max_len" in field_rules and current_len > field_rules["max_len"]:
                        error_msg = (
                            f"[{self.node_name}] Field '{field_name}' has excessive length. "
                            f"Max length {field_rules['max_len']}, got {current_len}."
                        )
                        logger.warning(error_msg)
                        raise DataValidationError(error_msg)

                # String specific validations
                if isinstance(field_value, str) and "pattern" in field_rules:
                    pattern = field_rules["pattern"]
                    if not re.fullmatch(pattern, field_value):
                        error_msg = (
                            f"[{self.node_name}] Field '{field_name}' (string) "
                            f"does not match pattern '{pattern}'."
                        )
                        logger.warning(error_msg)
                        raise DataValidationError(error_msg)

                # Numeric specific validations (int, float)
                if isinstance(field_value, (int, float)):
                    if "min" in field_rules and field_value < field_rules["min"]:
                        error_msg = (
                            f"[{self.node_name}] Field '{field_name}' (numeric) "
                            f"is below minimum. Min value {field_rules['min']}, got {field_value}."
                        )
                        logger.warning(error_msg)
                        raise DataValidationError(error_msg)
                    if "max" in field_rules and field_value > field_rules["max"]:
                        error_msg = (
                            f"[{self.node_name}] Field '{field_name}' (numeric) "
                            f"is above maximum. Max value {field_rules['max']}, got {field_value}."
                        )
                        logger.warning(error_msg)
                        raise DataValidationError(error_msg)

        logger.info(f"[{self.node_name}] Data successfully validated against schema.")
        return processed_data
