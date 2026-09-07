import logging
from typing import Any, Dict, Type, Union

# Assuming this path structure for the base node
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class DataValidatorNode(BaseNode):
    """
    A Vishustra processing node that validates input data against a defined schema
    provided in the context. This node ensures data integrity and conformity to
    expected structures before further processing.

    Expected `context` structure for validation configuration:
    ```python
    {
        "validation_config": {
            "expected_type": <Type>,  # e.g., dict, list, str, int. This is mandatory.
            "schema": {               # Optional: For dict-like data structures.
                                      # If expected_type is dict but schema is absent,
                                      # only the overall type check is performed.
                "field_name_1": {
                    "type": <Type>,           # e.g., str, int, bool.
                    "required": bool,         # Default: False. If True, field must exist.
                    "min_value": Union[int, float], # For numeric types.
                    "max_value": Union[int, float], # For numeric types.
                    "min_length": int,        # For string/list/dict types.
                    "max_length": int,        # For string/list/dict types.
                    # Add more validation rules as needed (e.g., regex, choices, custom_validator_func).
                },
                "field_name_2": { ... },
                # ... other fields
            }
        }
    }
    ```
    If `validation_config` or `expected_type` is missing/invalid, a `ValueError` will be raised.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "DataValidator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Validates the input data based on rules specified in the context.

        Args:
            data: The input data to be validated.
            context: A dictionary containing validation configuration.
                     Must include "validation_config" with "expected_type".
                     Optionally includes a "schema" for detailed validation of
                     dict-like data structures.

        Returns:
            The original `data`, if all validation checks pass.

        Raises:
            ValueError: If validation configuration is missing or invalid,
                        or if the input data fails any validation rule.
        """
        logger.debug("DataValidatorNode received data for validation.")

        validation_config = context.get("validation_config")
        if not validation_config or not isinstance(validation_config, dict):
            logger.error("Validation configuration is missing or malformed in the context for DataValidatorNode.")
            raise ValueError("Missing or invalid 'validation_config' in context. Cannot validate data.")

        expected_type: Type = validation_config.get("expected_type")
        if not isinstance(expected_type, type):
            logger.error(f"Invalid 'expected_type' in validation_config: {expected_type!r}. Must be a type object.")
            raise ValueError(f"Invalid 'expected_type' in validation_config: {expected_type!r}. Must be a type object.")

        # 1. Overall type validation
        if not isinstance(data, expected_type):
            logger.warning(
                f"Data type mismatch. Expected '{expected_type.__name__}', got '{type(data).__name__}'."
            )
            raise ValueError(
                f"Data validation failed: Expected type '{expected_type.__name__}',"
                f" but received '{type(data).__name__}'."
            )
        logger.debug(f"Overall type validation passed: Data is of type '{expected_type.__name__}'.")

        # 2. Schema-based validation for dict-like data
        if expected_type is dict:
            schema: Dict[str, Dict[str, Any]] = validation_config.get("schema", {})
            if not schema:
                logger.debug("No specific schema provided for dictionary data. Only overall type checked.")
                # If no schema, and it's a dict, overall type check is sufficient.
                logger.info("Data validated successfully (overall type check only).")
                return data

            for field_name, rules in schema.items():
                if not isinstance(rules, dict):
                    logger.error(f"Schema rules for field '{field_name}' are malformed. Expected a dictionary.")
                    raise ValueError(f"Validation failed: Malformed schema rules for field '{field_name}'.")

                is_required = rules.get("required", False)
                field_type = rules.get("type")

                if is_required and field_name not in data:
                    logger.warning(f"Required field '{field_name}' is missing from data.")
                    raise ValueError(f"Validation failed: Required field '{field_name}' is missing.")

                if field_name in data:
                    field_value = data[field_name]

                    # Field type validation
                    if field_type and not isinstance(field_value, field_type):
                        logger.warning(
                            f"Field '{field_name}' type mismatch. Expected '{field_type.__name__}', "
                            f"got '{type(field_value).__name__}'. Value: {field_value!r}"
                        )
                        raise ValueError(
                            f"Validation failed for field '{field_name}': Expected type '{field_type.__name__}',"
                            f" but received '{type(field_value).__name__}'."
                        )

                    # Value constraints (numeric types)
                    if isinstance(field_value, (int, float)):
                        min_value = rules.get("min_value")
                        max_value = rules.get("max_value")

                        if min_value is not None and not isinstance(min_value, (int, float)):
                            logger.warning(f"Invalid 'min_value' type for field '{field_name}'. Skipping check.")
                        elif min_value is not None and field_value < min_value:
                            logger.warning(
                                f"Field '{field_name}' value '{field_value}' is below minimum '{min_value}'."
                            )
                            raise ValueError(
                                f"Validation failed for field '{field_name}': Value '{field_value}'"
                                f" is below minimum '{min_value}'."
                            )

                        if max_value is not None and not isinstance(max_value, (int, float)):
                            logger.warning(f"Invalid 'max_value' type for field '{field_name}'. Skipping check.")
                        elif max_value is not None and field_value > max_value:
                            logger.warning(
                                f"Field '{field_name}' value '{field_value}' is above maximum '{max_value}'."
                            )
                            raise ValueError(
                                f"Validation failed for field '{field_name}': Value '{field_value}'"
                                f" is above maximum '{max_value}'."
                            )

                    # Length constraints (string, list, dict)
                    if isinstance(field_value, (str, list, dict)):
                        min_length = rules.get("min_length")
                        max_length = rules.get("max_length")

                        if min_length is not None and not isinstance(min_length, int):
                            logger.warning(f"Invalid 'min_length' type for field '{field_name}'. Skipping check.")
                        elif min_length is not None and len(field_value) < min_length:
                            logger.warning(
                                f"Field '{field_name}' length '{len(field_value)}' is below minimum '{min_length}'."
                            )
                            raise ValueError(
                                f"Validation failed for field '{field_name}': Length '{len(field_value)}'"
                                f" is below minimum '{min_length}'."
                            )

                        if max_length is not None and not isinstance(max_length, int):
                            logger.warning(f"Invalid 'max_length' type for field '{field_name}'. Skipping check.")
                        elif max_length is not None and len(field_value) > max_length:
                            logger.warning(
                                f"Field '{field_name}' length '{len(field_value)}' is above maximum '{max_length}'."
                            )
                            raise ValueError(
                                f"Validation failed for field '{field_name}': Length '{len(field_value)}'"
                                f" is above maximum '{max_length}'."
                            )
            logger.info("Data validated successfully against schema.")

        # If data passes all validations
        logger.info("Data validation completed successfully.")
        return data