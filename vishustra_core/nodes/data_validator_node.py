import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class VishustraValidationError(ValueError):
    """Custom exception for validation errors within Vishustra nodes."""
    pass

class DataValidatorNode(BaseNode):
    """
    A Vishustra processing node responsible for validating input data against
    a specified schema provided in the execution context.

    This node enforces data integrity by checking for the presence of required
    keys and verifying the types of data values according to the schema.
    If validation fails, a `VishustraValidationError` is raised, halting
    further processing and ensuring data quality.

    The validation schema should be passed within the `context` dictionary
    under the key 'validation_schema'. This schema is expected to be a
    dictionary with the following structure:
    - '_required_keys': A list of string keys that *must* be present in the
      input data.
    - Other keys: Expected types (e.g., `str`, `int`, `float`, `list`, `dict`)
      or a tuple of types (e.g., `(int, float)`) for type checking. If a key
      is present in the schema but not in `_required_keys`, it's considered
      optional but its type will be validated if present in the data.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this node."""
        return "DataValidator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Validates the input `data` against the `validation_schema` defined
        in the `context`.

        Args:
            data: The input data to be validated. For schema-based validation,
                  this is typically expected to be a dictionary.
            context: A dictionary containing runtime information and configuration,
                     including the 'validation_schema' for this node.

        Returns:
            The original, validated data if all checks pass.

        Raises:
            VishustraValidationError: If the input data fails any validation
                                      rule defined in the schema, or if the
                                      input data type is incompatible with schema
                                      validation (e.g., schema expects dict, gets list).
            TypeError: If the 'validation_schema' itself is malformed or contains
                       invalid type definitions.
            Exception: For any other unexpected errors during the validation process.
        """
        logger.debug(f"[{self.node_name}] Starting data validation process.")

        try:
            validation_schema = context.get("validation_schema")

            if validation_schema is None:
                logger.warning(
                    f"[{self.node_name}] No 'validation_schema' found in context. "
                    "Data will pass through without validation. Ensure this is intentional."
                )
                return data

            if not isinstance(validation_schema, dict):
                error_msg = (
                    f"[{self.node_name}] Malformed 'validation_schema' in context. "
                    f"Expected a dict, but received {type(validation_schema).__name__}."
                )
                logger.error(error_msg)
                raise TypeError(error_msg)

            if not isinstance(data, dict):
                error_msg = (
                    f"[{self.node_name}] Input 'data' is not a dictionary. "
                    f"Expected dict for schema validation, but received {type(data).__name__}. "
                    "Validation aborted."
                )
                logger.error(error_msg)
                raise VishustraValidationError(error_msg)

            # --- Validate required keys ---
            required_keys = validation_schema.get("_required_keys", [])
            if not isinstance(required_keys, list):
                error_msg = (
                    f"[{self.node_name}] Malformed 'validation_schema': "
                    "'_required_keys' must be a list of strings. "
                    f"Received {type(required_keys).__name__}."
                )
                logger.error(error_msg)
                raise TypeError(error_msg)

            for key in required_keys:
                if not isinstance(key, str):
                    error_msg = (
                        f"[{self.node_name}] Malformed 'validation_schema': "
                        f"'_required_keys' list contains non-string element: {key} (type: {type(key).__name__})."
                    )
                    logger.error(error_msg)
                    raise TypeError(error_msg)
                if key not in data:
                    error_msg = f"[{self.node_name}] Validation failed: Required key '{key}' is missing from data."
                    logger.error(error_msg)
                    raise VishustraValidationError(error_msg)
                logger.debug(f"[{self.node_name}] Required key '{key}' found in data.")

            # --- Validate types for specified keys ---
            for key, expected_type in validation_schema.items():
                if key.startswith('_'):  # Skip internal schema keys like '_required_keys'
                    continue

                if key in data:
                    if not isinstance(expected_type, (type, tuple)):
                        error_msg = (
                            f"[{self.node_name}] Malformed 'validation_schema' for key '{key}': "
                            f"Expected a type or tuple of types, but got {type(expected_type).__name__}."
                        )
                        logger.error(error_msg)
                        raise TypeError(error_msg)

                    if not isinstance(data[key], expected_type):
                        error_msg = (
                            f"[{self.node_name}] Validation failed for key '{key}': "
                            f"Expected type(s) {expected_type}, but received "
                            f"{type(data[key]).__name__} with value '{data[key]}'."
                        )
                        logger.error(error_msg)
                        raise VishustraValidationError(error_msg)
                    logger.debug(f"[{self.node_name}] Key '{key}' passed type validation (expected: {expected_type}).")
                # If key is not in data and not in _required_keys, it's an optional key
                # that is not present, which is a valid state.

            logger.info(f"[{self.node_name}] Data successfully validated against schema.")
            return data

        except (VishustraValidationError, TypeError) as e:
            # Re-raise specific validation and schema definition errors
            raise e
        except Exception as e:
            # Catch any other unexpected errors during validation
            logger.critical(
                f"[{self.node_name}] An unexpected error occurred during data validation: {e}",
                exc_info=True
            )
            raise VishustraValidationError(f"[{self.node_name}] Unexpected error during validation: {e}") from e