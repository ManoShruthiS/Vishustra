import logging
import re
from typing import Any, Dict, Optional, Union, get_origin, get_args

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class DataValidationError(ValueError):
    """Custom exception raised when data validation fails."""
    pass

# A safe mapping for common type names to their Python type objects.
# This avoids using `eval()` directly on arbitrary strings, enhancing security.
_TYPE_MAPPING = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "dict": dict,
    "list": list,
    "any": Any, # For explicitly allowing any type
}

class DataValidatorNode(BaseNode):
    """
    A processing node designed to validate input data against a defined schema or set of rules.

    This node is crucial for ensuring data quality and integrity throughout the
    Vishustra orchestration pipeline. It prevents invalid or malformed data
    from propagating to subsequent processing stages, thereby improving the
    reliability and predictability of complex workflows.

    The validation schema can be provided during the node's initialization or
    dynamically through the `context` dictionary during the `process` call.
    """

    def __init__(self, validation_schema: Optional[Dict[str, Any]] = None):
        """
        Initializes the DataValidatorNode with an optional validation schema.

        The schema defines the rules against which incoming data will be validated.
        If no schema is provided during initialization, the node will look for
        a 'validation_schema' key in the `context` dictionary during processing.

        Args:
            validation_schema: An optional dictionary defining the validation rules.
                               Example schema structure:
                               ```
                               {
                                   "type": "dict", # Overall expected type: can be type object (e.g., dict) or string ("dict", "list", "str", "int")
                                   "required_keys": ["id", "name"], # List of keys that must be present if data is a dict
                                   "key_types": { # Type validation for specific keys if data is a dict
                                       "id": int,              # Can be a type object directly
                                       "name": "str",          # Can be a string representation of a type
                                       "age": Optional[int],   # Type hint object (Union[int, NoneType])
                                       "email": "Optional[str]" # String representation for Optional types
                                   },
                                   "value_constraints": { # Value-based constraints for keys if data is a dict
                                       "age": {"min": 0, "max": 120, "exclusive_max": 121},
                                       "status": {"enum": ["active", "inactive", "pending"]},
                                       "email": {"pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"}
                                   },
                                   "item_schema": { # For list types, a nested schema applied to each item
                                       "type": "dict",
                                       "required_keys": ["item_id", "value"]
                                   }
                               }
                               ```
        """
        self._validation_schema = validation_schema
        logger.debug(f"DataValidatorNode initialized with schema: {self._validation_schema}")

    @property
    def node_name(self) -> str:
        """Returns the programmatic name of the node."""
        return "DataValidator"

    def _resolve_type(self, type_def: Any) -> Union[type, tuple, Any]:
        """
        Resolves a type definition (string, type object, or type hint) into a usable
        type or a tuple of types for `isinstance` checks. This handles simple types,
        `Optional[X]`, and `Union[X, Y]` constructs.
        """
        if isinstance(type_def, type):
            return type_def
        
        # Handle complex type hint objects (e.g., Optional[int], Union[str, int])
        origin = get_origin(type_def)
        args = get_args(type_def)

        if origin is Union:
            # Recursively resolve all types in the Union
            resolved_args = tuple(self._resolve_type(arg) for arg in args)
            # Filter out any `Any` results unless `Any` was explicitly defined in the union
            return tuple(t for t in resolved_args if t is not Any) or Any
        elif origin is not None: # For List[X], Dict[K,V], etc., we primarily care about the origin type
            return origin
        
        if isinstance(type_def, str):
            # Handle common types via mapping
            if type_def.lower() in _TYPE_MAPPING:
                return _TYPE_MAPPING[type_def.lower()]

            # Basic parsing for Optional[X] or Union[X, Y] string representations
            if type_def.startswith("Optional[") and type_def.endswith("]"):
                inner_type_str = type_def[len("Optional["):-1]
                resolved_inner = self._resolve_type(inner_type_str)
                # An Optional[X] effectively resolves to Union[X, NoneType], so return a tuple (X, type(None))
                return (resolved_inner, type(None)) if resolved_inner is not Any else type(None)
            elif type_def.startswith("Union[") and type_def.endswith("]"):
                inner_types_str = type_def[len("Union["):-1].split(',')
                resolved_inner_types = [self._resolve_type(ts.strip()) for ts in inner_types_str if ts.strip()]
                return tuple(t for t in resolved_inner_types if t is not Any) or Any
            else:
                logger.warning(f"Unsupported type string '{type_def}'. Treating as Any, which effectively disables type validation for this rule.")
                return Any # If we can't resolve, treat as Any to allow any type
        
        logger.warning(f"Unrecognized type definition format '{type_def}'. Treating as Any, which effectively disables type validation for this rule.")
        return Any # Default if no resolution possible.

    def _validate_data_against_schema(self, data: Any, schema: Dict[str, Any]) -> None:
        """
        Internal method to perform recursive validation based on a provided schema.
        Raises `DataValidationError` if validation fails.
        """
        if not schema:
            logger.debug("No schema provided for _validate_data_against_schema. Data implicitly passes.")
            return

        # 1. Validate overall data type if specified in the schema
        expected_type_def = schema.get("type")
        if expected_type_def:
            resolved_expected_type = self._resolve_type(expected_type_def)
            if resolved_expected_type is Any:
                logger.debug(f"Overall type definition '{expected_type_def}' resolved to Any. Skipping general type validation.")
            elif not isinstance(data, resolved_expected_type):
                raise DataValidationError(
                    f"Data type mismatch. Expected '{expected_type_def}', got '{type(data).__name__}'."
                )

        # 2. Perform dictionary-specific validations (required keys, key types, value constraints)
        if isinstance(data, dict):
            required_keys = schema.get("required_keys", [])
            for key in required_keys:
                if key not in data:
                    raise DataValidationError(f"Missing required key: '{key}' in data.")

            key_types = schema.get("key_types", {})
            for key, expected_type_def in key_types.items():
                if key in data: # Only validate type if key is present
                    resolved_expected_type = self._resolve_type(expected_type_def)
                    if resolved_expected_type is Any:
                        logger.debug(f"Type definition '{expected_type_def}' for key '{key}' resolved to Any. Skipping type validation for this key.")
                        continue
                    
                    if not isinstance(data[key], resolved_expected_type):
                        raise DataValidationError(
                            f"Key '{key}' has type mismatch. Expected '{expected_type_def}', "
                            f"got '{type(data[key]).__name__}'."
                        )

            value_constraints = schema.get("value_constraints", {})
            for key, constraints in value_constraints.items():
                if key in data and data[key] is not None: # Apply constraints only if value is present and not None
                    value = data[key]

                    # Numeric constraints
                    if isinstance(value, (int, float)):
                        if "min" in constraints and value < constraints["min"]:
                            raise DataValidationError(
                                f"Value for key '{key}' ({value}) is below minimum allowed ({constraints['min']})."
                            )
                        if "max" in constraints and value > constraints["max"]:
                            raise DataValidationError(
                                f"Value for key '{key}' ({value}) is above maximum allowed ({constraints['max']})."
                            )
                        if "exclusive_min" in constraints and value <= constraints["exclusive_min"]:
                            raise DataValidationError(
                                f"Value for key '{key}' ({value}) is not exclusively above minimum allowed ({constraints['exclusive_min']})."
                            )
                        if "exclusive_max" in constraints and value >= constraints["exclusive_max"]:
                            raise DataValidationError(
                                f"Value for key '{key}' ({value}) is not exclusively below maximum allowed ({constraints['exclusive_max']})."
                            )
                    
                    # Enumeration constraint
                    if "enum" in constraints and value not in constraints["enum"]:
                         raise DataValidationError(
                            f"Value for key '{key}' ({value}) is not in allowed enum values ({constraints['enum']})."
                        )
                    
                    # Regex pattern constraint for strings
                    if "pattern" in constraints and isinstance(value, str):
                         try:
                             if not re.match(constraints["pattern"], value):
                                raise DataValidationError(
                                    f"Value for key '{key}' ('{value}') does not match required pattern '{constraints['pattern']}'."
                                )
                         except re.error as e:
                             logger.error(f"Invalid regex pattern '{constraints['pattern']}' for key '{key}': {e}")
                             raise DataValidationError(f"Invalid regex pattern provided for key '{key}'.") from e

        # 3. Perform list-specific validations (e.g., item schema for each element)
        elif isinstance(data, list):
            item_schema = schema.get("item_schema")
            if item_schema:
                for i, item in enumerate(data):
                    try:
                        self._validate_data_against_schema(item, item_schema)
                    except DataValidationError as e:
                        raise DataValidationError(f"Validation failed for list item at index {i}: {e}") from e

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Validates the input data against the configured schema.

        If validation fails, a `DataValidationError` is raised, stopping the pipeline.
        If validation succeeds, the original data is returned, confirming its validity
        and allowing it to proceed to the next node.

        Args:
            data: The input data to be validated.
            context: A dictionary containing contextual information for processing.
                     It can include a 'validation_schema' key if no schema was
                     provided during the node's initialization.

        Returns:
            The validated (and potentially unchanged) input data.

        Raises:
            DataValidationError: If the input data does not conform to the validation schema,
                                 or if an internal error occurs during validation.
        """
        schema_to_use = self._validation_schema
        if not schema_to_use and 'validation_schema' in context:
            schema_to_use = context['validation_schema']
            logger.debug("Using validation schema from context.")
        elif not schema_to_use:
            logger.warning("No validation schema provided to DataValidatorNode (neither at init nor in context). Data will pass through without validation.")
            return data

        logger.info(f"Attempting to validate data with schema: {schema_to_use}")
        try:
            self._validate_data_against_schema(data, schema_to_use)
            logger.info("Data validated successfully.")
            return data
        except DataValidationError as e:
            # Re-raise the original validation error for upstream handling
            logger.error(f"Data validation failed: {e}")
            raise
        except Exception as e:
            # Catch any unexpected errors during the validation process itself
            logger.critical(f"An unexpected internal error occurred during data validation: {e}", exc_info=True)
            raise DataValidationError(f"An unexpected internal error occurred during validation: {e}") from e
