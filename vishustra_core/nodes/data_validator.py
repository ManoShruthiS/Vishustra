import logging
from typing import Any, Dict, Type, Union

# Assuming vishustra_core.nodes.base_node exists as per project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

# A lookup table for resolving common built-in types from their string names.
# This avoids the use of 'eval' for type resolution, enhancing security and clarity.
_type_lookup: Dict[str, Type] = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    "Any": Any, # For cases where any type is explicitly allowed via schema
}

def _resolve_type(type_indicator: Union[str, Type]) -> Type:
    """
    Resolves a type indicator (either a string name or an actual type object)
    to its corresponding Python type object.

    Args:
        type_indicator: A string representing a type name (e.g., "int", "str")
                        or a direct type object (e.g., int, str).

    Returns:
        The resolved Python type object.

    Raises:
        ValueError: If the type indicator is an unrecognized string or an invalid type.
    """
    if isinstance(type_indicator, str):
        resolved_type = _type_lookup.get(type_indicator)
        if resolved_type is None:
            raise ValueError(f"Unrecognized type string in schema: '{type_indicator}'")
        return resolved_type
    elif isinstance(type_indicator, type):
        return type_indicator
    else:
        raise ValueError(
            f"Invalid type indicator provided in schema: {type_indicator!r}. "
            f"Expected a string or a type object."
        )


class DataValidator(BaseNode):
    """
    A Vishustra processing node responsible for validating input data against
    a predefined schema provided in the operational context.

    This node ensures data integrity and adherence to expected structures,
    raising errors early if validation rules are violated.

    The validation schema is expected to be located in `context['validation_schema']`
    and can include the following rules:

    -   `expected_type`: The expected type for the top-level `data` itself
                         (e.g., `dict`, `list`, `str`, `int`, or their string names).
    -   `required_keys`: (Applicable if `data` is a dictionary) A list of keys
                         that must be present in the dictionary.
    -   `field_types`: (Applicable if `data` is a dictionary) A dictionary mapping
                       field names to their expected types (e.g., `{'id': int, 'name': 'str'}`).
                       Types can be provided as objects or their string names.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "DataValidator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by validating it against the schema defined
        in the `context`.

        Args:
            data: The input data payload to be validated.
            context: A dictionary containing operational context, crucially
                     including the 'validation_schema' with specific rules.

        Returns:
            The original `data` payload if all validation checks pass successfully.

        Raises:
            ValueError: If 'validation_schema' is missing, malformed, or if the
                        input `data` fails any of the specified validation rules.
        """
        node_id = context.get('node_id', self.node_name)
        logger.info(f"[{node_id}] Initiating data validation process.")

        validation_schema = context.get('validation_schema')
        if not validation_schema:
            error_msg = f"[{node_id}] Validation schema not found in context. Unable to perform validation."
            logger.error(error_msg)
            raise ValueError(error_msg)
        if not isinstance(validation_schema, dict):
            error_msg = (
                f"[{node_id}] Validation schema must be a dictionary, "
                f"but received type: {type(validation_schema).__name__}."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # 1. Validate the overall data type if 'expected_type' is specified in the schema.
        expected_type_indicator = validation_schema.get('expected_type')
        if expected_type_indicator is not None:
            try:
                expected_type = _resolve_type(expected_type_indicator)
                if not isinstance(data, expected_type):
                    error_msg = (
                        f"[{node_id}] Top-level data type mismatch. Expected '{expected_type.__name__}', "
                        f"but received '{type(data).__name__}' for data: {data!r}."
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                logger.debug(f"[{node_id}] Data passed top-level type check: '{expected_type.__name__}'.")
            except ValueError as ve:
                error_msg = (
                    f"[{node_id}] Configuration error: Failed to resolve 'expected_type' "
                    f"'{expected_type_indicator}' in schema: {ve}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg) from ve
            except Exception as e:
                error_msg = (
                    f"[{node_id}] An unexpected error occurred during top-level 'expected_type' "
                    f"validation for '{expected_type_indicator}': {e}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg) from e

        # 2. If the data is a dictionary, apply dictionary-specific validations.
        if isinstance(data, dict):
            # Validate required keys
            required_keys = validation_schema.get('required_keys')
            if required_keys is not None:
                if not isinstance(required_keys, list):
                    error_msg = (
                        f"[{node_id}] Schema error: 'required_keys' must be a list, "
                        f"but received type: {type(required_keys).__name__}."
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)

                missing_keys = [key for key in required_keys if key not in data]
                if missing_keys:
                    error_msg = (
                        f"[{node_id}] Missing required keys: {', '.join(map(str, missing_keys))}. "
                        f"Data keys provided: {', '.join(map(str, data.keys())) if data else 'None'}."
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                logger.debug(f"[{node_id}] Data passed required keys check.")

            # Validate field types
            field_types = validation_schema.get('field_types')
            if field_types is not None:
                if not isinstance(field_types, dict):
                    error_msg = (
                        f"[{node_id}] Schema error: 'field_types' must be a dictionary, "
                        f"but received type: {type(field_types).__name__}."
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)

                for field, expected_type_indicator_for_field in field_types.items():
                    if field not in data:
                        logger.debug(f"[{node_id}] Skipping type check for missing optional field '{field}'.")
                        continue

                    actual_value = data[field]
                    try:
                        field_expected_type = _resolve_type(expected_type_indicator_for_field)
                        if not isinstance(actual_value, field_expected_type):
                            error_msg = (
                                f"[{node_id}] Field '{field}' type mismatch. Expected '{field_expected_type.__name__}', "
                                f"but received '{type(actual_value).__name__}' for value: {actual_value!r}."
                            )
                            logger.error(error_msg)
                            raise ValueError(error_msg)
                    except ValueError as ve:
                        error_msg = (
                            f"[{node_id}] Configuration error: Failed to resolve type for field '{field}' "
                            f"with schema type '{expected_type_indicator_for_field}': {ve}"
                        )
                        logger.error(error_msg)
                        raise ValueError(error_msg) from ve
                    except Exception as e:
                        error_msg = (
                            f"[{node_id}] An unexpected error occurred during field '{field}' type validation "
                            f"with schema type '{expected_type_indicator_for_field}': {e}"
                        )
                        logger.error(error_msg)
                        raise ValueError(error_msg) from e
                logger.debug(f"[{node_id}] Data passed field types check.")

        # Additional validation logic for other data types (e.g., list length, string regex)
        # can be extended here based on future requirements.

        logger.info(f"[{node_id}] Data successfully validated against schema.")
        return data