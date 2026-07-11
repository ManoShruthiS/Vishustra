import logging
from typing import Any, Dict, List, Union, Type, Tuple

# Assuming the base_node module path as specified in the project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ValidationError(ValueError):
    """Custom exception raised when data fails validation rules."""
    pass

class DataValidatorNode(BaseNode):
    """
    A Vishustra processing node that validates input data against a set of
    predefined rules.

    This node expects validation rules to be provided in the `context` dictionary
    under the key 'validation_rules'. If no rules are found, the data is passed
    through unchanged with a warning.

    Validation rules can specify:
    - `expected_type`: The overall type of the data (e.g., "dict", "list", "str", "int").
    - `required_keys`: For dictionaries, a list of keys that must be present.
    - `key_types`: For dictionaries, a mapping of keys to their expected types.
                   Types can be specified as strings (e.g., "int", "str") or
                   as a list/tuple of strings/types for Union types (e.g., ["str", "int"]).
    - `min_length`, `max_length`: For strings, lists, or dictionaries, the minimum/maximum length.
    - `min_value`, `max_value`: For numeric types (int, float), the minimum/maximum value.

    Example 'validation_rules' structure in `context`:
    ```python
    {
        "validation_rules": {
            "expected_type": "dict",
            "required_keys": ["id", "name", "payload"],
            "key_types": {
                "id": "int",
                "name": ["str", "NoneType"], # Allows str or None
                "payload": "dict",
                "timestamp": float
            },
            "min_length": 1, # Minimum number of keys for the dict
        }
    }
    ```
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "DataValidator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by validating it against rules found in the context.

        Args:
            data: The input data to be validated.
            context: A dictionary containing execution context, expected to include
                     'validation_rules'.

        Returns:
            The original input data if all validations pass.

        Raises:
            ValidationError: If the data fails any of the specified validation rules.
        """
        validation_rules: Dict[str, Any] = context.get("validation_rules", {})

        # Map string type names to actual Python types
        expected_type_map = {
            "str": str, "int": int, "float": float, "bool": bool,
            "dict": dict, "list": list, "tuple": tuple, "set": set,
            "NoneType": type(None), "any": Any
        }

        if not validation_rules:
            logger.warning(
                f"[{self.node_name}] No validation rules found in context. "
                "Data passed through without validation."
            )
            return data

        logger.debug(
            f"[{self.node_name}] Starting validation for data with rules: {validation_rules}"
        )

        # 1. Validate overall data type if specified
        if "expected_type" in validation_rules:
            expected_type_str = validation_rules["expected_type"]
            expected_type: Union[Type, Tuple[Type, ...]] = expected_type_map.get(expected_type_str)

            if expected_type is None:
                logger.error(
                    f"[{self.node_name}] Invalid 'expected_type' '{expected_type_str}' "
                    "in rules. Skipping overall type validation."
                )
            elif not isinstance(data, expected_type):
                raise ValidationError(
                    f"[{self.node_name}] Data type mismatch. Expected '{expected_type_str}', "
                    f"got '{type(data).__name__}'."
                )
            else:
                logger.debug(f"[{self.node_name}] Data overall type '{type(data).__name__}' is valid.")

        # 2. Validate dictionary-specific rules
        if isinstance(data, dict):
            # Required keys
            if "required_keys" in validation_rules:
                required_keys: List[str] = validation_rules["required_keys"]
                missing_keys = [key for key in required_keys if key not in data]
                if missing_keys:
                    raise ValidationError(
                        f"[{self.node_name}] Missing required keys: {', '.join(missing_keys)}."
                    )
                logger.debug(f"[{self.node_name}] All required keys are present.")

            # Key types
            if "key_types" in validation_rules:
                key_types_rules: Dict[str, Any] = validation_rules["key_types"]
                for key, expected_type_spec in key_types_rules.items():
                    if key not in data:
                        # If a key is not present, we can't validate its type.
                        # It might be an optional key. 'required_keys' handles presence.
                        continue

                    resolved_expected_types: Union[Type, Tuple[Type, ...], Any] = Any

                    if isinstance(expected_type_spec, str):
                        resolved_expected_types = expected_type_map.get(expected_type_spec)
                        if resolved_expected_types is None:
                            logger.warning(
                                f"[{self.node_name}] Invalid expected type string "
                                f"'{expected_type_spec}' for key '{key}'. "
                                "Skipping type validation for this key."
                            )
                            continue
                    elif isinstance(expected_type_spec, type):
                        resolved_expected_types = expected_type_spec
                    elif isinstance(expected_type_spec, (list, tuple)):
                        # Handle Union types specified as a list/tuple of type strings or actual types
                        temp_types = []
                        for t_spec in expected_type_spec:
                            if isinstance(t_spec, str):
                                resolved_t = expected_type_map.get(t_spec)
                                if resolved_t:
                                    temp_types.append(resolved_t)
                                else:
                                    logger.warning(
                                        f"[{self.node_name}] Unresolved type string '{t_spec}' "
                                        f"in key_types for key '{key}'."
                                    )
                            elif isinstance(t_spec, type):
                                temp_types.append(t_spec)
                            else:
                                logger.warning(
                                    f"[{self.node_name}] Invalid element type specification "
                                    f"'{t_spec}' in key_types list for key '{key}'."
                                )
                        resolved_expected_types = tuple(temp_types) if temp_types else Any
                    else:
                        logger.warning(
                            f"[{self.node_name}] Invalid type specification for key '{key}'. "
                            "Skipping type validation for this key."
                        )
                        continue

                    if resolved_expected_types is Any:
                        logger.debug(
                            f"[{self.node_name}] Key '{key}' has 'Any' type requirement. "
                            "Skipping specific type check."
                        )
                    elif not isinstance(data[key], resolved_expected_types):
                        expected_type_names = []
                        if isinstance(resolved_expected_types, tuple):
                            expected_type_names = [t.__name__ for t in resolved_expected_types]
                        elif isinstance(resolved_expected_types, type):
                            expected_type_names = [resolved_expected_types.__name__]
                        else: # Fallback for unexpected `resolved_expected_types`
                            expected_type_names = [str(resolved_expected_types)]

                        raise ValidationError(
                            f"[{self.node_name}] Type mismatch for key '{key}'. Expected one of "
                            f"{'/'.join(expected_type_names)}, got '{type(data[key]).__name__}'."
                        )
                logger.debug(f"[{self.node_name}] All specified key types are valid.")

        # 3. Validate length (for strings, lists, dicts)
        if isinstance(data, (str, list, dict)):
            current_length = len(data)
            if "min_length" in validation_rules:
                min_length = validation_rules["min_length"]
                if not isinstance(min_length, int) or min_length < 0:
                    logger.warning(f"[{self.node_name}] Invalid 'min_length' rule: {min_length}. Skipping.")
                elif current_length < min_length:
                    raise ValidationError(
                        f"[{self.node_name}] Data length {current_length} is less than "
                        f"minimum required {min_length}."
                    )
            if "max_length" in validation_rules:
                max_length = validation_rules["max_length"]
                if not isinstance(max_length, int) or max_length < 0:
                    logger.warning(f"[{self.node_name}] Invalid 'max_length' rule: {max_length}. Skipping.")
                elif current_length > max_length:
                    raise ValidationError(
                        f"[{self.node_name}] Data length {current_length} is greater than "
                        f"maximum allowed {max_length}."
                    )
            logger.debug(f"[{self.node_name}] Data length {current_length} is within bounds.")

        # 4. Validate numeric values
        if isinstance(data, (int, float)):
            if "min_value" in validation_rules:
                min_value = validation_rules["min_value"]
                if not isinstance(min_value, (int, float)):
                    logger.warning(f"[{self.node_name}] Invalid 'min_value' rule: {min_value}. Skipping.")
                elif data < min_value:
                    raise ValidationError(
                        f"[{self.node_name}] Data value {data} is less than "
                        f"minimum allowed {min_value}."
                    )
            if "max_value" in validation_rules:
                max_value = validation_rules["max_value"]
                if not isinstance(max_value, (int, float)):
                    logger.warning(f"[{self.node_name}] Invalid 'max_value' rule: {max_value}. Skipping.")
                elif data > max_value:
                    raise ValidationError(
                        f"[{self.node_name}] Data value {data} is greater than "
                        f"maximum allowed {max_value}."
                    )
            logger.debug(f"[{self.node_name}] Data value {data} is within bounds.")

        logger.info(f"[{self.node_name}] Data successfully validated.")
        return data