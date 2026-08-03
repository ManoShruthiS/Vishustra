import logging
from typing import Any, Dict, Union, get_origin, get_args

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class NodeValidationError(ValueError):
    """
    Custom exception raised when data validation fails within a Node.
    """
    pass

class DataValidatorNode(BaseNode):
    """
    A Vishustra processing node responsible for validating input data against
    a predefined schema or set of rules provided in the context.

    This node ensures data integrity and consistency before it proceeds
    to subsequent processing stages, raising a NodeValidationError if rules are not met.

    Validation rules are expected to be provided in the `context` dictionary
    under the key 'validation_config'.
    Example `validation_config` structure:
    {
        "required_keys": ["id", "timestamp"],
        "schema": {
            "id": {"type": str, "min_length": 5},
            "timestamp": {"type": int, "min_value": 0},
            "payload": {"type": dict, "required_keys": ["action", "data"]},
            "optional_field": {"type": Union[str, None], "default": None}
        },
        "allow_extra_keys": False
    }
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "DataValidatorNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by validating it against rules specified in the context.

        Args:
            data: The input data to be validated.
            context: A dictionary containing operational context, including
                     'validation_config' with rules for validation.

        Returns:
            The original data if validation passes.

        Raises:
            NodeValidationError: If the data fails any validation rule.
            TypeError: If the 'validation_config' is malformed.
        """
        validation_config = context.get('validation_config')

        if not validation_config:
            logger.warning(
                f"[{self.node_name}] No 'validation_config' found in context. Skipping validation for data."
            )
            return data

        if not isinstance(validation_config, dict):
            raise TypeError(
                f"[{self.node_name}] 'validation_config' in context must be a dictionary, "
                f"but got {type(validation_config).__name__}."
            )

        logger.debug(
            f"[{self.node_name}] Starting validation with config: {validation_config}"
        )

        try:
            self._validate_data(data, validation_config)
            logger.info(f"[{self.node_name}] Data successfully validated.")
            return data
        except NodeValidationError as e:
            logger.error(
                f"[{self.node_name}] Data validation failed: {e}. "
                f"Input data (partial): {str(data)[:200]}..."
            )
            raise # Re-raise the validation error to stop processing

    def _validate_data(self, data: Any, config: Dict[str, Any], path: str = 'root') -> None:
        """
        Internal method to perform recursive data validation.
        """
        if not isinstance(data, dict):
            raise NodeValidationError(
                f"[{self.node_name}] Expected data at '{path}' to be a dictionary, "
                f"but got {type(data).__name__}."
            )

        required_keys = set(config.get('required_keys', []))
        schema = config.get('schema', {})
        allow_extra_keys = config.get('allow_extra_keys', False)

        # 1. Check for required keys
        missing_keys = required_keys - set(data.keys())
        if missing_keys:
            raise NodeValidationError(
                f"[{self.node_name}] Missing required keys at '{path}': {', '.join(missing_keys)}"
            )

        # 2. Check for extra keys if not allowed
        if not allow_extra_keys:
            extra_keys = set(data.keys()) - (required_keys | set(schema.keys()))
            if extra_keys:
                raise NodeValidationError(
                    f"[{self.node_name}] Unexpected extra keys at '{path}': {', '.join(extra_keys)}"
                )

        # 3. Validate against schema for each field
        for key, field_rules in schema.items():
            field_path = f"{path}.{key}"
            is_present = key in data

            # Handle optional fields with default values if not present
            if not is_present:
                if field_rules.get('default') is not None:
                    data[key] = field_rules['default']
                    logger.debug(f"[{self.node_name}] Set default value for '{field_path}'.")
                elif field_rules.get('required', False):
                    raise NodeValidationError(
                        f"[{self.node_name}] Required field '{field_path}' is missing."
                    )
                else:
                    # Field is optional and not present, continue
                    continue

            value = data[key]
            expected_type = field_rules.get('type')

            # Type check
            if expected_type:
                # Handle Union types correctly (e.g., Union[str, None])
                if get_origin(expected_type) is Union:
                    valid_types = get_args(expected_type)
                    if not isinstance(value, valid_types):
                        raise NodeValidationError(
                            f"[{self.node_name}] Field '{field_path}' expected type "
                            f"{valid_types}, but got {type(value).__name__} with value '{value}'."
                        )
                elif not isinstance(value, expected_type):
                    raise NodeValidationError(
                        f"[{self.node_name}] Field '{field_path}' expected type "
                        f"{expected_type.__name__}, but got {type(value).__name__} with value '{value}'."
                    )

            # Specific rules based on type
            if isinstance(value, str):
                min_length = field_rules.get('min_length')
                max_length = field_rules.get('max_length')
                if min_length is not None and len(value) < min_length:
                    raise NodeValidationError(
                        f"[{self.node_name}] Field '{field_path}' (string) length "
                        f"{len(value)} is less than min_length {min_length}."
                    )
                if max_length is not None and len(value) > max_length:
                    raise NodeValidationError(
                        f"[{self.node_name}] Field '{field_path}' (string) length "
                        f"{len(value)} is greater than max_length {max_length}."
                    )
            elif isinstance(value, (int, float)):
                min_value = field_rules.get('min_value')
                max_value = field_rules.get('max_value')
                if min_value is not None and value < min_value:
                    raise NodeValidationError(
                        f"[{self.node_name}] Field '{field_path}' (numeric) value "
                        f"{value} is less than min_value {min_value}."
                    )
                if max_value is not None and value > max_value:
                    raise NodeValidationError(
                        f"[{self.node_name}] Field '{field_path}' (numeric) value "
                        f"{value} is greater than max_value {max_value}."
                    )
            elif isinstance(value, dict):
                # Recursively validate nested dictionaries
                nested_schema = field_rules.get('schema')
                nested_required_keys = field_rules.get('required_keys', [])
                nested_allow_extra_keys = field_rules.get('allow_extra_keys', allow_extra_keys) # Inherit or override

                if nested_schema or nested_required_keys:
                    nested_config = {
                        "schema": nested_schema,
                        "required_keys": nested_required_keys,
                        "allow_extra_keys": nested_allow_extra_keys
                    }
                    self._validate_data(value, nested_config, field_path)
            elif isinstance(value, list):
                item_schema = field_rules.get('item_schema')
                min_items = field_rules.get('min_items')
                max_items = field_rules.get('max_items')

                if min_items is not None and len(value) < min_items:
                    raise NodeValidationError(
                        f"[{self.node_name}] Field '{field_path}' (list) has {len(value)} items, "
                        f"which is less than min_items {min_items}."
                    )
                if max_items is not None and len(value) > max_items:
                    raise NodeValidationError(
                        f"[{self.node_name}] Field '{field_path}' (list) has {len(value)} items, "
                        f"which is greater than max_items {max_items}."
                    )

                if item_schema:
                    for i, item in enumerate(value):
                        if not isinstance(item, dict):
                             raise NodeValidationError(
                                f"[{self.node_name}] Item {i} in list '{field_path}' expected "
                                f"to be a dictionary for item_schema validation, "
                                f"but got {type(item).__name__}."
                            )
                        self._validate_data(item, {"schema": item_schema, "allow_extra_keys": nested_allow_extra_keys}, f"{field_path}[{i}]")
                        
# Example of how to use this node (for testing/demonstration, not part of actual submission):
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(name)s - %(message)s')

    validator = DataValidatorNode()

    # --- Test Case 1: Valid data ---
    valid_data = {
        "id": "user123",
        "timestamp": 1678886400,
        "payload": {
            "action": "purchase",
            "data": {"item_id": "prodA", "quantity": 1},
            "options": {"async": True}
        },
        "source": "web"
    }
    valid_context = {
        "validation_config": {
            "required_keys": ["id", "timestamp", "payload"],
            "schema": {
                "id": {"type": str, "min_length": 5, "max_length": 10},
                "timestamp": {"type": int, "min_value": 0},
                "payload": {
                    "type": dict,
                    "required_keys": ["action", "data"],
                    "schema": {
                        "action": {"type": str},
                        "data": {"type": dict, "required_keys": ["item_id"]},
                        "options": {"type": dict, "required": False}
                    }
                },
                "source": {"type": str, "required": False, "default": "api"}
            },
            "allow_extra_keys": True
        }
    }
    print("\n--- Test Case 1: Valid Data ---")
    try:
        result = validator.process(valid_data.copy(), valid_context)
        print("Validation successful (expected). Result:", result)
    except NodeValidationError as e:
        print(f"Validation failed (unexpected): {e}")

    # --- Test Case 2: Missing required key ---
    invalid_data_missing_key = valid_data.copy()
    del invalid_data_missing_key["id"]
    print("\n--- Test Case 2: Missing Required Key ---")
    try:
        validator.process(invalid_data_missing_key, valid_context)
        print("Validation successful (unexpected).")
    except NodeValidationError as e:
        print(f"Validation failed (expected): {e}")

    # --- Test Case 3: Incorrect type ---
    invalid_data_type = valid_data.copy()
    invalid_data_type["timestamp"] = "not_an_int"
    print("\n--- Test Case 3: Incorrect Type ---")
    try:
        validator.process(invalid_data_type, valid_context)
        print("Validation successful (unexpected).")
    except NodeValidationError as e:
        print(f"Validation failed (expected): {e}")

    # --- Test Case 4: Nested validation failure (missing required nested key) ---
    invalid_data_nested = valid_data.copy()
    del invalid_data_nested["payload"]["data"]["item_id"]
    print("\n--- Test Case 4: Nested Validation Failure ---")
    try:
        validator.process(invalid_data_nested, valid_context)
        print("Validation successful (unexpected).")
    except NodeValidationError as e:
        print(f"Validation failed (expected): {e}")

    # --- Test Case 5: String length validation failure ---
    invalid_data_str_len = valid_data.copy()
    invalid_data_str_len["id"] = "short"
    print("\n--- Test Case 5: String Length Failure ---")
    try:
        validator.process(invalid_data_str_len, valid_context)
        print("Validation successful (unexpected).")
    except NodeValidationError as e:
        print(f"Validation failed (expected): {e}")

    # --- Test Case 6: Numeric range validation failure ---
    invalid_data_num_range = valid_data.copy()
    invalid_data_num_range["timestamp"] = -100
    print("\n--- Test Case 6: Numeric Range Failure ---")
    try:
        validator.process(invalid_data_num_range, valid_context)
        print("Validation successful (unexpected).")
    except NodeValidationError as e:
        print(f"Validation failed (expected): {e}")

    # --- Test Case 7: Allow extra keys = False ---
    invalid_data_extra_keys = valid_data.copy()
    invalid_data_extra_keys["unexpected_field"] = "value"
    context_no_extra = valid_context.copy()
    context_no_extra['validation_config']['allow_extra_keys'] = False
    print("\n--- Test Case 7: Disallow Extra Keys ---")
    try:
        validator.process(invalid_data_extra_keys, context_no_extra)
        print("Validation successful (unexpected).")
    except NodeValidationError as e:
        print(f"Validation failed (expected): {e}")
        
    # --- Test Case 8: No validation config in context ---
    print("\n--- Test Case 8: No validation config ---")
    try:
        result = validator.process({"test": 123}, {})
        print("Validation successful (expected, no config). Result:", result)
    except NodeValidationError as e:
        print(f"Validation failed (unexpected): {e}")

    # --- Test Case 9: Union type validation ---
    union_data = {"id": "test", "value": None}
    union_context = {
        "validation_config": {
            "required_keys": ["id", "value"],
            "schema": {
                "id": {"type": str},
                "value": {"type": Union[str, None]}
            }
        }
    }
    print("\n--- Test Case 9: Union Type Valid (None) ---")
    try:
        result = validator.process(union_data.copy(), union_context)
        print("Validation successful (expected). Result:", result)
    except NodeValidationError as e:
        print(f"Validation failed (unexpected): {e}")

    union_data_str = {"id": "test", "value": "some_string"}
    print("\n--- Test Case 9: Union Type Valid (str) ---")
    try:
        result = validator.process(union_data_str.copy(), union_context)
        print("Validation successful (expected). Result:", result)
    except NodeValidationError as e:
        print(f"Validation failed (unexpected): {e}")

    union_data_invalid = {"id": "test", "value": 123}
    print("\n--- Test Case 9: Union Type Invalid ---")
    try:
        result = validator.process(union_data_invalid.copy(), union_context)
        print("Validation successful (unexpected).")
    except NodeValidationError as e:
        print(f"Validation failed (expected): {e}")

    # --- Test Case 10: List validation ---
    list_data = {
        "items": [
            {"name": "item1", "qty": 10},
            {"name": "item2", "qty": 20}
        ]
    }
    list_context = {
        "validation_config": {
            "schema": {
                "items": {
                    "type": list,
                    "min_items": 1,
                    "max_items": 3,
                    "item_schema": {
                        "name": {"type": str, "min_length": 4},
                        "qty": {"type": int, "min_value": 1}
                    }
                }
            }
        }
    }
    print("\n--- Test Case 10: List Valid ---")
    try:
        result = validator.process(list_data.copy(), list_context)
        print("Validation successful (expected). Result:", result)
    except NodeValidationError as e:
        print(f"Validation failed (unexpected): {e}")

    invalid_list_data_item = {
        "items": [
            {"name": "it1", "qty": 10} # name too short
        ]
    }
    print("\n--- Test Case 10: List Invalid Item ---")
    try:
        result = validator.process(invalid_list_data_item.copy(), list_context)
        print("Validation successful (unexpected).")
    except NodeValidationError as e:
        print(f"Validation failed (expected): {e}")

    invalid_list_data_count = {
        "items": [] # too few items
    }
    print("\n--- Test Case 10: List Invalid Count ---")
    try:
        result = validator.process(invalid_list_data_count.copy(), list_context)
        print("Validation successful (unexpected).")
    except NodeValidationError as e:
        print(f"Validation failed (expected): {e}")

    # --- Test Case 11: Field with default ---
    data_with_default = {"id": "test", "value": "custom"}
    context_with_default = {
        "validation_config": {
            "required_keys": ["id"],
            "schema": {
                "id": {"type": str},
                "value": {"type": str, "default": "default_value"}
            }
        }
    }
    print("\n--- Test Case 11: Field with default (value present) ---")
    try:
        result = validator.process(data_with_default.copy(), context_with_default)
        print("Validation successful (expected). Result:", result)
        assert result["value"] == "custom"
    except NodeValidationError as e:
        print(f"Validation failed (unexpected): {e}")

    data_missing_default = {"id": "test"}
    print("\n--- Test Case 11: Field with default (value missing) ---")
    try:
        result = validator.process(data_missing_default.copy(), context_with_default)
        print("Validation successful (expected). Result:", result)
        assert result["value"] == "default_value"
    except NodeValidationError as e:
        print(f"Validation failed (unexpected): {e}")

    # --- Test Case 12: Field required but missing (no default) ---
    context_required_no_default = {
        "validation_config": {
            "required_keys": ["id"],
            "schema": {
                "id": {"type": str},
                "value": {"type": str, "required": True} # Missing 'value' and no default
            }
        }
    }
    print("\n--- Test Case 12: Required field missing (no default) ---")
    try:
        result = validator.process(data_missing_default.copy(), context_required_no_default)
        print("Validation successful (unexpected).")
    except NodeValidationError as e:
        print(f"Validation failed (expected): {e}")