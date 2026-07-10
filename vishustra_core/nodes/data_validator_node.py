import logging
import re
from typing import Any, Dict, List, Type

# Assuming BaseNode is located in vishustra_core/nodes/base_node relative to the project root
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class DataValidationException(ValueError):
    """Custom exception raised when input data fails against specified validation rules."""
    def __init__(self, message: str, field: str = None, rule: Dict[str, Any] = None, data_value: Any = None):
        super().__init__(message)
        self.field = field
        self.rule = rule
        self.data_value = data_value

class DataValidatorNode(BaseNode):
    """
    A Vishustra processing node responsible for validating input data against
    a set of predefined schema rules.

    Validation rules are expected to be provided in the `context` dictionary
    under the key 'validation_rules'. Each rule is a dictionary specifying checks
    for a particular field within the input data.

    Example `validation_rules` structure:
    ```python
    validation_rules = [
        {"field": "user_id", "type": "str", "required": True, "min_length": 5, "max_length": 50},
        {"field": "age", "type": "int", "required": True, "min_value": 0, "max_value": 120},
        {"field": "email", "type": "str", "required": False, "format": "email"},
        {"field": "tags", "type": "list", "items_type": "str", "max_items": 10},
        {"field": "is_active", "type": "bool", "required": True, "allow_none": False}
    ]
    ```
    Supported rule properties for each field:
    - `field` (str): The name of the field to validate (mandatory for each rule).
    - `required` (bool): If `True`, the field must be present in the data. Default `False`.
    - `allow_none` (bool): If `True` and `required` is `True`, the field can be `None`. Default `False`.
    - `type` (str): Expected data type (e.g., "str", "int", "float", "bool", "list", "dict").
    - `min_length` (int): Minimum length for strings, lists, or dictionaries.
    - `max_length` (int): Maximum length for strings, lists, or dictionaries.
    - `min_value` (Union[int, float]): Minimum numerical value.
    - `max_value` (Union[int, float]): Maximum numerical value.
    - `items_type` (str): For lists, the expected type of items within the list.
    - `format` (str): Special format checks (e.g., "email").

    If no validation rules are provided in the context, the node passes the data
    through without performing any checks.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "Data Validator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by validating it against rules defined in the context.

        Args:
            data: The data to be validated. Expected to be a dictionary if rules are provided.
            context: A dictionary which *may* contain 'validation_rules' (List[Dict[str, Any]]).

        Returns:
            The original, validated data if all checks pass.

        Raises:
            TypeError: If `data` is not a dictionary when validation rules are supplied.
            DataValidationException: If any part of the data fails validation.
        """
        validation_rules: List[Dict[str, Any]] = context.get("validation_rules", [])

        if not validation_rules:
            logger.info("No validation rules provided in context. Data passed without validation.")
            return data

        if not isinstance(data, dict):
            raise TypeError(
                f"DataValidatorNode expects input `data` to be a dictionary when validation rules "
                f"are present, but received type: {type(data).__name__}"
            )

        logger.debug(f"Starting data validation for input with keys: {list(data.keys()) if data else []}")

        for rule in validation_rules:
            field_name = rule.get("field")
            if not field_name:
                logger.warning(f"Validation rule is missing 'field' key: {rule}. Skipping this rule.")
                continue

            field_value = data.get(field_name)
            is_present = field_name in data
            is_required = rule.get("required", False)
            allow_none = rule.get("allow_none", False)

            # 1. Required field check
            if is_required:
                if not is_present:
                    raise DataValidationException(
                        f"Required field '{field_name}' is missing.",
                        field=field_name, rule=rule
                    )
                if field_value is None and not allow_none:
                    raise DataValidationException(
                        f"Required field '{field_name}' is None, but 'allow_none' is False.",
                        field=field_name, rule=rule, data_value=field_value
                    )

            # If the field is not present and not required, or is None and allowed to be,
            # skip further checks for this field.
            if not is_present or (field_value is None and allow_none):
                continue

            # 2. Type check
            expected_type_str: str = rule.get("type")
            if expected_type_str:
                expected_python_type: Type[Any] = self._get_python_type(expected_type_str, field_name)

                if expected_python_type is not Any and not isinstance(field_value, expected_python_type):
                    raise DataValidationException(
                        f"Field '{field_name}' expected type '{expected_type_str}' but got '{type(field_value).__name__}'.",
                        field=field_name, rule=rule, data_value=field_value
                    )
                
                # Special handling for list items type
                if expected_python_type is list and isinstance(field_value, list) and rule.get("items_type"):
                    expected_items_type_str = rule["items_type"]
                    expected_items_python_type: Type[Any] = self._get_python_type(
                        expected_items_type_str, f"{field_name} (list item)"
                    )

                    if expected_items_python_type is not Any:
                        for i, item in enumerate(field_value):
                            if not isinstance(item, expected_items_python_type):
                                raise DataValidationException(
                                    f"Field '{field_name}' (list) item at index {i} expected type "
                                    f"'{expected_items_type_str}' but got '{type(item).__name__}'.",
                                    field=field_name, rule=rule, data_value=item
                                )

            # 3. Length/Size checks (for strings, lists, dicts)
            if isinstance(field_value, (str, list, dict)):
                length = len(field_value)
                min_len = rule.get("min_length") or rule.get("min_items") or rule.get("min_properties")
                max_len = rule.get("max_length") or rule.get("max_items") or rule.get("max_properties")

                if min_len is not None and length < min_len:
                    raise DataValidationException(
                        f"Field '{field_name}' length ({length}) is less than minimum required length ({min_len}).",
                        field=field_name, rule=rule, data_value=field_value
                    )
                if max_len is not None and length > max_len:
                    raise DataValidationException(
                        f"Field '{field_name}' length ({length}) is greater than maximum allowed length ({max_len}).",
                        field=field_name, rule=rule, data_value=field_value
                    )
            
            # 4. Value range checks (for numbers)
            if isinstance(field_value, (int, float)):
                min_val = rule.get("min_value")
                max_val = rule.get("max_value")

                if min_val is not None and field_value < min_val:
                    raise DataValidationException(
                        f"Field '{field_name}' value ({field_value}) is less than minimum allowed value ({min_val}).",
                        field=field_name, rule=rule, data_value=field_value
                    )
                if max_val is not None and field_value > max_val:
                    raise DataValidationException(
                        f"Field '{field_name}' value ({field_value}) is greater than maximum allowed value ({max_val}).",
                        field=field_name, rule=rule, data_value=field_value
                    )
            
            # 5. Format check (e.g., "email")
            if rule.get("format") == "email":
                if not isinstance(field_value, str):
                     raise DataValidationException(
                        f"Field '{field_name}' expected string for 'email' format validation but got '{type(field_value).__name__}'.",
                        field=field_name, rule=rule, data_value=field_value
                    )
                # Basic email format validation regex
                if not re.match(r"[^@]+@[^@]+\.[^@]+", field_value):
                    raise DataValidationException(
                        f"Field '{field_name}' has an invalid email format: '{field_value}'.",
                        field=field_name, rule=rule, data_value=field_value
                    )

        logger.debug("Data validation completed successfully.")
        return data

    def _get_python_type(self, type_str: str, field_identifier: str) -> Type[Any]:
        """Maps a string type name to a Python type object."""
        type_map: Dict[str, Type[Any]] = {
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "any": Any # Explicitly allow 'any' as a type, effectively skipping strict type checks
        }
        
        python_type = type_map.get(type_str.lower())
        if python_type is None:
            logger.warning(
                f"Unsupported type string '{type_str}' for field '{field_identifier}'. "
                f"Validation will skip this type check."
            )
            return Any # Fallback to Any if type string is not recognized
        return python_type
