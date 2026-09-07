import logging
import re
from typing import Any, Dict, Callable

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class DataValidatorNode(BaseNode):
    """
    A Vishustra node for robustly validating input data against a predefined schema.

    This node ensures that incoming data conforms to structural, type, and value
    constraints, preventing malformed data from proceeding through the orchestration
    pipeline. It supports a comprehensive set of rules including required fields,
    type checking, length constraints for strings/collections, numerical range
    checks, regular expression pattern matching, and custom validation functions.
    """

    def __init__(self, schema: Dict[str, Dict[str, Any]]):
        """
        Initializes the DataValidatorNode with a validation schema.

        The schema is a dictionary where each key represents an expected field
        in the input data, and its value is another dictionary defining validation
        rules for that field.

        Supported validation rules for each field:
        - 'type': (Python Type) The expected Python type (e.g., str, int, bool, list, dict).
        - 'required': (bool) If True, the field must be present in the data. Defaults to False.
        - 'min_length': (int) Minimum length for string or collection types (e.g., list, dict).
        - 'max_length': (int) Maximum length for string or collection types.
        - 'min_value': (int/float) Minimum numerical value for int or float types.
        - 'max_value': (int/float) Maximum numerical value for int or float types.
        - 'pattern': (str) A regular expression pattern (e.g., r"^\\w+$") that the string
                     value must fully match.
        - 'validator': (Callable[[Any], bool]) A custom callable function that takes
                       the field's value as input and returns True if valid, False otherwise.
                       This rule is executed last if present.

        Args:
            schema: A dictionary defining the validation rules for the input data.
                    Example schema structure:
                    ```python
                    {
                        "user_id": {"type": int, "required": True},
                        "user_name": {"type": str, "required": True, "min_length": 3, "max_length": 50},
                        "email": {"type": str, "pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"},
                        "age": {"type": int, "min_value": 0, "max_value": 120, "required": False},
                        "is_active": {"type": bool}
                    }
                    ```
        Raises:
            TypeError: If the provided `schema` is not a dictionary.
        """
        if not isinstance(schema, dict):
            logger.error(f"[{self.node_name}] Schema provided to DataValidatorNode must be a dictionary.")
            raise TypeError("Schema must be a dictionary.")

        self._schema = schema
        logger.info(f"[{self.node_name}] Initialized with schema defining {len(self._schema)} fields.")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "DataValidator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Validates the input `data` against the schema configured during initialization.

        The method iterates through each field defined in the schema and applies
        the corresponding validation rules. If any rule fails, a ValueError
        is raised, stopping further processing and signaling an invalid data state.

        Args:
            data: The input data to be validated. Expected to be a dictionary.
            context: A dictionary containing contextual information relevant to the
                     current orchestration run (not directly used by this node).

        Returns:
            The original `data` dictionary if all validation checks pass.
            (Note: This node primarily validates; it does not transform the data,
            though future enhancements could include default value assignment).

        Raises:
            TypeError: If the input `data` is not a dictionary.
            ValueError: If the `data` fails any of the validation rules defined in the schema.
        """
        logger.debug(f"[{self.node_name}] Starting validation for incoming data.")

        if not isinstance(data, dict):
            error_msg = (f"[{self.node_name}] Invalid input type. Data must be a dictionary, "
                         f"but received type: {type(data).__name__}.")
            logger.error(error_msg)
            raise TypeError(error_msg)

        # Create a shallow copy to prevent accidental modification of the original data
        processed_data = data.copy()

        for field_name, rules in self._schema.items():
            field_value = processed_data.get(field_name)
            is_present = field_name in processed_data

            # Rule 1: 'required' field check
            required = rules.get('required', False)
            if required and not is_present:
                error_msg = (f"[{self.node_name}] Validation failed: Required field "
                             f"'{field_name}' is missing.")
                logger.warning(error_msg)
                raise ValueError(error_msg)

            # If field is not present and not required, skip further checks for this field.
            if not is_present and not required:
                logger.debug(f"[{self.node_name}] Field '{field_name}' not present and not required. Skipping.")
                continue

            # Rule 2: 'type' check
            expected_type = rules.get('type')
            if expected_type is not None and not isinstance(field_value, expected_type):
                error_msg = (f"[{self.node_name}] Validation failed for field '{field_name}': "
                             f"Expected type '{expected_type.__name__}', got '{type(field_value).__name__}'.")
                logger.warning(error_msg)
                raise ValueError(error_msg)

            # Apply value-specific validations only if the field is present and its type is correct (if specified)
            if is_present:
                # Rule 3: 'min_length' and 'max_length' for sized types
                if isinstance(field_value, (str, list, dict)): # Apply to strings, lists, dictionaries
                    current_length = len(field_value)
                    min_len = rules.get('min_length')
                    if min_len is not None and current_length < min_len:
                        error_msg = (f"[{self.node_name}] Validation failed for field '{field_name}': "
                                     f"Length {current_length} is less than minimum required {min_len}.")
                        logger.warning(error_msg)
                        raise ValueError(error_msg)

                    max_len = rules.get('max_length')
                    if max_len is not None and current_length > max_len:
                        error_msg = (f"[{self.node_name}] Validation failed for field '{field_name}': "
                                     f"Length {current_length} is greater than maximum allowed {max_len}.")
                        logger.warning(error_msg)
                        raise ValueError(error_msg)

                # Rule 4: 'min_value' and 'max_value' for numeric types
                if isinstance(field_value, (int, float)):
                    min_val = rules.get('min_value')
                    if min_val is not None and field_value < min_val:
                        error_msg = (f"[{self.node_name}] Validation failed for field '{field_name}': "
                                     f"Value {field_value} is less than minimum allowed {min_val}.")
                        logger.warning(error_msg)
                        raise ValueError(error_msg)

                    max_val = rules.get('max_value')
                    if max_val is not None and field_value > max_val:
                        error_msg = (f"[{self.node_name}] Validation failed for field '{field_name}': "
                                     f"Value {field_value} is greater than maximum allowed {max_val}.")
                        logger.warning(error_msg)
                        raise ValueError(error_msg)

                # Rule 5: 'pattern' (regex) for string types
                pattern = rules.get('pattern')
                if pattern is not None and isinstance(field_value, str):
                    if not re.fullmatch(pattern, field_value):
                        error_msg = (f"[{self.node_name}] Validation failed for field '{field_name}': "
                                     f"Value '{field_value}' does not match regex pattern '{pattern}'.")
                        logger.warning(error_msg)
                        raise ValueError(error_msg)

                # Rule 6: 'validator' (custom callable function)
                custom_validator = rules.get('validator')
                if custom_validator is not None:
                    if not isinstance(custom_validator, Callable):
                        # This should ideally be caught during schema initialization
                        # but as a failsafe, we check here.
                        error_msg = (f"[{self.node_name}] Configuration error: 'validator' rule for "
                                     f"field '{field_name}' must be a callable function, "
                                     f"got {type(custom_validator).__name__}.")
                        logger.error(error_msg)
                        raise ValueError(error_msg)
                    try:
                        if not custom_validator(field_value):
                            error_msg = (f"[{self.node_name}] Validation failed for field '{field_name}': "
                                         f"Custom validator returned False for value '{field_value}'.")
                            logger.warning(error_msg)
                            raise ValueError(error_msg)
                    except Exception as e:
                        error_msg = (f"[{self.node_name}] Custom validator for field '{field_name}' "
                                     f"raised an unhandled exception: {e.__class__.__name__}: {e}")
                        logger.error(error_msg, exc_info=True)
                        raise ValueError(error_msg) from e # Re-raise as ValueError with original exception context

        logger.info(f"[{self.node_name}] Data successfully validated. All schema checks passed.")
        return processed_data