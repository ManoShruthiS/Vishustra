import logging
from typing import Any, Dict, Type, Callable, Union, Tuple

# Assuming vishustra_core is installed and available
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ValidationError(ValueError):
    """Custom exception raised for validation failures within the DataValidatorNode."""
    pass

class DataValidatorNode(BaseNode):
    """
    A Vishustra processing node that validates incoming data against predefined rules.

    This node is designed to ensure data integrity by checking for expected data types,
    the presence of required fields, and the application of custom validation functions.
    It offers configurable behavior to either log warnings for validation failures
    or to halt processing by raising a `ValidationError`.

    Validation rules are specified as a dictionary where keys correspond to the
    expected data fields, and values define the validation criteria. Criteria can be:
    1.  A Python type (e.g., `str`, `int`, `list`) to check using `isinstance()`.
    2.  A tuple of Python types (e.g., `(str, type(None))` to allow `str` or `None`).
    3.  A callable (function or lambda) that accepts the data value as its single
        argument and returns `True` if the value is valid, `False` otherwise.

    Example validation rules:
    ```python
    rules = {
        "user_id": int,
        "username": str,
        "email": lambda x: isinstance(x, str) and "@" in x and "." in x,
        "message_content": (str, type(None)), # Allows string or None
        "timestamp": float
    }
    ```
    """

    def __init__(
        self,
        validation_rules: Dict[str, Union[Type, Callable[[Any], bool], Tuple[Type, ...]]],
        raise_on_error: bool = False
    ):
        """
        Initializes the DataValidatorNode with a set of validation rules.

        Args:
            validation_rules: A dictionary where keys are data field names and values
                              are the validation criteria (type, tuple of types, or callable).
            raise_on_error: If True, a `ValidationError` will be raised upon the first
                            validation failure. If False, validation failures will be
                            logged as warnings, and the data will continue to pass through.
        
        Raises:
            TypeError: If `validation_rules` is not a dictionary.
        """
        if not isinstance(validation_rules, dict):
            raise TypeError("`validation_rules` must be a dictionary.")
        
        self._validation_rules = validation_rules
        self._raise_on_error = raise_on_error
        logger.debug(
            f"[{self.node_name}] Initialized with {len(self._validation_rules)} rules. "
            f"Raise on error: {self._raise_on_error}."
        )

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this node."""
        return "DataValidator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Validates the incoming data against the configured rules.

        If `data` is not a dictionary, and key-based validation rules are present,
        a warning will be logged (or an error raised if `raise_on_error` is True),
        and the data will be passed through without key-based validation.

        Args:
            data: The data payload to be validated. Typically expected to be a dictionary.
            context: The operational context for the current processing flow.

        Returns:
            The original `data` if validation passes, or if `raise_on_error` is False
            and warnings were logged.

        Raises:
            ValidationError: If `raise_on_error` is True and any validation rule fails.
            TypeError: If a validation rule itself is malformed (e.g., neither a type,
                       tuple of types, nor a callable).
        """
        logger.info(f"[{self.node_name}] Starting data validation for incoming payload.")
        
        # If data is not a dict, we cannot apply key-based rules.
        # This check prevents errors if `data` is, for instance, a string or list.
        if not isinstance(data, Dict):
            error_msg = (
                f"[{self.node_name}] Input data is not a dictionary ({type(data).__name__}). "
                "Cannot apply key-based validation rules. Skipping validation."
            )
            if self._raise_on_error:
                logger.error(error_msg)
                raise ValidationError(error_msg)
            else:
                logger.warning(error_msg)
                return data

        validation_issues = []

        for key, rule in self._validation_rules.items():
            value_to_validate = data.get(key)
            is_key_present = key in data
            is_valid_for_rule = False
            issue_detail = ""

            # Handle missing keys explicitly
            if not is_key_present:
                # If a key is in rules but not in data, it's a validation failure.
                # Currently, all keys in `_validation_rules` are treated as required.
                # Future enhancements might include an 'optional' flag.
                issue_detail = f"Missing required key '{key}' in data."
                logger.warning(f"[{self.node_name}] {issue_detail}")
                validation_issues.append(issue_detail)
                continue # Move to the next rule

            try:
                if isinstance(rule, (Type, Tuple)):  # Rule is a type or a tuple of types
                    if isinstance(value_to_validate, rule):
                        is_valid_for_rule = True
                    else:
                        expected_types = rule if isinstance(rule, tuple) else (rule,)
                        issue_detail = (
                            f"Expected type(s) {', '.join(t.__name__ for t in expected_types)}, "
                            f"got '{type(value_to_validate).__name__}'."
                        )
                elif callable(rule):  # Rule is a custom validation function
                    if rule(value_to_validate):
                        is_valid_for_rule = True
                    else:
                        issue_detail = "Custom validation function returned False."
                else:
                    # The rule itself is malformed (not a type, tuple of types, or callable)
                    err_msg = (
                        f"Invalid validation rule type for key '{key}'. "
                        f"Expected Type, Tuple[Type, ...], or Callable, "
                        f"got '{type(rule).__name__}'."
                    )
                    logger.critical(f"[{self.node_name}] {err_msg}")
                    # This indicates a configuration error in the node itself, not just data.
                    if self._raise_on_error:
                        raise TypeError(err_msg)
                    else:
                        # If not raising, treat this misconfiguration as a validation issue.
                        validation_issues.append(f"Misconfigured rule for key '{key}': {err_msg}")
                        continue # Move to next rule, don't attempt further processing for this one.

            except Exception as e:
                # Catch any unexpected errors during rule execution (e.g., custom callable fails)
                issue_detail = (
                    f"An unexpected error occurred during validation for key '{key}' "
                    f"with value '{value_to_validate}': {type(e).__name__}: {e}"
                )
                logger.error(f"[{self.node_name}] {issue_detail}", exc_info=True)
                validation_issues.append(issue_detail)
                # An exception during rule application means validation failed for this key.

            if not is_valid_for_rule:
                final_issue = (
                    f"Validation failed for key '{key}' "
                    f"(value: '{value_to_validate}' of type '{type(value_to_validate).__name__}'). "
                    f"{issue_detail}"
                )
                logger.warning(f"[{self.node_name}] {final_issue}")
                validation_issues.append(final_issue)

        if validation_issues:
            combined_error_msg = (
                f"[{self.node_name}] Data validation detected {len(validation_issues)} issue(s)."
            )
            if self._raise_on_error:
                logger.error(f"{combined_error_msg}\nDetails:\n - " + "\n - ".join(validation_issues))
                raise ValidationError(combined_error_msg)
            else:
                logger.warning(
                    f"{combined_error_msg} Data will proceed despite warnings. "
                    "Set `raise_on_error=True` to halt processing on failures."
                )
        else:
            logger.info(f"[{self.node_name}] Data validated successfully.")

        return data