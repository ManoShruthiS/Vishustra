import re
import logging
from typing import Any, Dict, Union, List, Pattern, Tuple, Literal, Optional

# Assuming BaseNode is available at this path as per Vishustra framework structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class RegexMatcherNode(BaseNode):
    """
    A Vishustra processing node that performs regular expression matching
    on input data.

    This node can be configured to return different types of results:
    - The first full string match (`'first_str'`).
    - All full string matches (`'all_strs'`).
    - The first tuple of captured groups (`'first_groups'`).
    - A list of tuples of captured groups for all matches (`'all_groups'`).
    - The first `re.Match` object (`'first_obj'`).
    - A list of all `re.Match` objects (`'all_objs'`).

    Configuration Parameters in `__init__`:
    - `pattern` (Union[str, Pattern]): The regular expression pattern to use.
                                       Can be a string or a pre-compiled regex object.
    - `flags` (int): Optional regex flags, e.g., `re.IGNORECASE`, `re.MULTILINE`.
                     Defaults to 0.
    - `output_type` (Literal): Defines the format of the output.
                               Options: `'first_str'`, `'all_strs'`, `'first_groups'`,
                               `'all_groups'`, `'first_obj'`, `'all_objs'`.
                               Defaults to `'first_str'`.
    """

    def __init__(
        self,
        pattern: Union[str, Pattern],
        flags: int = 0,
        output_type: Literal[
            'first_str', 'all_strs', 'first_groups',
            'all_groups', 'first_obj', 'all_objs'
        ] = 'first_str'
    ):
        """
        Initializes the RegexMatcherNode with a regex pattern and output configuration.

        Args:
            pattern: The regular expression pattern as a string or a compiled `re.Pattern` object.
            flags: Optional regex flags (e.g., `re.IGNORECASE`, `re.MULTILINE`). Defaults to 0.
            output_type: Specifies the desired output format for the `process` method.
        
        Raises:
            ValueError: If the provided regex pattern is invalid or `output_type` is not recognized.
            TypeError: If the `pattern` argument is not a string or `re.Pattern` object.
        """
        if isinstance(pattern, str):
            try:
                self._compiled_pattern: Pattern = re.compile(pattern, flags)
                logger.debug(f"[{self.__class__.__name__}] Compiled regex pattern: '{pattern}' with flags: {flags}")
            except re.error as e:
                logger.exception(f"[{self.__class__.__name__}] Failed to compile regex pattern '{pattern}': {e}")
                raise ValueError(f"Invalid regex pattern provided: {e}") from e
        elif isinstance(pattern, Pattern):
            self._compiled_pattern = pattern
            logger.debug(f"[{self.__class__.__name__}] Using pre-compiled regex pattern: '{pattern.pattern}'")
        else:
            raise TypeError(f"Pattern must be a string or a compiled re.Pattern, got {type(pattern)}")

        valid_output_types = {
            'first_str', 'all_strs', 'first_groups',
            'all_groups', 'first_obj', 'all_objs'
        }
        if output_type not in valid_output_types:
            raise ValueError(f"Invalid output_type: '{output_type}'. "
                             f"Must be one of {sorted(list(valid_output_types))}.")
        self._output_type = output_type
        logger.debug(f"[{self.__class__.__name__}] Initialized with output_type: '{self._output_type}'")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "RegexMatcher"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by applying the configured regex pattern.

        Args:
            data: The input data to be matched against the regex. Expected to be
                  string-like or convertible to a string.
            context: A dictionary containing contextual information for the node.
                     (This node does not directly use `context` for processing logic,
                     but it is available for general framework needs or future extensions).

        Returns:
            Any: The result of the regex matching based on the `output_type`
                 configured during initialization. Possible return types:
                 - `Optional[str]`: for `'first_str'`
                 - `List[str]`: for `'all_strs'`
                 - `Optional[Tuple[str, ...]]`: for `'first_groups'`
                 - `List[Tuple[str, ...]]`: for `'all_groups'`
                 - `Optional[re.Match]`: for `'first_obj'`
                 - `List[re.Match]`: for `'all_objs'`
                 Returns `None` or an empty list/tuple if no matches are found,
                 depending on the `output_type`.

        Raises:
            TypeError: If the input `data` cannot be converted to a string.
        """
        logger.debug(f"[{self.node_name}] Attempting to process data (type: {type(data)}) with output_type: {self._output_type}")

        try:
            text = str(data)
        except Exception as e:
            logger.error(f"[{self.node_name}] Input data could not be converted to string. Type: {type(data)}, Error: {e}")
            raise TypeError(f"Input data must be convertible to string, but got type {type(data)}") from e

        results: Any = None

        if self._output_type.startswith('first_'):
            match: Optional[re.Match] = self._compiled_pattern.search(text)
            if match:
                if self._output_type == 'first_str':
                    results = match.group(0)
                elif self._output_type == 'first_groups':
                    results = match.groups()
                elif self._output_type == 'first_obj':
                    results = match
            else:
                logger.debug(f"[{self.node_name}] No first match found for pattern '{self._compiled_pattern.pattern}'.")
        else:  # Output type specifies all matches
            if self._output_type == 'all_objs':
                matches: List[re.Match] = list(self._compiled_pattern.finditer(text))
                results = matches
                logger.debug(f"[{self.node_name}] Found {len(matches)} match objects.")
            elif self._output_type == 'all_strs':
                matches_str: List[str] = [m.group(0) for m in self._compiled_pattern.finditer(text)]
                results = matches_str
                logger.debug(f"[{self.node_name}] Found {len(matches_str)} full string matches.")
            elif self._output_type == 'all_groups':
                # re.findall is suitable here; if pattern has capturing groups, it returns a list of tuples.
                # If no capturing groups, it effectively returns a list of full matches (strings).
                # This behavior is generally robust for 'all_groups'.
                matches_groups: List[Tuple[str, ...]] = list(self._compiled_pattern.findall(text))
                results = matches_groups
                logger.debug(f"[{self.node_name}] Found {len(matches_groups)} group match tuples.")
            
            if not results:
                logger.debug(f"[{self.node_name}] No matches found for pattern '{self._compiled_pattern.pattern}'. Returning empty list.")

        logger.info(f"[{self.node_name}] Successfully processed data with output type: {self._output_type}. Result: {results!r}")
        return results