from typing import Any, Dict
from vishustra_core.nodes.base_node import BaseNode

class ExampleNode(BaseNode):
    @property
    def node_name(self) -> str:
        return "Example Processor Node"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        print("[ExampleNode] Processing data...")
        return f"{data} -> [Processed by ExampleNode]"
