import os
import importlib.util
import inspect
import logging
from typing import List, Dict, Any
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class VishustraEngine:
    """
    Core execution engine for the Vishustra Orchestrator.
    Dynamically loads node modules and manages the data pipeline.
    """
    def __init__(self) -> None:
        self.nodes: List[BaseNode] = []
        self._load_nodes()

    def _load_nodes(self) -> None:
        nodes_dir = os.path.join(os.path.dirname(__file__), "nodes")
        if not os.path.exists(nodes_dir):
            logger.warning(f"Nodes directory not found at {nodes_dir}")
            return

        for filename in os.listdir(nodes_dir):
            if filename.endswith(".py") and filename not in ("__init__.py", "base_node.py"):
                module_name = f"vishustra_core.nodes.{filename[:-3]}"
                
                try:
                    spec = importlib.util.spec_from_file_location(module_name, os.path.join(nodes_dir, filename))
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        
                        for name, obj in inspect.getmembers(module, inspect.isclass):
                            if issubclass(obj, BaseNode) and obj is not BaseNode:
                                self.nodes.append(obj())
                except Exception as e:
                    logger.error(f"Failed to load module {filename}: {e}")
                            
    def get_active_nodes(self) -> List[str]:
        return [node.node_name for node in self.nodes]

    def run_simulation(self, initial_data: str) -> None:
        logger.info("Starting Vishustra Pipeline Simulation")
        logger.debug(f"Initial Data: '{initial_data}'")
        
        current_data = initial_data
        context: Dict[str, Any] = {}
        
        pipeline = self.nodes[:5] 
        
        if not pipeline:
            logger.warning("No nodes available to process data.")
            return

        for node in pipeline:
            try:
                logger.info(f"Routing to Node: {node.node_name}")
                current_data = node.process(current_data, context)
                logger.debug(f"Result: {current_data}")
            except Exception as e:
                logger.error(f"Node {node.node_name} failed: {e}")
                
        logger.info("Simulation Complete")
