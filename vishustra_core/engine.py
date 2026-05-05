import os
import importlib.util
import inspect
from vishustra_core.nodes.base_node import BaseNode

class VishustraEngine:
    def __init__(self):
        self.nodes = []
        self._load_nodes()

    def _load_nodes(self):
        nodes_dir = os.path.join(os.path.dirname(__file__), "nodes")
        if not os.path.exists(nodes_dir):
            return

        for filename in os.listdir(nodes_dir):
            if filename.endswith(".py") and filename != "__init__.py" and filename != "base_node.py":
                module_name = f"vishustra_core.nodes.{filename[:-3]}"
                
                try:
                    # Load module dynamically
                    spec = importlib.util.spec_from_file_location(module_name, os.path.join(nodes_dir, filename))
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        
                        # Find classes that inherit from BaseNode
                        for name, obj in inspect.getmembers(module, inspect.isclass):
                            if issubclass(obj, BaseNode) and obj is not BaseNode:
                                self.nodes.append(obj())
                except Exception as e:
                    print(f"[Engine] Warning: Failed to load module {filename} - {e}")
                            
    def get_active_nodes(self):
        return [node.node_name for node in self.nodes]

    def run_simulation(self, initial_data: str):
        print(f"\n--- Starting Vishustra Pipeline Simulation ---")
        print(f"Initial Data: '{initial_data}'")
        
        current_data = initial_data
        context = {}
        
        # We simulate passing data through up to 5 nodes randomly or sequentially
        pipeline = self.nodes[:5] 
        
        if not pipeline:
            print("No nodes available to process data.")
            return

        for node in pipeline:
            try:
                print(f"-> Routing to Node: {node.node_name}")
                current_data = node.process(current_data, context)
                print(f"   Result: {current_data}")
            except Exception as e:
                print(f"   [Error] Node {node.node_name} failed: {e}")
                
        print(f"--- Simulation Complete ---\n")
