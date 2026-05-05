import time
from vishustra_core.engine import VishustraEngine

def run_dashboard():
    print("=================================================")
    print("      VISHUSTRA AI ORCHESTRATOR - DASHBOARD      ")
    print("=================================================")
    
    engine = VishustraEngine()
    active_nodes = engine.get_active_nodes()
    
    print(f"System Health: ONLINE")
    print(f"Total Active Modules/Nodes: {len(active_nodes)}")
    
    if active_nodes:
        print("\nLoaded Modules:")
        for idx, name in enumerate(active_nodes[-10:], 1):
            print(f"  {idx}. {name}")
        if len(active_nodes) > 10:
            print(f"  ... and {len(active_nodes) - 10} more.")
            
        print("\n")
        engine.run_simulation("Test input packet from main dashboard.")
    else:
        print("\n[!] No active nodes found. Waiting for Ayan to build components...")
        
    print("=================================================")

if __name__ == "__main__":
    run_dashboard()
