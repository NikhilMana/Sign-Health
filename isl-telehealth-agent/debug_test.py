import sys, logging, traceback
logging.basicConfig(level=logging.WARNING)
try:
    from agents.perception_agent import PerceptionAgent
    agent = PerceptionAgent()
    print(f"Demo mode: {agent.demo_mode}")
except Exception as e:
    print(f"EXCEPTION TYPE: {type(e).__name__}")
    print(f"EXCEPTION MSG: {e}")
    traceback.print_exc()
