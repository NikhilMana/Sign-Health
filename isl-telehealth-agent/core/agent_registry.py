import time
import logging
import numpy as np
import asyncio
from typing import Dict, Any, Optional

# Import all agents
from agents.perception_agent import PerceptionAgent
from agents.context_agent import ContextAgent
from agents.intent_agent import IntentAgent
from agents.dialogue_agent import DialogueAgent
from agents.tool_agent import ToolAgent
from agents.response_agent import ResponseAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent_registry")

class AgentRegistry:
    """
    Registry for managing and discovering available agents.
    Orchestrates the full processing pipeline from sign detection to response.
    """
    
    def __init__(self):
        logger.info("Initializing AgentRegistry and instantiating all agents...")
        self.perception = PerceptionAgent()
        self.context = ContextAgent()
        self.intent = IntentAgent()
        self.dialogue = DialogueAgent()
        self.tool = ToolAgent()
        self.response = ResponseAgent()
        logger.info("All agents successfully instantiated.")

    def get_agent(self, name: str) -> Optional[Any]:
        """Returns the agent instance by name."""
        agents = {
            "perception": self.perception,
            "context": self.context,
            "intent": self.intent,
            "dialogue": self.dialogue,
            "tool": self.tool,
            "response": self.response
        }
        return agents.get(name.lower())

    async def run_pipeline(self, frame: np.ndarray, session_id: str, message_bus=None) -> Dict[str, Any]:
        """
        Runs the full 8-stage multi-agent pipeline:
        Perception(P) -> Perception(Inference) -> Context(Load) -> Context(Extract) -> Intent -> Dialogue -> Tool -> Response
        
        Args:
            frame (np.ndarray): Video frame from the webcam.
            session_id (str): Current active session ID.
            message_bus: Optional MessageBus instance to emit stage events for frontend.
            
        Returns:
            Dict[str, Any]: The pipeline response with status and optional formatted output.
        """
        start_time = time.time()
        timings = {}

        try:
            # 1. Perception: Process Frame
            t0 = time.time()
            res_p = self.perception.process_frame(frame)
            timings["perception_process"] = (time.time() - t0) * 1000
            
            if message_bus:
                message_bus.publish("agent_status", {"pipeline_status": "buffering", "buffer_count": res_p["buffer_count"]})
                
            if not res_p["ready"]:
                return {"pipeline_status": "buffering", "buffer_count": res_p["buffer_count"], "timings": timings}

            # 2. Perception: Predict
            t0 = time.time()
            pred_res = self.perception.predict()
            timings["perception_predict"] = (time.time() - t0) * 1000
            
            if not pred_res.get("should_display", True): # or if confidence is too low
                return {"pipeline_status": "low_confidence", "raw_confidence": pred_res.get("raw_confidence"), "timings": timings}
            
            top_pred = pred_res["top_predictions"][0]
            sign = top_pred["sign"]
            confidence = top_pred["confidence"]
            
            if message_bus:
                message_bus.publish("sign_detected", {"top_predictions": pred_res.get("top_predictions", [])})
                
            self.perception.reset_buffer()
            logger.info(f"[Registry] Stage 2: Detected '{sign}' ({confidence:.2f})")

            # 3. Context: Process Sign
            t0 = time.time()
            self.context.process_sign(sign, confidence)
            timings["context_process"] = (time.time() - t0) * 1000

            # 4. Context: Extract Complaint
            t0 = time.time()
            complaint = self.context.extract_complaint()
            timings["context_extract"] = (time.time() - t0) * 1000
            
            if complaint.get("completeness_score", 0) < 0.4:
                return {"pipeline_status": "collecting_signs", "complaint": complaint, "timings": timings}
                
            if message_bus:
                message_bus.publish("complaint_ready", {"complaint": complaint})

            # 5. Intent: Classify Intent
            t0 = time.time()
            intent_result = self.intent.classify_intent(complaint)
            timings["intent_classify"] = (time.time() - t0) * 1000
            
            if message_bus:
                message_bus.publish("intent_classified", {"intent": intent_result.get("intent")})

            # 6. Dialogue: Manage Turn
            t0 = time.time()
            action = self.dialogue.manage_turn(intent_result, session_id)
            timings["dialogue_manage"] = (time.time() - t0) * 1000

            # 7. Tool: Execute
            t0 = time.time()
            tool_result = await self.tool.execute(action)
            timings["tool_execute"] = (time.time() - t0) * 1000

            # 8. Response: Format
            t0 = time.time()
            final_response = self.response.format_response(tool_result, intent_result["intent"])
            timings["response_format"] = (time.time() - t0) * 1000
            
            # Additional cleanup/updates
            self.dialogue.add_doctor_response(final_response["text_response"])
            
            final_response.update({
                "pipeline_status": "complete",
                "timings": timings,
                "total_time_ms": (time.time() - start_time) * 1000,
                # Pass back results to main.py for drawing landmarks
                "perception_results": res_p.get("results")
            })
            
            logger.info(f"[Registry] Pipeline COMPLETE for session {session_id} in {final_response['total_time_ms']:.2f}ms")
            return final_response

        except Exception as e:
            logger.error(f"Pipeline FAILURE at session {session_id}: {e}", exc_info=True)
            return {
                "pipeline_status": "error",
                "error": str(e),
                "timings": timings
            }
