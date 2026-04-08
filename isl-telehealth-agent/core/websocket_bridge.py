import json
import base64
import asyncio
import logging
import numpy as np
import cv2
import websockets
from typing import Dict, Any, Set

logger = logging.getLogger("websocket_bridge")

class WebSocketBridge:
    """
    Bidirectional WebSocket bridge.
    - Receives video frames from the browser, decodes them, and runs them through the agent pipeline.
    - Broadcasts agent pipeline results back to the browser.
    """
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.orchestrator = None
        self._ws_loop = None  # The event loop running in the WS thread
        self._processing = False  # Lock to skip frames while inference runs

    def set_orchestrator(self, orchestrator):
        """Give the bridge a reference to the orchestrator for processing frames."""
        self.orchestrator = orchestrator

    def subscribe_to_bus(self, message_bus):
        """Subscribes the broadcast method to all relevant MessageBus topics."""
        topics = [
            "sign_detected", "complaint_ready", "intent_classified", 
            "action_decided", "tool_result_ready", "response_ready", 
            "emergency_triggered", "agent_status"
        ]
        for topic in topics:
            message_bus.subscribe(topic, lambda msg, t=topic: self.broadcast(t, msg))
        logger.info(f"WebSocketBridge subscribed to {len(topics)} topics.")

    def broadcast(self, topic: str, message: Dict[str, Any]):
        """Formats the message based on the topic and sends to all connected clients."""
        if not self.clients or not self._ws_loop:
            return

        formatted_msg = {"topic": topic}
        
        try:
            if topic == "sign_detected":
                top3 = message.get("top_predictions", [])
                sign = top3[0].get("sign") if top3 else "unknown"
                conf = top3[0].get("confidence") if top3 else 0.0
                formatted_msg.update({
                    "sign": sign,
                    "confidence": conf,
                    "top3": top3,
                    "buffer_count": message.get("buffer_count", 20)
                })
            elif topic == "complaint_ready":
                comp = message.get("complaint", {})
                formatted_msg.update({
                    "body_part": comp.get("body_part"),
                    "symptom": comp.get("symptom"),
                    "duration": comp.get("duration"),
                    "completeness": comp.get("completeness_score", 0.0)
                })
            elif topic == "intent_classified":
                formatted_msg.update({
                    "intent": message.get("intent"),
                    "confidence": message.get("confidence", 1.0),
                    "requires_clarification": message.get("requires_clarification", False)
                })
            elif topic == "response_ready":
                formatted_msg.update({
                    "text_response": message.get("text_response"),
                    "patient_card": message.get("patient_card"),
                    "voice_sentence": message.get("voice_sentence")
                })
            elif topic == "emergency_triggered":
                formatted_msg.update({
                    "emergency_type": message.get("emergency_type", "MEDICAL"),
                    "immediate_action": message.get("immediate_action", "Alerting staff")
                })
            elif topic == "agent_status":
                formatted_msg.update({
                    "agent_name": message.get("agent_name"),
                    "status": message.get("status", "idle"),
                    "pipeline_status": message.get("pipeline_status"),
                    "buffer_count": message.get("buffer_count")
                })
            else:
                formatted_msg.update(message)

            json_payload = json.dumps(formatted_msg)
            
            # Schedule the send on the WS event loop
            asyncio.run_coroutine_threadsafe(self._async_send_all(json_payload), self._ws_loop)

        except Exception as e:
            logger.error(f"Error formatting/broadcasting WebSocket message for topic {topic}: {e}")

    async def _async_send_all(self, payload: str):
        if self.clients:
            disconnected = set()
            for client in self.clients.copy():
                try:
                    await client.send(payload)
                except websockets.exceptions.ConnectionClosed:
                    disconnected.add(client)
            self.clients -= disconnected

    async def _process_incoming_frame(self, data_url: str):
        """
        Decode a base64 JPEG data URL into a numpy frame and run it through the pipeline.
        Skips frames if previous inference is still running.
        """
        if not self.orchestrator or not self.orchestrator.active_session_id:
            return

        # Skip this frame if we're still processing the previous one
        if self._processing:
            return
        self._processing = True

        try:
            # Strip the "data:image/jpeg;base64," prefix
            header, encoded = data_url.split(",", 1)
            img_bytes = base64.b64decode(encoded)
            
            # Decode JPEG bytes into a numpy BGR array (OpenCV format)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return

            # Run the agent pipeline directly
            await self.orchestrator.process_webcam_frame(frame, self.orchestrator.active_session_id)

        except Exception as e:
            logger.error(f"Error processing incoming frame: {e}", exc_info=True)
        finally:
            self._processing = False

    async def handler(self, websocket, path=None):
        """Manages client connections and processes incoming frames."""
        self.clients.add(websocket)
        logger.info(f"Client connected. Total clients: {len(self.clients)}")
        try:
            async for message in websocket:
                try:
                    msg = json.loads(message)
                    msg_type = msg.get("type")

                    if msg_type == "frame":
                        # Browser sent a video frame
                        await self._process_incoming_frame(msg.get("data", ""))
                    elif msg_type == "start_session":
                        patient_id = msg.get("patient_id", "browser_patient_001")
                        if self.orchestrator and not self.orchestrator.active_session_id:
                            self.orchestrator.start_session(patient_id)
                            logger.info(f"Session started from browser for patient: {patient_id}")
                    # Other message types can be added here

                except json.JSONDecodeError:
                    logger.warning("Received non-JSON message from client.")
                except Exception as e:
                    logger.error(f"Error handling client message: {e}")

        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.discard(websocket)
            logger.info("Client disconnected.")

    def run_in_thread(self):
        """Entry point for running the WS server in a background thread."""
        self._ws_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._ws_loop)
        self._ws_loop.run_until_complete(self._run())

    async def _run(self):
        """Starts the WebSocket server and keeps it running."""
        logger.info(f"Starting WebSocket server on ws://{self.host}:{self.port}")
        async with websockets.serve(self.handler, self.host, self.port):
            await asyncio.Future()  # run forever
