import json
import asyncio
import logging
import websockets
from typing import Dict, Any, Set

logger = logging.getLogger("websocket_bridge")

class WebSocketBridge:
    """
    Bridges backend MessageBus events to the frontend dashboard over WebSockets.
    Translates backend topic data into the format requested by the user.
    """
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()

    def subscribe_to_bus(self, message_bus):
        """
        Subscribes the broadcast method to all relevant MessageBus topics.
        """
        topics = [
            "sign_detected", "complaint_ready", "intent_classified", 
            "action_decided", "tool_result_ready", "response_ready", 
            "emergency_triggered", "agent_status"
        ]
        for topic in topics:
            message_bus.subscribe(topic, lambda msg, t=topic: self.broadcast(t, msg))
        logger.info(f"WebSocketBridge subscribed to {len(topics)} topics.")

    def broadcast(self, topic: str, message: Dict[str, Any]):
        """
        Formats the message based on the topic and sends to all connected clients.
        """
        if not self.clients:
            return

        # Format message according to user specifications
        formatted_msg = {"topic": topic}
        
        try:
            if topic == "sign_detected":
                # Expecting top_predictions list in message
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
                    "status": message.get("status", "idle")
                })
            else:
                # Default format for action_decided, tool_result_ready, etc.
                formatted_msg.update(message)

            json_payload = json.dumps(formatted_msg)
            
            # Use the loop to schedule the sending
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.run_coroutine_threadsafe(self._async_send_all(json_payload), loop)
            except RuntimeError:
                pass # Event loop not running in this thread context

        except Exception as e:
            logger.error(f"Error formatting/broadcasting WebSocket message for topic {topic}: {e}")

    async def _async_send_all(self, payload: str):
        if self.clients:
            disconnected = set()
            for client in self.clients:
                try:
                    await client.send(payload)
                except websockets.exceptions.ConnectionClosed:
                    disconnected.add(client)
            self.clients -= disconnected

    async def handler(self, websocket, path=None):
        """Manages client connections."""
        self.clients.add(websocket)
        logger.info(f"Client connected. Total clients: {len(self.clients)}")
        try:
            async for _ in websocket:
                pass # No input expected from clients
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.remove(websocket)
            logger.info("Client disconnected.")

    async def run(self):
        """Starts the WebSocket server and keeps it running."""
        logger.info(f"Starting WebSocket server on ws://{self.host}:{self.port}")
        async with websockets.serve(self.handler, self.host, self.port):
            await asyncio.Future() # run forever
