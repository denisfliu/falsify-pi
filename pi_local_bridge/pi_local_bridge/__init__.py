"""Self-hosted gateway bridge for pi-inference-client.

Loads a Pi checkpoint locally (via ``pi_inference_client.local``) and exposes
it over WebSocket using the same wire protocol the hosted Pi gateway speaks
— so any ``pi_inference_client.PolicyClient`` (incl. falsify's
``PiGatewayPolicy``) can connect to it by URL alone.
"""

__version__ = "0.1.0"
