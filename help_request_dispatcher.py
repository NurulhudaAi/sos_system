import os
import requests
import logging
import json
from datetime import datetime
from typing import Optional, Dict, Any
import numpy as np

logger = logging.getLogger("help_dispatcher")


class HelpRequestDispatcher:
    """Handles real-time help request webhook notifications for SOS events."""

    def __init__(self, webhook_url: str = "", timeout: int = 5, db=None):
        self.webhook_url = webhook_url or os.getenv("HELP_WEBHOOK_URL", "")
        self.timeout = timeout
        self.db = db
        self.enabled = bool(self.webhook_url)

        if self.enabled:
            logger.info(f"✅ Help dispatcher enabled: {self._mask_url(self.webhook_url)}")
        else:
            logger.info("⚠️  Help dispatcher disabled (no webhook URL)")

    def should_send_help_request(self, event_type: str, severity: int) -> bool:
        """Determine if event should trigger help request.

        Rules:
        - hand_sos: ALL severities
        - fall: severity >= 2 (HIGH or CRITICAL)
        - others: NO
        """
        if not self.enabled:
            return False

        if event_type == "hand_sos":
            return True  # ALL hand SOS events

        if event_type == "fall" and severity >= 2:  # MED (1) < HIGH (2) <= CRITICAL (3)
            return True

        return False

    def dispatch_help_request(
        self,
        event_uuid: str,
        event_type: str,
        severity: int,
        severity_name: str,
        location: str,
        source_id: str,
        track_id: int,
        image_path: str,
        frame=None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Send help request webhook. Returns success status."""
        if not self.enabled:
            return False

        try:
            # Build payload with image path (NOT base64)
            payload = {
                "event_uuid": event_uuid,
                "event_type": event_type,
                "severity": severity,
                "severity_name": severity_name,
                "location": location,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "track_id": track_id,
                "source_id": source_id,
                "image_path": image_path,  # FILE PATH ONLY
                "status": "PENDING_RESPONSE",
                "meta": self._serialize_meta(extra or {})
            }

            # Send with retries
            success = self._send_with_retries(payload)

            # Log result to MongoDB if available
            if self.db:
                try:
                    status = "SENT" if success else "FAILED"
                    self.db.insert_help_request(
                        event_uuid=event_uuid,
                        webhook_url=self.webhook_url,
                        status=status,
                        error=None if success else "Max retries exceeded"
                    )
                except Exception as e:
                    logger.warning(f"Could not log help request: {e}")

            return success

        except Exception as e:
            logger.error(f"❌ Help request failed: {e}")
            return False

    def _send_with_retries(self, payload: Dict, max_retries: int = 3) -> bool:
        """Send webhook with retry logic."""
        import time

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.webhook_url,
                    json=payload,
                    timeout=self.timeout,
                    headers={"Content-Type": "application/json"}
                )

                if response.status_code in [200, 201, 202, 204]:
                    logger.info(
                        f"✅ Help request sent (attempt {attempt + 1}): "
                        f"{response.status_code} in {response.elapsed.total_seconds():.2f}s"
                    )
                    return True
                else:
                    logger.warning(
                        f"⚠️  Help request returned {response.status_code}: {response.text[:100]}"
                    )
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
                    return False

            except requests.Timeout:
                logger.warning(f"⚠️  Webhook timeout (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                return False

            except Exception as e:
                logger.error(f"❌ Webhook error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                return False

        return False

    def _serialize_meta(self, extra: Dict) -> Dict:
        """Convert metadata to JSON-serializable format."""
        if not extra:
            return {}

        serialized = {}
        for key, value in extra.items():
            try:
                # Handle numpy types
                if isinstance(value, (np.integer, np.floating)):
                    serialized[key] = float(value) if isinstance(value, np.floating) else int(value)
                elif isinstance(value, np.ndarray):
                    serialized[key] = value.tolist()
                elif isinstance(value, (list, dict, str, int, float, bool, type(None))):
                    serialized[key] = value
                else:
                    serialized[key] = str(value)
            except Exception:
                serialized[key] = str(value)

        return serialized

    @staticmethod
    def _mask_url(url: str) -> str:
        """Mask sensitive parts of webhook URL."""
        if not url:
            return "(empty)"
        if len(url) > 40:
            return url[:30] + "..."
        return url
