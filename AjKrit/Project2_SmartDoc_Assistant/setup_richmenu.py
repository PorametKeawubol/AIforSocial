"""Validate LINE UX and install the original artwork as the default rich menu."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import httpx
from PIL import Image

from config import ROOT, settings
from line_ux import CATEGORIES, LineUX, quick_reply, rich_menu_payload

API = "https://api.line.me/v2/bot"


def prepare_image(path: Path) -> tuple[bytes, str, tuple[int, int]]:
    """Only re-encode oversized PNGs; preserve original pixels and dimensions."""
    data = path.read_bytes()
    with Image.open(io.BytesIO(data)) as image:
        size = image.size
        if not (800 <= size[0] <= 2500 and 250 <= size[1] and size[0] / size[1] >= 1.45):
            raise ValueError("Image dimensions do not meet LINE rich menu requirements")
        if image.format in ("PNG", "JPEG") and len(data) <= 1_000_000:
            return data, "image/png" if image.format == "PNG" else "image/jpeg", size
        for quality in (95, 90, 85, 80, 75):
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, format="JPEG", quality=quality, optimize=True)
            encoded = buffer.getvalue()
            if len(encoded) <= 1_000_000:
                return encoded, "image/jpeg", size
    raise ValueError("Image remains over LINE's 1 MB limit")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=ROOT / "assets/richmenu_thaipol.png")
    parser.add_argument("--validate-only", action="store_true", help="Validate with LINE without changing the menu")
    args = parser.parse_args()
    if not settings.line_access_token:
        parser.error("CHANNEL_ACCESS_TOKEN is not configured")
    data, content_type, (width, height) = prepare_image(args.image)
    payload = rich_menu_payload(width, height)
    fingerprint = hashlib.sha256(data + json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    payload["name"] += f" {fingerprint}"
    with httpx.Client(headers={"Authorization": f"Bearer {settings.line_access_token}"}, timeout=30) as client:
        def request(method, url, **kwargs):
            response = client.request(method, url, **kwargs)
            if not response.is_success:
                # Do not print request headers, tokens, or exception tracebacks.
                raise RuntimeError(f"LINE API {response.status_code}: {response.text[:500]}")
            return response.json() if response.content else {}
        request("POST", f"{API}/richmenu/validate", json=payload)
        ux = LineUX(settings.index_dir)
        for label in ("ถาม Assistant", *CATEGORIES, "หัวข้อทั้งหมด"):
            request("POST", f"{API}/message/validate/reply", json={"messages": ux.navigation(label)})
        request("POST", f"{API}/message/validate/push", json={"messages": [
            {"type": "text", "text": "ตัวอย่างคำตอบ", "quickReply": quick_reply("1.3 แจ้งความเอกสารสำคัญหาย")}]})
        print(f"LINE validation passed: 6 navigation messages + quick reply; image {width}x{height}, {len(data)} bytes")
        if args.validate_only:
            return
        menus = request("GET", f"{API}/richmenu/list").get("richmenus", [])
        existing = next((menu for menu in menus if menu.get("name") == payload["name"]), None)
        menu_id = existing["richMenuId"] if existing else request("POST", f"{API}/richmenu", json=payload)["richMenuId"]
        image_missing = not existing
        if existing:
            check = client.get(f"https://api-data.line.me/v2/bot/richmenu/{menu_id}/content")
            if check.status_code == 404:
                image_missing = True
            elif not check.is_success:
                raise RuntimeError(f"Cannot inspect rich menu image: LINE API {check.status_code}")
        if image_missing:
            request("POST", f"https://api-data.line.me/v2/bot/richmenu/{menu_id}/content",
                    content=data, headers={"Content-Type": content_type})
        request("POST", f"{API}/user/all/richmenu/{menu_id}")
        print(f"Default rich menu installed: {menu_id}")
        print("Existing menus were preserved; users with per-user menus may keep their previous menu.")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, httpx.HTTPError) as exc:
        raise SystemExit(str(exc)) from None
