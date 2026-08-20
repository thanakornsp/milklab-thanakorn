"""
DropExpress Caption Generator

Usage:
    python caption_generator.py

Reads GOOGLE_API_KEY from environment.
Generates a Thai marketing caption for DropExpress services.
"""

import os
import sys

from dotenv import load_dotenv
from google import genai


PROMPT_TEMPLATE = """\
คุณคือ Social Media Manager ของ DropExpress
บริการตู้ล็อคเกอร์ฝากของและฝากส่งพัสดุอัจฉริยะ 24 ชั่วโมง

จงเขียนแคปชั่นภาษาไทยสำหรับโปรโมตบริการ:

{service}

เงื่อนไข:
- เขียน 2 ถึง 3 ประโยค
- ใช้ภาษาไทยที่อ่านง่าย
- โทนทันสมัย เป็นกันเอง และน่าเชื่อถือ
- สามารถใช้ emoji ได้อย่างเหมาะสม
- เน้นความสะดวก รวดเร็ว และการให้บริการ 24 ชั่วโมง
- ต้องมี Call-to-Action ตอนท้าย
- ห้ามใช้ em dash
- ห้ามกล่าวอ้างข้อมูลที่ไม่ได้ระบุในคำสั่ง
"""


def generate_caption(
    service: str,
    api_key: str | None = None,
) -> str:
    """Generate a Thai marketing caption for DropExpress."""

    key = api_key or os.environ.get("GOOGLE_API_KEY")

    if not key:
        raise RuntimeError(
            "GOOGLE_API_KEY not set in env or argument"
        )

    client = genai.Client(api_key=key)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=PROMPT_TEMPLATE.format(service=service),
    )

    return response.text or ""


def main() -> int:
    """Run the caption generator from command line."""

    load_dotenv()

    service = input(
        "บริการที่ต้องการโปรโมต: "
    ).strip()

    if not service:
        print("กรุณาใส่ชื่อบริการ")
        return 1

    try:
        caption = generate_caption(service)
    except Exception as exc:
        print(
            f"[ERROR] {exc}",
            file=sys.stderr,
        )
        return 1

    print()
    print("===== DropExpress Caption =====")
    print(caption)
    print("===============================")

    return 0


if __name__ == "__main__":
    sys.exit(main())