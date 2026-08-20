"""
DropExpress Agent Harness

รับคำสั่งภาษาไทยจากผู้ใช้
ส่งให้ Gemini วิเคราะห์ว่าเหมาะกับ Tool ไหน
จากนั้นเรียก Tool จริง
"""

import argparse
import json
import os
import sys
from datetime import datetime

from dotenv import load_dotenv
from google import genai

from sales_logger import (
    append_transaction,
    query_transactions,
    send_notification,
)


# =========================================================
# TOOL SCHEMA
# =========================================================

TOOL_SCHEMA = [
    {
        "name": "log_locker",
        "description": "บันทึกธุรกรรมการฝากของเข้าตู้ DropExpress",
        "parameters": {
            "type": "object",
            "properties": {
                "locker_size": {
                    "type": "string",
                    "description": "ขนาดตู้ S, M หรือ L",
                },
                "phone": {
                    "type": "string",
                    "description": "เบอร์โทรศัพท์ลูกค้า ถ้าไม่มีให้เป็นค่าว่าง",
                },
                "amount": {
                    "type": "number",
                    "description": "จำนวนเงิน",
                },
            },
            "required": ["locker_size", "amount"],
        },
    },
    {
        "name": "log_parcel",
        "description": "บันทึกธุรกรรมการส่งพัสดุผ่านบริษัทขนส่ง",
        "parameters": {
            "type": "object",
            "properties": {
                "locker_size": {
                    "type": "string",
                    "description": "ขนาดตู้ S, M หรือ L ถ้าไม่มีให้เป็นค่าว่าง",
                },
                "carrier": {
                    "type": "string",
                    "description": "บริษัทขนส่ง",
                },
                "tracking_number": {
                    "type": "string",
                    "description": "Tracking Number",
                },
                "phone": {
                    "type": "string",
                    "description": "เบอร์โทรศัพท์ลูกค้า ถ้าไม่มีให้เป็นค่าว่าง",
                },
                "amount": {
                    "type": "number",
                    "description": "จำนวนเงิน",
                },
            },
            "required": [
                "carrier",
                "tracking_number",
                "amount",
            ],
        },
    },
    {
        "name": "query_transactions",
        "description": "สรุปจำนวนและยอดเงินของธุรกรรม DropExpress ตามวันที่",
        "parameters": {
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": "วันที่รูปแบบ YYYY-MM-DD",
                },
            },
            "required": ["date"],
        },
    },
    {
        "name": "send_alert",
        "description": "ส่งข้อความแจ้งเตือนผ่าน Telegram",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "ข้อความแจ้งเตือน",
                },
            },
            "required": ["message"],
        },
    },
]


ALLOWED_TOOLS = {
    "log_locker",
    "log_parcel",
    "query_transactions",
    "send_alert",
}


# =========================================================
# TRACE
# =========================================================

def write_trace(event: str, data) -> None:
    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    with open(
        "agent_trace.log",
        "a",
        encoding="utf-8",
    ) as log_file:
        log_file.write(
            f"[{timestamp}] {event}: "
            f"{json.dumps(data, ensure_ascii=False, default=str)}\n"
        )


# =========================================================
# PARSE COMMAND
# =========================================================

def parse_command(
    cmd: str,
    api_key: str | None = None,
) -> dict:
    """
    วิเคราะห์คำสั่งภาษาไทยด้วย Gemini

    คืนค่า:
    {
        "tool": "...",
        "args": {...}
    }
    """

    load_dotenv()

    if api_key is None:
        api_key = os.getenv(
            "GOOGLE_API_KEY"
        )

    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY not found"
        )

    client = genai.Client(
        api_key=api_key
    )

    today = datetime.now().strftime(
        "%Y-%m-%d"
    )

    prompt = f"""
คุณคือ AI Agent ของระบบ DropExpress

วันนี้คือ {today}

หน้าที่ของคุณ:
วิเคราะห์คำสั่งภาษาไทยของผู้ใช้
แล้วเลือก Tool ที่เหมาะสมที่สุด

Tools ที่อนุญาต:

{json.dumps(
    TOOL_SCHEMA,
    ensure_ascii=False,
    indent=2,
)}

กฎสำคัญ:

1. ต้องตอบเป็น JSON object เท่านั้น
2. ห้ามใช้ Markdown
3. ห้ามใส่คำอธิบาย
4. JSON ต้องมี key ชื่อ "tool"
5. JSON ต้องมี key ชื่อ "args"
6. ค่า "tool" ต้องเป็นหนึ่งใน:
   - log_locker
   - log_parcel
   - query_transactions
   - send_alert
7. ถ้าผู้ใช้ต้องการฝากของ ให้ใช้ log_locker
8. ถ้าผู้ใช้ต้องการส่งพัสดุ ให้ใช้ log_parcel
9. ถ้าผู้ใช้ถามจำนวนหรือยอดธุรกรรม ให้ใช้ query_transactions
10. ถ้าผู้ใช้ต้องการส่งแจ้งเตือน ให้ใช้ send_alert
11. วันที่ต้องเป็น YYYY-MM-DD
12. ถ้าไม่มี phone ให้ใช้ ""
13. ถ้าไม่มี locker_size สำหรับส่งพัสดุ ให้ใช้ ""
14. ห้ามสร้าง Tool ใหม่
15. ห้ามตอบ null
16. ห้ามตอบ None

ตัวอย่าง:

ผู้ใช้:
บันทึกฝากตู้ Size M ราคา 35 บาท

ตอบ:
{{
  "tool": "log_locker",
  "args": {{
    "locker_size": "M",
    "amount": 35
  }}
}}

ผู้ใช้:
ส่งพัสดุ Flash Express Tracking TH123456789 ราคา 65 บาท

ตอบ:
{{
  "tool": "log_parcel",
  "args": {{
    "carrier": "Flash Express",
    "tracking_number": "TH123456789",
    "amount": 65
  }}
}}

ผู้ใช้:
วันนี้มีธุรกรรมกี่รายการ

ตอบ:
{{
  "tool": "query_transactions",
  "args": {{
    "date": "{today}"
  }}
}}

ผู้ใช้:
ส่งแจ้งเตือนว่าระบบทำงานปกติ

ตอบ:
{{
  "tool": "send_alert",
  "args": {{
    "message": "ระบบทำงานปกติ"
  }}
}}

คำสั่งของผู้ใช้:

{cmd}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    text = (
        response.text or ""
    ).strip()

    if not text:
        raise RuntimeError(
            "Gemini ไม่ส่งข้อความกลับมา"
        )

    # ลบ Markdown fence ถ้ามี
    if text.startswith("```"):
        text = text.replace(
            "```json",
            "",
        )
        text = text.replace(
            "```",
            "",
        )
        text = text.strip()

    try:
        tool_call = json.loads(
            text
        )
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Gemini ส่ง JSON ไม่ถูกต้อง: "
            f"{text}"
        ) from exc

    if not isinstance(
        tool_call,
        dict,
    ):
        raise RuntimeError(
            "Gemini response ไม่ใช่ JSON object"
        )

    tool_name = tool_call.get(
        "tool"
    )

    args = tool_call.get(
        "args"
    )

    if not tool_name:
        raise RuntimeError(
            "Gemini ไม่ได้ระบุ tool"
        )

    if not isinstance(
        args,
        dict,
    ):
        raise RuntimeError(
            "Gemini ไม่ได้ระบุ args เป็น object"
        )

    if tool_name not in ALLOWED_TOOLS:
        raise RuntimeError(
            f"Gemini เลือก Tool ไม่ถูกต้อง: {tool_name}"
        )

    return {
        "tool": tool_name,
        "args": args,
    }


# =========================================================
# DISPATCH TOOL
# =========================================================

def dispatch_tool(
    tool_call: dict,
) -> str:

    tool_name = tool_call.get(
        "tool"
    )

    args = tool_call.get(
        "args",
        {},
    )

    if tool_name == "log_locker":

        locker_size = str(
            args.get(
                "locker_size",
                "",
            )
        ).upper()

        amount = float(
            args.get(
                "amount",
                0,
            )
        )

        row = append_transaction(
            service="ฝากตู้",
            locker_size=locker_size,
            phone=args.get(
                "phone",
                "",
            ),
            amount=amount,
        )

        try:
            provider = send_notification(
                (
                    "📦 DropExpress\n"
                    "บันทึกฝากตู้สำเร็จ\n\n"
                    f"ตู้: {row['locker_size']}\n"
                    f"เบอร์: {row['phone'] or '-'}\n"
                    f"ราคา: {row['amount']:.2f} บาท"
                )
            )

            return (
                "บันทึกการฝากตู้สำเร็จ "
                f"Size {row['locker_size']} "
                f"ราคา {row['amount']:.2f} บาท "
                f"และแจ้งเตือนผ่าน {provider}"
            )

        except Exception as exc:

            return (
                "บันทึกการฝากตู้สำเร็จ "
                f"Size {row['locker_size']} "
                f"ราคา {row['amount']:.2f} บาท "
                f"แต่ส่งแจ้งเตือนไม่สำเร็จ: {exc}"
            )

    if tool_name == "log_parcel":

        row = append_transaction(
            service="ส่งพัสดุ",
            locker_size=args.get(
                "locker_size",
                "",
            ),
            carrier=args.get(
                "carrier",
                "",
            ),
            tracking_number=args.get(
                "tracking_number",
                "",
            ),
            phone=args.get(
                "phone",
                "",
            ),
            amount=float(
                args.get(
                    "amount",
                    0,
                )
            ),
        )

        try:
            provider = send_notification(
                (
                    "📦 DropExpress\n"
                    "บันทึกส่งพัสดุสำเร็จ\n\n"
                    f"ขนส่ง: {row['carrier']}\n"
                    f"Tracking: {row['tracking_number']}\n"
                    f"ตู้: {row['locker_size'] or '-'}\n"
                    f"ราคา: {row['amount']:.2f} บาท"
                )
            )

            return (
                "บันทึกการส่งพัสดุสำเร็จ "
                f"{row['carrier']} "
                f"Tracking {row['tracking_number']} "
                f"ราคา {row['amount']:.2f} บาท "
                f"และแจ้งเตือนผ่าน {provider}"
            )

        except Exception as exc:

            return (
                "บันทึกการส่งพัสดุสำเร็จ "
                f"{row['carrier']} "
                f"Tracking {row['tracking_number']} "
                f"ราคา {row['amount']:.2f} บาท "
                f"แต่ส่งแจ้งเตือนไม่สำเร็จ: {exc}"
            )

    if tool_name == "query_transactions":

        date = args.get(
            "date"
        )

        if not date:
            raise ValueError(
                "ไม่พบวันที่สำหรับ query_transactions"
            )

        result = query_transactions(
            date
        )

        return (
            f"วันที่ {result['date']}\n"
            f"ธุรกรรมทั้งหมด: {result['count']} รายการ\n"
            f"ฝากตู้: {result['locker_count']} รายการ\n"
            f"ส่งพัสดุ: {result['parcel_count']} รายการ\n"
            f"ยอดรวม: {result['total_amount']:.2f} บาท"
        )

    if tool_name == "send_alert":

        message = args.get(
            "message"
        )

        if not message:
            raise ValueError(
                "ไม่พบข้อความแจ้งเตือน"
            )

        provider = send_notification(
            message
        )

        return (
            f"ส่งข้อความแจ้งเตือนผ่าน "
            f"{provider} สำเร็จ"
        )

    raise ValueError(
        f"ไม่รู้จัก Tool: {tool_name}"
    )


# =========================================================
# MAIN
# =========================================================

def main() -> int:

    parser = argparse.ArgumentParser(
        description="DropExpress Agent Harness"
    )

    parser.add_argument(
        "--cmd",
        required=True,
        help="คำสั่งภาษาไทย",
    )

    args = parser.parse_args()

    command = args.cmd

    print(
        f"[USER] {command}"
    )

    write_trace(
        "user_input",
        command,
    )

    try:

        tool_call = parse_command(
            command
        )

        print(
            "[LLM] "
            f"tool={tool_call['tool']} "
            f"args={tool_call['args']}"
        )

        write_trace(
            "llm_response",
            tool_call,
        )

        result = dispatch_tool(
            tool_call
        )

        print(
            "[TOOL] "
            f"{tool_call['tool']} "
            f"{result}"
        )

        print(
            f"[USER] ← {result}"
        )

        write_trace(
            "tool_result",
            {
                "tool": tool_call["tool"],
                "result": result,
            },
        )

        return 0

    except Exception as exc:

        print(
            f"[ERROR] {exc}",
            file=sys.stderr,
        )

        write_trace(
            "tool_error",
            {
                "error_type": type(exc).__name__,
                "message": str(exc),
            },
        )

        return 1


if __name__ == "__main__":
    sys.exit(main())