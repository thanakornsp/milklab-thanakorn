"""
DropExpress Agent Harness

รับคำสั่งภาษาไทยจากผู้ใช้
ส่งให้ Gemini วิเคราะห์ว่าเหมาะกับ Tool ไหน
จากนั้นเรียก Tool จริง

ตัวอย่าง:

python agent_harness.py --cmd "บันทึกฝากตู้ Size M ราคา 35 บาท"

python agent_harness.py --cmd "ส่งพัสดุ Flash Express Tracking TH123456789 ราคา 65 บาท"

python agent_harness.py --cmd "วันนี้มีธุรกรรมกี่รายการ"

python agent_harness.py --cmd "ส่งแจ้งเตือนว่าระบบตู้ Size M เต็ม"
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
        "description": (
            "บันทึกธุรกรรมการฝากของเข้าตู้ DropExpress"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "locker_size": {
                    "type": "string",
                    "description": "ขนาดตู้ S, M หรือ L",
                },
                "phone": {
                    "type": "string",
                    "description": "เบอร์โทรศัพท์ลูกค้า",
                },
                "amount": {
                    "type": "number",
                    "description": "จำนวนเงิน",
                },
            },
            "required": [
                "locker_size",
                "amount",
            ],
        },
    },
    {
        "name": "log_parcel",
        "description": (
            "บันทึกธุรกรรมการส่งพัสดุ "
            "ผ่านบริษัทขนส่ง"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "locker_size": {
                    "type": "string",
                    "description": "ขนาดตู้ S, M หรือ L",
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
                    "description": "เบอร์โทรศัพท์ลูกค้า",
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
        "description": (
            "สรุปจำนวนและยอดเงินของธุรกรรม "
            "DropExpress ตามวันที่"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": (
                        "วันที่รูปแบบ YYYY-MM-DD"
                    ),
                },
            },
            "required": ["date"],
        },
    },
    {
        "name": "send_alert",
        "description": (
            "ส่งข้อความแจ้งเตือนผ่าน Telegram"
        ),
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


# =========================================================
# TRACE LOG
# =========================================================

def write_trace(
    event: str,
    data,
) -> None:
    """บันทึก Agent trace"""

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
    ส่งคำสั่งให้ Gemini
    และขอผลลัพธ์ในรูปแบบ:

    {
        "tool": "tool_name",
        "args": {}
    }
    """

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

    prompt = f"""
คุณคือ AI Agent ของ DropExpress

หน้าที่คือวิเคราะห์คำสั่งภาษาไทย
แล้วเลือก Tool ที่เหมาะสมที่สุด

Tools ที่ใช้ได้:

{json.dumps(
    TOOL_SCHEMA,
    ensure_ascii=False,
    indent=2,
)}

กฎ:

1. ตอบ JSON เท่านั้น
2. ห้ามใส่ markdown
3. ห้ามใส่คำอธิบายเพิ่มเติม
4. ต้องใช้ชื่อ Tool ที่มีอยู่เท่านั้น
5. ถ้าผู้ใช้ต้องการฝากของ ให้ใช้ log_locker
6. ถ้าผู้ใช้ต้องการส่งพัสดุ ให้ใช้ log_parcel
7. ถ้าผู้ใช้ถามยอดหรือจำนวนธุรกรรม ให้ใช้ query_transactions
8. ถ้าผู้ใช้ต้องการแจ้งเตือน ให้ใช้ send_alert
9. วันที่ต้องอยู่ในรูปแบบ YYYY-MM-DD

คำสั่งผู้ใช้:

{cmd}

ตอบในรูปแบบ:

{{
    "tool": "ชื่อ tool",
    "args": {{}}
}}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    text = (
        response.text
        or ""
    ).strip()

    text = (
        text
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )

    try:

        tool_call = json.loads(
            text
        )

    except json.JSONDecodeError as exc:

        raise RuntimeError(
            "Gemini ส่ง JSON ไม่ถูกต้อง: "
            f"{text}"
        ) from exc

    if (
        "tool" not in tool_call
        or "args" not in tool_call
    ):
        raise RuntimeError(
            "Response ขาด key tool หรือ args"
        )

    return tool_call


# =========================================================
# DISPATCH TOOL
# =========================================================

def dispatch_tool(
    tool_call: dict,
) -> str:
    """เรียก Tool จริงตามผลจาก Gemini"""

    tool_name = tool_call.get(
        "tool"
    )

    args = tool_call.get(
        "args",
        {},
    )

    # -----------------------------------------------------
    # LOG LOCKER
    # -----------------------------------------------------

    if tool_name == "log_locker":

        row = append_transaction(
            service="ฝากตู้",
            locker_size=args.get(
                "locker_size",
                "",
            ),
            phone=args.get(
                "phone",
                "",
            ),
            amount=args.get(
                "amount",
                0,
            ),
        )

        try:

            provider = send_notification(
                (
                    "📦 DropExpress\n"
                    "บันทึกฝากตู้สำเร็จ\n\n"
                    f"ตู้: {row['locker_size']}\n"
                    f"เบอร์: "
                    f"{row['phone'] or '-'}\n"
                    f"ราคา: "
                    f"{row['amount']:.2f} บาท"
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
                "แต่ส่งแจ้งเตือนไม่สำเร็จ: "
                f"{exc}"
            )

    # -----------------------------------------------------
    # LOG PARCEL
    # -----------------------------------------------------

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
            amount=args.get(
                "amount",
                0,
            ),
        )

        try:

            provider = send_notification(
                (
                    "📦 DropExpress\n"
                    "บันทึกส่งพัสดุสำเร็จ\n\n"
                    f"ขนส่ง: "
                    f"{row['carrier']}\n"
                    f"Tracking: "
                    f"{row['tracking_number']}\n"
                    f"ตู้: "
                    f"{row['locker_size'] or '-'}\n"
                    f"ราคา: "
                    f"{row['amount']:.2f} บาท"
                )
            )

            return (
                "บันทึกการส่งพัสดุสำเร็จ "
                f"{row['carrier']} "
                f"Tracking "
                f"{row['tracking_number']} "
                f"ราคา {row['amount']:.2f} บาท "
                f"และแจ้งเตือนผ่าน {provider}"
            )

        except Exception as exc:

            return (
                "บันทึกการส่งพัสดุสำเร็จ "
                f"{row['carrier']} "
                f"Tracking "
                f"{row['tracking_number']} "
                f"ราคา {row['amount']:.2f} บาท "
                "แต่ส่งแจ้งเตือนไม่สำเร็จ: "
                f"{exc}"
            )

    # -----------------------------------------------------
    # QUERY TRANSACTIONS
    # -----------------------------------------------------

    if tool_name == "query_transactions":

        result = query_transactions(
            args["date"]
        )

        return (
            f"วันที่ {result['date']}\n"
            f"ธุรกรรมทั้งหมด: "
            f"{result['count']} รายการ\n"
            f"ฝากตู้: "
            f"{result['locker_count']} รายการ\n"
            f"ส่งพัสดุ: "
            f"{result['parcel_count']} รายการ\n"
            f"ยอดรวม: "
            f"{result['total_amount']:.2f} บาท"
        )

    # -----------------------------------------------------
    # SEND ALERT
    # -----------------------------------------------------

    if tool_name == "send_alert":

        provider = send_notification(
            args["message"]
        )

        return (
            "ส่งข้อความแจ้งเตือนผ่าน "
            f"{provider} สำเร็จ"
        )

    raise ValueError(
        f"ไม่รู้จัก Tool: {tool_name}"
    )


# =========================================================
# MAIN
# =========================================================

def main() -> int:

    load_dotenv()

    parser = argparse.ArgumentParser(
        description=(
            "DropExpress Agent Harness"
        )
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

        # -------------------------------------------------
        # Gemini Parse
        # -------------------------------------------------

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

        # -------------------------------------------------
        # Execute Tool
        # -------------------------------------------------

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
                "tool": tool_call[
                    "tool"
                ],
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
                "error_type": type(
                    exc
                ).__name__,
                "message": str(exc),
            },
        )

        return 1


if __name__ == "__main__":
    sys.exit(main())