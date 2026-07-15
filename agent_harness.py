"""MilkLab Agent Harness (S2).

Usage:
    python agent_harness.py --cmd "บันทึกขายนมหมี 2 ขวด ขวดละ 65"

รับคำสั่งภาษาไทย ส่งให้ Gemini พร้อม tool schema parse response เป็น tool call
เรียก tool จริง print trace log

นักศึกษาต้องเติม TODO ใน 3 จุด ใน Session 2 Lab 2.3
"""

import argparse
import json
import os
import sys
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from sales_logger import append_to_sheet, query_sales, send_notification

TOOL_SCHEMA = [
    {
        "name": "log_sale",
        "description": "บันทึกการขายลง Google Sheets และส่ง notification",
        "parameters": {
            "type": "object",
            "properties": {
                "menu": {"type": "string", "description": "ชื่อเมนู"},
                "qty": {"type": "integer", "description": "จำนวนที่ขาย"},
                "price": {"type": "number", "description": "ราคาต่อหน่วย"},
            },
            "required": ["menu", "qty", "price"],
        },
    },
    {
        "name": "query_sales",
        "description": "ดูยอดขายของวันที่ระบุ",
        "parameters": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "วันที่ format YYYY-MM-DD"},
            },
            "required": ["date"],
        },
    },
    {
        "name": "send_alert",
        "description": "ส่ง message แจ้งเตือนผ่าน Bot",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
            },
            "required": ["message"],
        },
    },
]


def parse_command(cmd: str, api_key: str | None = None) -> dict:
    """TODO 1: ส่ง cmd ไป Gemini พร้อม TOOL_SCHEMA ขอให้ตอบเป็น JSON {tool, args}

    Returns dict {"tool": <name>, "args": <dict>}
    Raises RuntimeError ถ้า parse ไม่ได้
    """
    if api_key is None:
        api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not found")

    client = genai.Client(api_key=api_key)

    prompt = f"""คุณเป็นตัวช่วยแปลงคำสั่งภาษาไทยเป็น tool call

Tools ที่ใช้ได้:
{json.dumps(TOOL_SCHEMA, ensure_ascii=False, indent=2)}

คำสั่งจากผู้ใช้:
{cmd}

ตอบกลับเป็น JSON เท่านั้น โดยใช้รูปแบบ:
{{"tool": "ชื่อ tool", "args": {{}}}}

ห้ามตอบข้อความอื่น และห้ามมี markdown backtick
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    text = response.text.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        tool_call = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"parse JSON ไม่ได้: {text}") from exc

    if "tool" not in tool_call or "args" not in tool_call:
        raise RuntimeError(f"response ขาด key tool/args: {tool_call}")

    return tool_call


def dispatch_tool(tool_call: dict) -> str:
    """TODO 2: เรียก tool ตาม tool_call["tool"] ด้วย args จริง

    Returns: ข้อความสรุปผลที่ tool คืน
    """
    tool_name = tool_call.get("tool")
    tool_args = tool_call.get("args", {})

    if tool_name == "log_sale":
        row = append_to_sheet(
            menu=tool_args["menu"],
            qty=tool_args["qty"],
            price=tool_args["price"],
        )

        total = row["total"]

        try:
            provider = send_notification(
                f"บันทึก {row['menu']} x{row['qty']} = {total} บาท"
            )
            return (
                f"บันทึกยอดขายสำเร็จ "
                f"{row['menu']} x{row['qty']} = {total} บาท "
                f"และแจ้งเตือนผ่าน {provider}"
            )
        except Exception as exc:
            return (
                f"บันทึกยอดขายสำเร็จ "
                f"{row['menu']} x{row['qty']} = {total} บาท "
                f"แต่ส่งแจ้งเตือนไม่สำเร็จ: {exc}"
            )

    if tool_name == "send_alert":
        message = tool_args["message"]
        provider = send_notification(message)
        return f"ส่งข้อความแจ้งเตือนผ่าน {provider} สำเร็จ"

    if tool_name == "query_sales":
        result = query_sales(tool_args["date"])

        return (
            f"วันที่ {result['date']} "
            f"มี {result['count']} รายการ "
            f"ขายรวม {result['total_qty']} ชิ้น "
            f"ยอดรวม {result['total_sales']} บาท"
        )

    raise ValueError(f"ไม่รู้จัก tool: {tool_name}")

def write_trace(event: str, data) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open("agent_trace.log", "a", encoding="utf-8") as log_file:
        log_file.write(
            f"[{timestamp}] {event}: "
            f"{json.dumps(data, ensure_ascii=False, default=str)}\n"
        )

def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--cmd", required=True, help="คำสั่งภาษาไทย")
    args = parser.parse_args()

    print(f"[USER] {args.cmd}")
    write_trace("user_input", args.cmd)

    try:
        tool_call = parse_command(args.cmd)

        print(
            f"[LLM]  tool={tool_call['tool']} "
            f"args={tool_call['args']}"
        )

        write_trace("llm_response", tool_call)

        result = dispatch_tool(tool_call)

        print(f"[TOOL] {tool_call['tool']} {result}")
        print(f"[USER] ← {result}")

        write_trace(
            "tool_result",
            {
                "tool": tool_call["tool"],
                "result": result,
            },
        )

        return 0

    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)

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
