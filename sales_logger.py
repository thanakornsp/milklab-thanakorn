"""
DropExpress Transaction Logger

บันทึกธุรกรรมของ DropExpress ลง Google Sheets
และส่งการแจ้งเตือนผ่าน Telegram

ตัวอย่างการใช้งาน:

python sales_logger.py \
    --service "ฝากตู้" \
    --locker-size "M" \
    --carrier "" \
    --tracking "" \
    --phone "0812345678" \
    --amount 35

หรือ

python sales_logger.py \
    --service "ส่งพัสดุ" \
    --locker-size "M" \
    --carrier "Flash Express" \
    --tracking "TH123456789" \
    --phone "0812345678" \
    --amount 65
"""

import argparse
import json
import os
import sys
from datetime import datetime

import gspread
import requests

from google.oauth2.service_account import Credentials


# =========================================================
# Google Sheets
# =========================================================

SHEET_WORKSHEET = "Sales"


def get_credentials():
    """โหลด Google Service Account credentials"""

    creds_json = os.getenv(
        "GOOGLE_SHEETS_CREDENTIALS"
    )

    if not creds_json:
        raise RuntimeError(
            "GOOGLE_SHEETS_CREDENTIALS not found"
        )

    try:
        creds_dict = json.loads(
            creds_json
        )
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "GOOGLE_SHEETS_CREDENTIALS "
            "ไม่ใช่ JSON ที่ถูกต้อง"
        ) from exc

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]

    return Credentials.from_service_account_info(
        creds_dict,
        scopes=scopes,
    )


def get_worksheet():
    """เชื่อมต่อ Google Sheet"""

    sheet_id = os.getenv(
        "GOOGLE_SHEET_ID"
    )

    if not sheet_id:
        raise RuntimeError(
            "GOOGLE_SHEET_ID not found"
        )

    credentials = get_credentials()

    client = gspread.authorize(
        credentials
    )

    spreadsheet = client.open_by_key(
        sheet_id
    )

    try:
        worksheet = spreadsheet.worksheet(
            SHEET_WORKSHEET
        )
    except gspread.WorksheetNotFound:
        worksheet = spreadsheet.add_worksheet(
            title=SHEET_WORKSHEET,
            rows=1000,
            cols=10,
        )

        worksheet.append_row(
            [
                "timestamp",
                "service",
                "locker_size",
                "carrier",
                "tracking_number",
                "phone",
                "amount",
            ]
        )

    return worksheet


# =========================================================
# Validation
# =========================================================

def validate_transaction(
    service: str,
    locker_size: str,
    carrier: str,
    tracking_number: str,
    phone: str,
    amount: float,
) -> None:
    """ตรวจสอบข้อมูลธุรกรรม"""

    if not service.strip():
        raise ValueError(
            "service ต้องไม่ว่าง"
        )

    valid_sizes = {
        "",
        "S",
        "M",
        "L",
    }

    if locker_size.upper() not in valid_sizes:
        raise ValueError(
            "locker_size ต้องเป็น S, M, L "
            "หรือเว้นว่าง"
        )

    if amount < 0:
        raise ValueError(
            "amount ต้องไม่ติดลบ"
        )

    if service == "ส่งพัสดุ":

        if not carrier.strip():
            raise ValueError(
                "การส่งพัสดุต้องระบุ carrier"
            )

        if not tracking_number.strip():
            raise ValueError(
                "การส่งพัสดุต้องระบุ tracking_number"
            )


# =========================================================
# บันทึกธุรกรรม
# =========================================================

def append_transaction(
    service: str,
    locker_size: str = "",
    carrier: str = "",
    tracking_number: str = "",
    phone: str = "",
    amount: float = 0,
) -> dict:
    """บันทึกธุรกรรม DropExpress ลง Google Sheets"""

    service = service.strip()
    locker_size = locker_size.strip().upper()
    carrier = carrier.strip()
    tracking_number = tracking_number.strip()
    phone = phone.strip()

    amount = float(amount)

    validate_transaction(
        service=service,
        locker_size=locker_size,
        carrier=carrier,
        tracking_number=tracking_number,
        phone=phone,
        amount=amount,
    )

    worksheet = get_worksheet()

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    row = [
        timestamp,
        service,
        locker_size,
        carrier,
        tracking_number,
        phone,
        amount,
    ]

    worksheet.append_row(
        row
    )

    return {
        "timestamp": timestamp,
        "service": service,
        "locker_size": locker_size,
        "carrier": carrier,
        "tracking_number": tracking_number,
        "phone": phone,
        "amount": amount,
    }


# =========================================================
# Query ธุรกรรม
# =========================================================

def query_transactions(
    date: str,
) -> dict:
    """สรุปธุรกรรมของวันที่กำหนด"""

    if not date or len(date) != 10:
        raise ValueError(
            "date ต้องเป็นรูปแบบ YYYY-MM-DD"
        )

    worksheet = get_worksheet()

    records = worksheet.get_all_records()

    matched = []

    for row in records:

        timestamp = str(
            row.get(
                "timestamp",
                "",
            )
        )

        if timestamp.startswith(date):
            matched.append(row)

    total_transactions = len(
        matched
    )

    total_amount = 0.0

    for row in matched:

        try:
            total_amount += float(
                row.get(
                    "amount",
                    0,
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            pass

    locker_count = 0
    parcel_count = 0

    for row in matched:

        service = str(
            row.get(
                "service",
                "",
            )
        )

        if service == "ฝากตู้":
            locker_count += 1

        elif service == "ส่งพัสดุ":
            parcel_count += 1

    return {
        "date": date,
        "count": total_transactions,
        "locker_count": locker_count,
        "parcel_count": parcel_count,
        "total_amount": total_amount,
    }


# =========================================================
# Telegram Notification
# =========================================================

def send_notification(
    message: str,
) -> str:
    """ส่งข้อความแจ้งเตือนผ่าน Telegram"""

    token = os.getenv(
        "TELEGRAM_BOT_TOKEN"
    )

    chat_id = os.getenv(
        "TELEGRAM_CHAT_ID"
    )

    if not token or not chat_id:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN หรือ "
            "TELEGRAM_CHAT_ID not found"
        )

    url = (
        "https://api.telegram.org/"
        f"bot{token}/sendMessage"
    )

    payload = {
        "chat_id": chat_id,
        "text": message,
    }

    response = requests.post(
        url,
        data=payload,
        timeout=10,
    )

    response.raise_for_status()

    return "telegram"


# =========================================================
# CLI
# =========================================================

def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "DropExpress Transaction Logger"
        )
    )

    parser.add_argument(
        "--service",
        required=True,
        help=(
            "ประเภทบริการ เช่น "
            "ฝากตู้ หรือ ส่งพัสดุ"
        ),
    )

    parser.add_argument(
        "--locker-size",
        default="",
        help="ขนาดตู้ S/M/L",
    )

    parser.add_argument(
        "--carrier",
        default="",
        help="บริษัทขนส่ง",
    )

    parser.add_argument(
        "--tracking",
        default="",
        help="Tracking Number",
    )

    parser.add_argument(
        "--phone",
        default="",
        help="เบอร์โทรศัพท์ลูกค้า",
    )

    parser.add_argument(
        "--amount",
        type=float,
        required=True,
        help="จำนวนเงิน",
    )

    args = parser.parse_args()

    try:

        transaction = append_transaction(
            service=args.service,
            locker_size=args.locker_size,
            carrier=args.carrier,
            tracking_number=args.tracking,
            phone=args.phone,
            amount=args.amount,
        )

    except Exception as exc:

        print(
            f"[ERROR] บันทึกธุรกรรมล้มเหลว: {exc}",
            file=sys.stderr,
        )

        return 1

    message = (
        "📦 DropExpress\n"
        "บันทึกธุรกรรมสำเร็จ\n\n"
        f"บริการ: {transaction['service']}\n"
        f"ตู้: {transaction['locker_size'] or '-'}\n"
        f"ขนส่ง: {transaction['carrier'] or '-'}\n"
        f"Tracking: "
        f"{transaction['tracking_number'] or '-'}\n"
        f"เบอร์โทร: "
        f"{transaction['phone'] or '-'}\n"
        f"จำนวนเงิน: "
        f"{transaction['amount']:.2f} บาท"
    )

    try:

        provider = send_notification(
            message
        )

        print(
            "[OK] บันทึกธุรกรรมและ "
            f"แจ้งเตือนผ่าน {provider} สำเร็จ"
        )

    except Exception as exc:

        print(
            "[WARN] บันทึกธุรกรรมสำเร็จ "
            "แต่ส่งแจ้งเตือนไม่สำเร็จ: "
            f"{exc}",
            file=sys.stderr,
        )

    print(
        f"[TRANSACTION] {transaction}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())