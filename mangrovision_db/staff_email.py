"""Staff codes through the configured Supabase HTTPS sender, or local SMTP."""

import json
import os
import re
from pathlib import Path

import httpx

from .compat import get_connection
from .notifications import _send_smtp


def _cloud_transport():
    conn = get_connection()
    try:
        row = conn.execute('SELECT mangrovision.staff_email_transport() AS transport').fetchone()
        return json.loads(row['transport']) if row and row['transport'] else None
    finally:
        conn.close()


def send_verification_email(token: str, code: str, recipient: str, purpose: str) -> None:
    mode = os.getenv('STAFF_EMAIL_TRANSPORT', 'auto').strip().lower()
    if mode not in {'auto', 'cloud', 'smtp'}:
        raise RuntimeError('Invalid staff email transport configuration.')
    transport = None if mode == 'smtp' else _cloud_transport()
    if transport:
        # Never forward challenge secrets or authorization tokens to an
        # arbitrary URL or HTTP redirect supplied through configuration.
        if not re.fullmatch(r'https://[a-z0-9]{20}\.supabase\.co/functions/v1/staff-verification', transport.get('url', '')):
            raise RuntimeError('Invalid Supabase staff email endpoint.')
        with httpx.Client(timeout=30, follow_redirects=False) as client:
            response = client.post(transport['url'], headers={'x-mangrovision-staff-email-token':transport['token']},
                                   json={'challenge_token':token, 'code':code})
        if response.status_code != 200 or response.json().get('sent') is not True:
            raise RuntimeError('Cloud verification email was not accepted.')
        return
    if mode == 'cloud':
        raise RuntimeError('Cloud verification email is not configured.')
    template_path = Path(__file__).resolve().parents[1] / 'supabase/functions/staff-verification/email-template.json'
    template = json.loads(template_path.read_text(encoding='utf-8'))
    if purpose not in template['copy'] or not re.fullmatch(r'[0-9]{6}', code):
        raise ValueError('Invalid verification email.')
    values = {**template['copy'][purpose], 'code': code}

    def render(body):
        return re.sub(r'\{\{([a-z]+)\}\}', lambda match: values[match[1]], body)

    _send_smtp(recipient, f"MangroVision {values['label']} code", render(template['text']),
               html_body=render('\n'.join(template['html'])))
