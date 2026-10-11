"""Booking email transport checks with simulated providers; no real email."""

import json

import httpx
import pytest

from mangrovision_db import appointment_email as delivery


@pytest.fixture(autouse=True)
def isolated_sender(monkeypatch):
    for name in ('BREVO_API_KEY', 'SMTP_FROM', 'SMTP_HOST', 'SMTP_USERNAME', 'SMTP_PASSWORD'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('BREVO_API_KEY', 'xkeysib-test-only')
    monkeypatch.setenv('SMTP_FROM', 'sender@example.org')
    monkeypatch.setattr(delivery, 'get_connection', lambda: pytest.fail('Transport test accessed the database'))
    monkeypatch.setattr(delivery, '_send_smtp', lambda *args: pytest.fail('Unexpected SMTP connection'))
    monkeypatch.setattr(delivery.httpx, 'Client', lambda **kwargs: pytest.fail('Unexpected network connection'))


# Capture before patching the shared httpx module in the fixture.
REAL_CLIENT = httpx.Client


def mock_provider(monkeypatch, handler):
    settings = {}

    def client(**kwargs):
        settings.update(kwargs)
        return REAL_CLIENT(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(delivery.httpx, 'Client', client)
    return settings


def test_https_configuration_does_not_require_smtp_login(monkeypatch):
    assert delivery._sender_configured()
    monkeypatch.delenv('SMTP_FROM')
    assert not delivery._sender_configured()
    with pytest.raises(ValueError, match='SMTP_FROM'):
        delivery._send_confirmation('recipient@example.org', 'Subject', 'Body')


def test_empty_api_key_uses_existing_smtp_transport(monkeypatch):
    monkeypatch.setenv('BREVO_API_KEY', ' ')
    assert not delivery._sender_configured()
    for name in ('SMTP_HOST', 'SMTP_USERNAME', 'SMTP_PASSWORD'):
        monkeypatch.setenv(name, 'test-only')
    assert delivery._sender_configured()
    calls = []
    monkeypatch.setattr(delivery, '_send_smtp', lambda *args: calls.append(args))
    delivery._send_confirmation('recipient@example.org', 'Subject', 'Body')
    assert calls == [('recipient@example.org', 'Subject', 'Body')]


def test_smtp_key_in_api_setting_is_rejected_before_network(monkeypatch):
    monkeypatch.setenv('BREVO_API_KEY', 'xsmtpsib-test-only')
    with pytest.raises(ValueError, match='API keys page'):
        delivery._send_confirmation('recipient@example.org', 'Subject', 'Body')


def test_https_confirmation_payload_and_receipt(monkeypatch):
    requests = []

    def accepted(request):
        requests.append(request)
        return httpx.Response(201, json={'messageId': '<test-receipt@example.org>'})

    settings = mock_provider(monkeypatch, accepted)
    delivery._send_confirmation('recipient@example.org', 'Appointment confirmed', 'Private login details')
    assert settings == {'timeout': 30, 'follow_redirects': False}
    assert len(requests) == 1
    request = requests[0]
    assert request.method == 'POST'
    assert str(request.url) == 'https://api.brevo.com/v3/smtp/email'
    assert request.headers['api-key'] == 'xkeysib-test-only'
    assert request.headers['content-type'] == 'application/json'
    assert json.loads(request.content) == {
        'sender': {'name': 'MangroVision', 'email': 'sender@example.org'},
        'to': [{'email': 'recipient@example.org'}],
        'subject': 'Appointment confirmed',
        'textContent': 'Private login details',
    }


@pytest.mark.parametrize('status', [200, 400, 401, 429, 500])
def test_provider_rejection_is_a_sanitized_failure(monkeypatch, status):
    mock_provider(monkeypatch, lambda request: httpx.Response(
        status, json={'message': 'Private login details for recipient@example.org; xkeysib-test-only'}))
    with pytest.raises(RuntimeError) as error:
        delivery._send_confirmation('recipient@example.org', 'Subject', 'Body')
    assert str(error.value) == f'Brevo did not accept the confirmation email (HTTP {status}).'


def test_redirect_is_not_followed(monkeypatch):
    urls = []

    def redirected(request):
        urls.append(str(request.url))
        return httpx.Response(307, headers={'location': 'https://other.example.org/email'})

    mock_provider(monkeypatch, redirected)
    with pytest.raises(RuntimeError, match='HTTP 307'):
        delivery._send_confirmation('recipient@example.org', 'Subject', 'Body')
    assert urls == ['https://api.brevo.com/v3/smtp/email']


@pytest.mark.parametrize('receipt', [{}, [], {'messageId': None}, {'messageId': ''}, {'messageId': ' '}])
def test_missing_receipt_is_a_failure(monkeypatch, receipt):
    mock_provider(monkeypatch, lambda request: httpx.Response(201, json=receipt))
    with pytest.raises(RuntimeError, match='no confirmation-email receipt'):
        delivery._send_confirmation('recipient@example.org', 'Subject', 'Body')


def test_invalid_receipt_json_is_a_sanitized_failure(monkeypatch):
    mock_provider(monkeypatch, lambda request: httpx.Response(201, text='Private provider details'))
    with pytest.raises(RuntimeError, match='invalid confirmation-email receipt'):
        delivery._send_confirmation('recipient@example.org', 'Subject', 'Body')


def test_timeout_reaches_worker_for_retry(monkeypatch):
    def timed_out(request):
        raise httpx.ReadTimeout('Provider unavailable', request=request)

    mock_provider(monkeypatch, timed_out)
    with pytest.raises(httpx.ReadTimeout):
        delivery._send_confirmation('recipient@example.org', 'Subject', 'Body')


def test_html_and_text_are_both_sent_to_brevo(monkeypatch):
    payloads = []
    def accepted(request):
        payloads.append(json.loads(request.content))
        return httpx.Response(201, json={'messageId': 'test-receipt'})
    mock_provider(monkeypatch, accepted)
    delivery._send_confirmation('recipient@example.org', 'Confirmed', 'Plain details', html_body='<h1>Confirmed</h1>')
    assert payloads[0]['textContent'] == 'Plain details'
    assert payloads[0]['htmlContent'] == '<h1>Confirmed</h1>'


def test_html_uses_multipart_smtp_fallback(monkeypatch):
    from mangrovision_db import notifications
    monkeypatch.delenv('BREVO_API_KEY')
    monkeypatch.setenv('SMTP_HOST', 'smtp.example.org')
    monkeypatch.setenv('SMTP_USERNAME', 'test-only')
    monkeypatch.setenv('SMTP_PASSWORD', 'test-only')
    monkeypatch.setenv('SMTP_PORT', '587')
    messages = []
    class FakeSMTP:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def starttls(self): pass
        def login(self, *args): pass
        def send_message(self, message): messages.append(message)
    monkeypatch.setattr(notifications.smtplib, 'SMTP', FakeSMTP)
    monkeypatch.setattr(delivery, '_send_smtp', notifications._send_smtp)
    delivery._send_confirmation('recipient@example.org', 'Confirmed', 'Plain details', html_body='<h1>Confirmed</h1>')
    assert messages[0].get_content_type() == 'multipart/alternative'
    assert messages[0].get_body(preferencelist=('plain',)).get_content().strip() == 'Plain details'
    assert '<h1>Confirmed</h1>' in messages[0].get_body(preferencelist=('html',)).get_content()


def test_email_design_escapes_details_credentials_and_unsafe_links():
    from mangrovision_db.activity_email_design import render_activity_email
    html = render_activity_email(kind='confirmation', title='<script>bad</script>', greeting='Hello <img>',
        introduction='A & B', details=[('Organization', '<img src=x onerror=bad()>')],
        access_lines=['Planter access', 'Open: javascript:alert(1)', 'Participant 1: <b>team</b>', 'Participant 2: <private>&"'])
    assert '<script>' not in html and '<img' not in html and 'javascript:' not in html
    assert '&lt;script&gt;' in html and '&lt;private&gt;&amp;&quot;' in html
    assert 'Individual planter codes' in html and 'Open planter workspace' not in html


def test_cancelled_email_hides_account_details_and_access_button():
    from mangrovision_db.activity_email_design import render_activity_email
    html = render_activity_email(kind='cancellation', title='Group planting', greeting='Hello team',
        introduction='Cancelled', details=[('Time', '08:00 AM')],
        access_lines=['Participant 1: private-test', 'Open: https://example.org/field'])
    assert 'This activity will not proceed' in html
    assert 'private-test' not in html and 'test-group' not in html and '/field' not in html


def test_readable_credentials_keep_the_organization_name_and_independent_secret():
    from mangrovision_db.readable_credentials import organization_password, organization_slug, WORDS
    import math
    assert organization_slug('  Otón Youth & Volunteers  ', 1) == 'oton-youth-volunteers'
    assert organization_slug('!!!', 7) == 'organization-7'
    passwords = {organization_password('Oton Youth') for _ in range(30)}
    assert len(passwords) == 30
    assert all(password.startswith('OtonYouth-') and password.split('-')[-1].isdigit() for password in passwords)
    assert 4 * math.log2(len(set(WORDS))) + math.log2(100_000_000) > 54
