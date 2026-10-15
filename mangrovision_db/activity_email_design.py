"""Inline, image-free HTML for transactional activity emails."""

from html import escape
from urllib.parse import urlsplit


def safe_link(value):
    value = str(value or '').strip()
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
    except ValueError:
        return None
    if (parsed.scheme not in {'http', 'https'} or not hostname
            or parsed.username or parsed.password or any(c in value for c in '\r\n')):
        return None
    return value


def render_activity_email(*, kind, title, greeting, introduction, details,
                          access_lines=(), action_url=None, footer='MangroVision · Leganes, Iloilo'):
    """Escape every dynamic value; the matching plain-text body remains available."""
    e = lambda value: escape(str(value), quote=True)
    label, accent, tint = {
        'confirmation': ('Confirmed', '#166534', '#ecfdf3'),
        'update': ('Schedule updated', '#92400e', '#fff7e6'),
        'cancellation': ('Cancelled', '#991b1b', '#fff1f2'),
    }[kind]

    def rows(values):
        return ''.join(
            f'<tr><td style="padding:11px 14px;border-bottom:1px solid #e1ebe5;'
            f'vertical-align:top;width:32%;font-size:12px;color:#52675c">{e(key)}</td>'
            f'<td style="padding:11px 14px;border-bottom:1px solid #e1ebe5;'
            f'vertical-align:top;font-size:14px;font-weight:600;color:#173c2c;'
            f'overflow-wrap:anywhere">{e(value)}</td></tr>' for key, value in values)

    credentials = []
    notes = []
    for line in access_lines if kind != 'cancellation' else ():
        if line.startswith('Open: '):
            action_url = line.removeprefix('Open: ')
        elif line.startswith('Participant ') and ': ' in line:
            credentials.append(tuple(line.split(': ', 1)))
        elif line.startswith('Organization code: '):
            credentials.append(tuple(line.split(': ', 1)))
        elif line != 'Planter access':
            notes.append(line)
    action_url = safe_link(action_url) if kind != 'cancellation' else None
    access = ''
    if credentials:
        heading = 'Organization planting access' if any(key == 'Organization code' for key, _ in credentials) else 'Individual planter codes'
        access = (
            f'<h2 style="margin:24px 0 10px;font-size:18px;color:#173c2c">{heading}</h2>'
            '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" '
            f'style="background:#edf7f1;border:1px solid #d5e6db;border-radius:10px">{rows(credentials)}</table>'
            + ''.join(f'<p style="margin:10px 0;font-size:12px;line-height:1.6;color:#52675c">{e(note)}</p>' for note in notes))
    button = ''
    if action_url:
        button = (
            '<table role="presentation" cellpadding="0" cellspacing="0" style="margin:24px 0 12px">'
            '<tr><td bgcolor="#166534" style="border-radius:8px">'
            f'<a href="{e(action_url)}" style="display:inline-block;padding:14px 22px;'
            'color:#ffffff;font-size:14px;font-weight:700;text-decoration:none;border-radius:8px">'
            'Open planter workspace</a></td></tr></table>'
            f'<p style="font-size:11px;line-height:1.5;color:#63786d;overflow-wrap:anywhere">'
            f'Button not opening? <a href="{e(action_url)}" style="color:#166534">{e(action_url)}</a></p>')
    caution = ''
    if kind == 'cancellation':
        caution = '<p style="padding:14px;background:#fff1f2;border-left:3px solid #b91c1c;color:#991b1b;font-size:14px;line-height:1.6">This activity will not proceed at the time shown. Contact LGU staff to arrange another activity.</p>'
    footer_note = 'Share planting access only with your organization’s participants.' if credentials else 'Schedule times use Philippine time (Asia/Manila).'
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{e(title)}</title></head>
<body style="margin:0;padding:0;background:#eef4f0;font-family:Arial,Helvetica,sans-serif">
<div style="display:none;max-height:0;overflow:hidden;mso-hide:all">{e(label)}: {e(title)}</div>
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" bgcolor="#eef4f0"><tr><td align="center" style="padding:24px 12px">
<table role="presentation" width="620" cellpadding="0" cellspacing="0" style="width:100%;max-width:620px;background:#ffffff;border:1px solid #d9e5dd;border-radius:14px;overflow:hidden">
<tr><td bgcolor="#153f2d" style="padding:26px 28px;border-radius:14px 14px 0 0"><div style="font-size:26px;font-weight:700;color:#ffffff">Mangro<span style="color:#94d5b5">Vision</span></div><div style="margin-top:7px;font-size:11px;letter-spacing:1.5px;color:#c3e4d2">PLANTING &amp; COMMUNITY ACTIVITIES</div></td></tr>
<tr><td style="padding:28px"><span style="display:inline-block;padding:6px 10px;background:{tint};color:{accent};border-radius:20px;font-size:12px;font-weight:700">{e(label)}</span>
<h1 style="margin:16px 0 18px;font-size:24px;line-height:1.3;color:#173c2c">{e(title)}</h1>
<p style="font-size:14px;color:#334e40;line-height:1.7;margin:0 0 12px">{e(greeting)}</p>
<p style="font-size:14px;color:#52675c;line-height:1.7;margin:0 0 20px">{e(introduction)}</p>
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#f7faf8;border:1px solid #e1ebe5;border-radius:10px">{rows(details)}</table>
{caution}{access}{button}
<p style="margin:24px 0 0;padding-top:18px;border-top:1px solid #e1ebe5;font-size:12px;color:#52675c;line-height:1.7">For questions or changes, contact the LGU staff coordinating your activity.</p>
</td></tr><tr><td style="padding:16px 28px;background:#f7faf8;color:#63786d;font-size:11px;line-height:1.6;border-radius:0 0 14px 14px">{e(footer)}<br>{e(footer_note)}</td></tr>
</table></td></tr></table></body></html>'''
