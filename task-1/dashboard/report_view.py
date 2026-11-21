"""Small offline renderer for the Markdown subset used by experiment reports."""
import html
import re
from urllib.parse import urlsplit


def inline(value):
    tokens = []

    def stash(markup):
        tokens.append(markup)
        return f'\x00{len(tokens) - 1}\x00'

    value = re.sub(r'`([^`]+)`', lambda m: stash('<code>' + html.escape(m[1]) + '</code>'), value)
    value = html.escape(value)

    def link(match):
        label, url = match.groups()
        url = html.unescape(url)
        if urlsplit(url).scheme not in ('', 'http', 'https', 'mailto'):
            return label
        if url.endswith('.md'):
            url = url[:-3] + '.html'
        return stash(f'<a href="{html.escape(url, quote=True)}">{label}</a>')

    value = re.sub(r'\[([^\]]+)\]\(([^\s)]+)\)', link, value)
    value = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', value)
    value = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'<em>\1</em>', value)
    return re.sub(r'\x00(\d+)\x00', lambda m: tokens[int(m[1])], value)


def render_report(source, siblings):
    lines = source.read_text(encoding='utf-8').splitlines()
    body, toc = [], []
    i = 0
    title = next((line[2:] for line in lines if line.startswith('# ')), source.stem)
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith('```'):
            code = []
            i += 1
            while i < len(lines) and not lines[i].startswith('```'):
                code.append(lines[i])
                i += 1
            body.append('<pre><code>' + html.escape('\n'.join(code)) + '</code></pre>')
        elif re.match(r'^#{1,6} ', line):
            marks, text = line.split(' ', 1)
            anchor = f'section-{len(toc)}'
            toc.append((len(marks), anchor, text))
            body.append(f'<h{len(marks)} id="{anchor}">{inline(text)}</h{len(marks)}>')
        elif i + 1 < len(lines) and '|' in line and re.fullmatch(r'[\s|:\-]+', lines[i + 1]) and '-' in lines[i + 1]:
            cells = lambda row: [cell.strip() for cell in row.strip().strip('|').split('|')]
            headers = cells(line)
            align = ['right' if cell.endswith(':') else 'left' for cell in cells(lines[i + 1])]
            table = ['<div class="table-scroll" role="region" aria-label="Tabela wyników" tabindex="0"><table><thead><tr>']
            table.extend(f'<th scope="col">{inline(cell)}</th>' for cell in headers)
            table.append('</tr></thead><tbody>')
            i += 2
            while i < len(lines) and '|' in lines[i] and lines[i].strip():
                table.append('<tr>')
                for column, cell in enumerate(cells(lines[i])):
                    alignment = align[column] if column < len(align) else 'left'
                    table.append(f'<td class="{alignment}">{inline(cell)}</td>')
                table.append('</tr>')
                i += 1
            table.append('</tbody></table></div>')
            body.append(''.join(table))
            continue
        elif re.match(r'^(?:[-*] |\d+\. )', line):
            ordered = bool(re.match(r'^\d+\. ', line))
            tag = 'ol' if ordered else 'ul'
            start = f' start="{int(line.split(".", 1)[0])}"' if ordered else ''
            pattern = r'^\d+\. ' if ordered else r'^[-*] '
            items = []
            while i < len(lines) and re.match(pattern, lines[i].strip()):
                items.append('<li>' + inline(re.sub(pattern, '', lines[i].strip())) + '</li>')
                i += 1
            body.append(f'<{tag}{start}>' + ''.join(items) + f'</{tag}>')
            continue
        else:
            body.append('<p>' + inline(line) + '</p>')
        i += 1
    navigation = ''.join(f'<a class="level-{level}" href="#{anchor}">{html.escape(text)}</a>' for level, anchor, text in toc if level > 1)
    related = ''.join(f'<a href="{html.escape(path.stem)}.html">{html.escape(path.stem.replace("_", " "))}</a>' for path in siblings if path != source)
    return f'''<!doctype html>
<html lang="pl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} · Raport eksperymentu</title><style>
:root{{color-scheme:light;--ink:#172b43;--muted:#5c6d80;--line:#dde5ee;--accent:#126a63}}
*{{box-sizing:border-box}}html{{scroll-behavior:smooth;scroll-padding-top:24px}}
body{{margin:0;background:#f3f6fa;color:var(--ink);font:16px/1.75 system-ui,-apple-system,"Segoe UI",sans-serif}}
a{{color:var(--accent);text-underline-offset:4px}}a:hover{{color:#103c67}}a:focus-visible,[tabindex]:focus-visible{{outline:3px solid #258eab;outline-offset:4px}}
.topbar{{max-width:1440px;margin:auto;padding:24px 40px;display:flex;gap:24px;flex-wrap:wrap;font-size:14px}}
.layout{{max-width:1440px;margin:0 auto 64px;padding:0 40px;display:grid;grid-template-columns:240px minmax(0,1fr);gap:32px}}
aside{{position:sticky;top:24px;align-self:start;font-size:14px;max-height:calc(100vh - 48px);overflow:auto}}
aside strong{{display:block;font-size:12px;text-transform:uppercase;letter-spacing:.12em;color:var(--muted);margin:20px 0 12px}}
aside a{{display:block;padding:7px 10px;text-decoration:none;border-left:2px solid var(--line);line-height:1.5}}
aside a:hover{{background:#e5eef0}}aside .level-3{{padding-left:24px;font-size:13px}}
article{{min-width:0;background:white;border:1px solid var(--line);border-radius:18px;padding:40px 48px;box-shadow:0 8px 32px #172b4305}}
.eyebrow{{color:var(--accent);font-size:12px;font-weight:700;letter-spacing:.12em;text-transform:uppercase}}
h1{{font-size:clamp(28px,3vw,40px);line-height:1.2;letter-spacing:-.035em;margin:16px 0 24px}}
h2{{font-size:25px;line-height:1.35;margin:48px 0 20px;padding-top:24px;border-top:1px solid var(--line)}}
h3{{font-size:20px;margin:32px 0 12px;color:var(--accent)}}p{{margin:14px 0;overflow-wrap:anywhere}}
h1+p{{font-size:18px;color:var(--muted);border-left:3px solid #6fb6ae;padding-left:18px}}
.table-scroll{{overflow-x:auto;margin:24px 0;border:1px solid var(--line);border-radius:10px}}
table{{border-collapse:collapse;width:100%;font-size:14px;line-height:1.5;font-variant-numeric:tabular-nums}}
th,td{{padding:13px 16px;border-bottom:1px solid var(--line);text-align:left}}th{{background:#eaf2f4;font-weight:650;white-space:nowrap}}
td:first-child{{font-weight:600}}tr:last-child td{{border-bottom:0}}tbody tr:nth-child(even){{background:#f8fafc}}tbody tr:hover{{background:#edf6f4}}td.right{{text-align:right;white-space:nowrap}}
code{{background:#edf2f7;border-radius:4px;padding:2px 5px;font: .86em/1.6 Consolas,monospace;overflow-wrap:anywhere}}
pre{{overflow:auto;background:#172b43;color:#edf6ff;padding:20px;border-radius:10px}}pre code{{background:none;padding:0;white-space:pre}}
footer{{margin-top:40px;padding-top:20px;border-top:1px solid var(--line);font-size:13px;color:var(--muted)}}
@media(max-width:1000px){{.layout{{grid-template-columns:1fr;padding:0 20px;gap:20px}}aside{{position:static;max-height:none}}aside nav{{display:flex;flex-wrap:wrap;gap:4px 12px}}aside .level-3{{display:none}}article{{padding:28px}}.topbar{{padding:20px}}}}
@media(max-width:540px){{article{{padding:22px 16px}}.layout{{padding:0 10px}}th,td{{padding:10px}}h2{{font-size:22px}}}}
@media print{{body{{background:white;font-size:11pt}}.topbar,aside{{display:none}}.layout{{display:block;padding:0;margin:0}}article{{border:0;box-shadow:none;padding:0}}.table-scroll{{overflow:visible}}h2,h3{{break-after:avoid}}tr{{break-inside:avoid}}}}
</style></head><body>
<header class="topbar"><a href="../../index.html">← Dashboard</a><a href="../../index-en.html" lang="en">English dashboard</a><a href="{html.escape(source.name)}" download>Pobierz Markdown ↓</a></header>
<div class="layout"><aside><strong>Spis treści</strong><nav aria-label="Spis treści">{navigation}</nav>{('<strong>Powiązane raporty</strong>' + related) if related else ''}</aside>
<article lang="en"><div class="eyebrow" lang="pl">Dziennik eksperymentów · {html.escape(source.parent.name.replace('_', ' '))}</div>{''.join(body)}
<footer lang="pl">Raport z zapisanych wyników eksperymentu. Wyniki holdout, cross-validation i publicznego leaderboardu należy interpretować osobno.</footer></article></div></body></html>'''
