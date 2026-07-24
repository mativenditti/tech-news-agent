"""Render mínimo de markdown→HTML para el cuerpo del email.

El LLM genera prosa markdown-ish (encabezados, listas, links, bold/italic). Este
módulo la convierte a un HTML simple y seguro, sin dependencias externas.

Seguridad: el body puede contener texto traído de la web (web_search), así que
TODO el contenido se escapa con html.escape ANTES de reintroducir las etiquetas
que nosotros generamos. Nunca se emiten tags provenientes del input crudo.
"""

from __future__ import annotations

import html
import re

# Estilos inline: los clientes de correo suelen ignorar <style>/<head>.
_WRAPPER = (
    '<div style="font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;'
    'font-size:15px;line-height:1.5;color:#1a1a1a;max-width:640px">\n{body}\n</div>'
)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_ULIST_RE = re.compile(r"^\s*[-*]\s+(.*)$")
_OLIST_RE = re.compile(r"^\s*\d+[.)]\s+(.*)$")
# Inline: se aplican sobre texto YA escapado (no rompen el escape porque los
# metacaracteres markdown no colisionan con las entidades HTML).
_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^\s)]+)\)")
_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")
_ITALIC_RE = re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)")


def _looks_like_html(text: str) -> str:
    """True si el body ya parece un documento HTML (empieza con <html>/<!doctype>)."""
    head = text.lstrip()[:64].lower()
    return head.startswith("<!doctype") or head.startswith("<html")


def _inline(text: str) -> str:
    """Aplica links/bold/italic sobre texto ya escapado en HTML."""
    text = _LINK_RE.sub(
        lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>', text
    )
    text = _BOLD_RE.sub(r"<strong>\1</strong>", text)
    text = _ITALIC_RE.sub(r"<em>\1</em>", text)
    return text


def markdown_to_html(text: str) -> str:
    """Convierte markdown simple a un fragmento HTML seguro y estilado.

    Cubre encabezados (#..######), listas (- / * / 1.), links [t](url),
    **bold**, *italic* y párrafos separados por líneas en blanco. Cualquier
    otro contenido se trata como texto plano escapado.

    Defensa en profundidad: si el body ya viene como un documento HTML (el LLM
    debería mandar markdown, pero a veces se desvía), lo devolvemos tal cual sin
    escaparlo, para no terminar mostrando las etiquetas como texto en el cliente.
    """
    if _looks_like_html(text):
        return text

    out: list[str] = []
    list_tag: str | None = None  # "ul" | "ol" | None
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            out.append("<p>" + "<br>".join(paragraph) + "</p>")
            paragraph.clear()

    def close_list() -> None:
        nonlocal list_tag
        if list_tag:
            out.append(f"</{list_tag}>")
            list_tag = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip()

        if not line.strip():
            flush_paragraph()
            close_list()
            continue

        heading = _HEADING_RE.match(line)
        if heading:
            flush_paragraph()
            close_list()
            level = len(heading.group(1))
            content = _inline(html.escape(heading.group(2)))
            out.append(f"<h{level}>{content}</h{level}>")
            continue

        ulist = _ULIST_RE.match(line)
        olist = _OLIST_RE.match(line)
        if ulist or olist:
            flush_paragraph()
            want = "ul" if ulist else "ol"
            if list_tag != want:
                close_list()
                out.append(f"<{want}>")
                list_tag = want
            item = (ulist or olist).group(1)
            out.append(f"<li>{_inline(html.escape(item))}</li>")
            continue

        # Línea normal: parte de un párrafo.
        close_list()
        paragraph.append(_inline(html.escape(line)))

    flush_paragraph()
    close_list()

    return _WRAPPER.format(body="\n".join(out))
