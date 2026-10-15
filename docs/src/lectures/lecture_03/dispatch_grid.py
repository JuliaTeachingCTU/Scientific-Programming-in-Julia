"""Generate the two-argument dispatch grid of lecture 3: python3 dispatch_grid.py <outdir>.
Idea after M. Cox, "The Art of Multiple Dispatch", https://scientificcoder.com/the-art-of-multiple-dispatch"""
import sys, html
out = sys.argv[1]

TYPES = ["Grass", "Sheep", "Wolf"]          # rows: 1st argument, columns: 2nd argument
BG, CELL, TXT, MUTED = "#ffffff", "#e4e7eb", "#1f2328", "#6e7781"
GREEN, ORANGE, BLUE = "#8fd16a", "#f6a96b", "#4aa3f0"
TGREEN, TORANGE, TBLUE = "#2f7d12", "#b3550f", "#1764ad"   # darker shades for titles
CONFLICT = "url(#conflict)"

CW, CH, GAP = 66, 30, 4
LW, TH = 66, 20
PANEL_W = LW + 3 * (CW + GAP) + 24
TITLE_H = 44
PANEL_H = TITLE_H + TH + 3 * (CH + GAP) + 30

def text(x, y, s, size=12, color=MUTED, anchor="middle", mono=False, style="normal", weight="normal"):
    font = "JuliaMono, Menlo, monospace" if mono else "Helvetica, Arial, sans-serif"
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{font}" font-size="{size}" '
            f'font-style="{style}" font-weight="{weight}" fill="{color}">{html.escape(s)}</text>')

def panel(x0, y0, titles, fill, note):
    """titles: [(signature, color)], fill(r, c) -> color or None."""
    cx = x0 + LW + 1.5 * (CW + GAP)
    e = [text(cx, y0 + 16 + 17 * i, t, 13, col, mono=True) for i, (t, col) in enumerate(titles)]
    gy = y0 + TITLE_H
    e.append(text(x0 + LW - 6, gy + 14, "1st ╲ 2nd", 10, MUTED, anchor="end"))
    for c, name in enumerate(TYPES):
        e.append(text(x0 + LW + c * (CW + GAP) + CW / 2, gy + 14, name))
    for r, name in enumerate(TYPES):
        y = gy + TH + r * (CH + GAP)
        e.append(text(x0 + LW - 8, y + CH / 2 + 4, name, anchor="end"))
        for c in range(3):
            x = x0 + LW + c * (CW + GAP)
            f = fill(r, c) or CELL
            e.append(f'<rect x="{x}" y="{y}" width="{CW}" height="{CH}" rx="3" fill="{f}"/>')
            if f == CONFLICT:
                e.append(text(x + CW / 2, y + CH / 2 + 5, "?", 15, TXT, weight="bold"))
    e.append(text(cx, gy + TH + 3 * (CH + GAP) + 18, note, 12, MUTED, style="italic"))
    return "\n".join(e)

G, S, W = 0, 1, 2
animal = (S, W)
panels = [
    ([("eat!(::Sheep, ::Grass)", TGREEN)], lambda r, c: GREEN if (r, c) == (S, G) else None,
     "concrete types: one cell"),
    ([("eat!(::Wolf, ::Animal)", TGREEN)], lambda r, c: GREEN if r == W and c in animal else None,
     "abstract 2nd argument: a row"),
    ([("eat!(::Animal, ::Grass)", TGREEN)], lambda r, c: GREEN if c == G and r in animal else None,
     "abstract 1st argument: a column"),
    ([("mates(::T, ::T) where T<:Animal", TGREEN)], lambda r, c: GREEN if r == c and r in animal else None,
     "the same type twice: the diagonal"),
    ([("meet(::Wolf, ::Agent)", TGREEN), ("meet(::Agent, ::Sheep)", TORANGE)],
     lambda r, c: CONFLICT if (r, c) == (W, S) else (GREEN if r == W else (ORANGE if c == S else None)),
     "two methods overlap: ambiguity"),
    ([("+ meet(::Wolf, ::Sheep)", TBLUE)],
     lambda r, c: BLUE if (r, c) == (W, S) else (GREEN if r == W else (ORANGE if c == S else None)),
     "the more specific method resolves it"),
]

W_ = 3 * PANEL_W + 16
H_ = 2 * PANEL_H + 8
body = "\n".join(panel(8 + (i % 3) * PANEL_W, 4 + (i // 3) * PANEL_H, *p) for i, p in enumerate(panels))
svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W_} {H_}" width="{W_}" height="{H_}" role="img">\n'
       f'<title>Methods of a two-argument function cover cells of the type space: a cell, a row, a column, '
       f'the diagonal, and an ambiguity resolved by a more specific method.</title>\n'
       f'<defs><pattern id="conflict" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
       f'<rect width="8" height="8" fill="#d0d4da"/><rect width="3" height="8" fill="#9aa1ab"/></pattern></defs>\n'
       f'<rect width="{W_}" height="{H_}" rx="8" fill="{BG}"/>\n{body}\n</svg>\n')
open(f"{out}/dispatch_grid.svg", "w").write(svg)
