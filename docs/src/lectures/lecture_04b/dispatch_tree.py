"""Generate the animal-world dispatch grid of lecture 4b: python3 dispatch_tree.py <outdir>.
Idea after M. Cox, "The Art of Multiple Dispatch", https://scientificcoder.com/the-art-of-multiple-dispatch"""
import sys, html
out = sys.argv[1]

ROWS = ["birds", "mammals", "insects"]
COLS = ["swims", "flies", "walks"]
CELLS = [["penguin", "eagle", "kiwi"],
         ["dolphin", "bat", "sheep, wolf"],
         ["backswimmer", "fly", "beetle"]]

BG, CELL, TXT, MUTED = "#ffffff", "#e4e7eb", "#1f2328", "#6e7781"
ROW, COL, BOTH = "#8fd16a", "#f6a96b", "#4aa3f0"   # green row, orange column, blue cell
CONFLICT = "url(#conflict)"

CW, CH, GAP = 92, 34, 4          # cell size
LW, TH = 62, 22                  # row-label width, column-label height
PANEL_W = LW + 3 * (CW + GAP) + 16
TITLE_H = 30

def panel(x0, y0, title, colors, note=None):
    """colors[r][c] -> fill or None."""
    e = []
    e.append(f'<text x="{x0 + LW + 1.5*(CW+GAP)}" y="{y0 + 18}" text-anchor="middle" '
             f'font-family="JuliaMono, Menlo, monospace" font-size="14" fill="{TXT}">{html.escape(title)}</text>')
    gy = y0 + TITLE_H
    for c, name in enumerate(COLS):
        e.append(f'<text x="{x0 + LW + c*(CW+GAP) + CW/2}" y="{gy + 15}" text-anchor="middle" '
                 f'font-family="Helvetica, Arial, sans-serif" font-size="12" fill="{MUTED}">{name}</text>')
    for r, name in enumerate(ROWS):
        y = gy + TH + r*(CH+GAP)
        e.append(f'<text x="{x0 + LW - 8}" y="{y + CH/2 + 4}" text-anchor="end" '
                 f'font-family="Helvetica, Arial, sans-serif" font-size="12" fill="{MUTED}">{name}</text>')
        for c in range(3):
            x = x0 + LW + c*(CW+GAP)
            fill = colors[r][c] or CELL
            label = CELLS[r][c] + (" ?" if fill == CONFLICT else "")
            e.append(f'<rect x="{x}" y="{y}" width="{CW}" height="{CH}" rx="3" fill="{fill}"/>')
            e.append(f'<text x="{x + CW/2}" y="{y + CH/2 + 4}" text-anchor="middle" '
                     f'font-family="Helvetica, Arial, sans-serif" font-size="12" '
                     f'font-weight="{"bold" if fill == CONFLICT else "normal"}" fill="{TXT}">{label}</text>')
    if note:
        ny = gy + TH + 3*(CH+GAP) + 16
        e.append(f'<text x="{x0 + LW + 1.5*(CW+GAP)}" y="{ny}" text-anchor="middle" '
                 f'font-family="Helvetica, Arial, sans-serif" font-size="12" font-style="italic" fill="{MUTED}">{html.escape(note)}</text>')
    return "\n".join(e)

def grid(fn):
    return [[fn(r, c) for c in range(3)] for r in range(3)]

def figure(name, panels, desc):
    n = len(panels)
    W = n * PANEL_W + 16
    H = TITLE_H + TH + 3*(CH+GAP) + 34
    body = "\n".join(panel(8 + i*PANEL_W, 4, *p) for i, p in enumerate(panels))
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img">\n'
           f'<title>{html.escape(desc)}</title>\n'
           f'<defs><pattern id="conflict" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
           f'<rect width="8" height="8" fill="#d0d4da"/><rect width="3" height="8" fill="#9aa1ab"/></pattern></defs>\n'
           f'<rect width="{W}" height="{H}" rx="8" fill="{BG}"/>\n{body}\n</svg>\n')
    open(f"{out}/{name}", "w").write(svg)

row  = grid(lambda r, c: ROW if r == 0 else None)
col  = grid(lambda r, c: COL if c == 1 else None)
both_tree = grid(lambda r, c: CONFLICT if (r == 0 and c == 1) else (ROW if r == 0 else (COL if c == 1 else None)))

figure("dispatch_tree.svg", [
    ("f(::Bird)", row, "tree by taxonomy: a row"),
    ("f(::Flyer)", col, "tree by movement: a column"),
    ("f(::Bird) and f(::Flyer)?", both_tree, "one parent: is eagle a Bird or a Flyer?"),
], "An abstract type of a single type tree selects a row (taxonomy) or a column (movement), not both.")
