"""A tiny dependency-free SVG plotting helper (line / scatter charts with error bars and log axes, and grouped bar charts).

Kept deliberately small so the toolkit needs nothing beyond NumPy.
"""
import math
from xml.sax.saxutils import escape

from ._util import Checker

PALETTE = ["#2b6cb0", "#dd6b20", "#2f855a", "#805ad5", "#c53030", "#718096"]
FONT = "font-family='Helvetica, Arial, sans-serif'"   # reference-line labels get a white halo so they stay readable over bars


def nice_ticks(lo, hi, n=5):
    """Round tick values covering [lo, hi]."""
    if hi <= lo:
        hi = lo + 1.0
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw), default=10 * mag)
    start = math.floor(lo / step + 1e-9) * step
    stop = math.ceil(hi / step - 1e-9) * step
    k = int(round((stop - start) / step))
    return [round(start + i * step, 12) for i in range(k + 1)]


def log_ticks(lo, hi):
    return [10.0 ** k for k in range(int(math.floor(math.log10(lo))), int(math.ceil(math.log10(hi))) + 1)]


def _fmt(v):
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 1e9:
        return f"{v / 1e9:g}bn"
    if a >= 1e6:
        return f"{v / 1e6:g}m"
    if a >= 1e3:
        return f"{v / 1e3:g}k"
    return f"{v:g}"


class Figure:
    def __init__(self, title, xlabel="", ylabel="", width=720, height=420, xlog=False, ylog=False, xfmt=_fmt, yfmt=_fmt, legend="top left"):
        self.title, self.xlabel, self.ylabel, self.legend = title, xlabel, ylabel, legend
        self.W, self.H, self.xlog, self.ylog, self.xfmt, self.yfmt = width, height, xlog, ylog, xfmt, yfmt
        self.ml, self.mr, self.mt, self.mb = 70, 24, 44, 56
        self.series, self.hlines, self.notes = [], [], []

    def line(self, xs, ys, label=None, color=None, dashed=False, markers=True, yerr=None):
        self.series.append(dict(xs=list(map(float, xs)), ys=list(map(float, ys)), label=label, color=color, dashed=dashed, markers=markers,
                                yerr=None if yerr is None else list(map(float, yerr)), connect=True))
        return self

    def points(self, xs, ys, label=None, color=None, yerr=None):
        self.series.append(dict(xs=list(map(float, xs)), ys=list(map(float, ys)), label=label, color=color, dashed=False, markers=True,
                                yerr=None if yerr is None else list(map(float, yerr)), connect=False))
        return self

    def hline(self, y, label=None, color="#a0aec0", label_side="right"):
        self.hlines.append((float(y), label, color, label_side))
        return self

    def note(self, text):
        self.notes.append(text)
        return self

    def _ranges(self):
        xs = [x for s in self.series for x in s["xs"]]
        ys = [y for s in self.series for y in s["ys"]] + [h[0] for h in self.hlines]                          # (y, label, color, side)
        ys += [y + e for s in self.series if s["yerr"] for y, e in zip(s["ys"], s["yerr"])]
        ys += [y - e for s in self.series if s["yerr"] for y, e in zip(s["ys"], s["yerr"])]
        return min(xs), max(xs), min(ys), max(ys)

    def svg(self):
        x0, x1, y0, y1 = self._ranges()
        xt = log_ticks(x0, x1) if self.xlog else nice_ticks(x0, x1)
        yt = log_ticks(y0, y1) if self.ylog else nice_ticks(min(y0, 0) if y0 > 0 and y0 < 0.3 * y1 else y0, y1)
        xa, xb, ya, yb = xt[0], xt[-1], yt[0], yt[-1]
        pw, ph = self.W - self.ml - self.mr, self.H - self.mt - self.mb
        fx = (lambda v: self.ml + (math.log10(v) - math.log10(xa)) / (math.log10(xb) - math.log10(xa)) * pw) if self.xlog else \
             (lambda v: self.ml + (v - xa) / (xb - xa) * pw)
        fy = (lambda v: self.mt + ph - (math.log10(v) - math.log10(ya)) / (math.log10(yb) - math.log10(ya)) * ph) if self.ylog else \
             (lambda v: self.mt + ph - (v - ya) / (yb - ya) * ph)
        out = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{self.W}' height='{self.H}' viewBox='0 0 {self.W} {self.H}'>",
               f"<rect width='100%' height='100%' fill='white'/>",
               f"<text x='{self.W / 2}' y='24' text-anchor='middle' {FONT} font-size='16' font-weight='bold'>{escape(self.title)}</text>"]
        for t in yt:
            y = fy(t)
            out.append(f"<line x1='{self.ml}' x2='{self.ml + pw}' y1='{y:.1f}' y2='{y:.1f}' stroke='#edf2f7'/>")
            out.append(f"<text x='{self.ml - 8}' y='{y + 4:.1f}' text-anchor='end' {FONT} font-size='11' fill='#4a5568'>{escape(self.yfmt(t))}</text>")
        for t in xt:
            x = fx(t)
            out.append(f"<line x1='{x:.1f}' x2='{x:.1f}' y1='{self.mt + ph}' y2='{self.mt + ph + 4}' stroke='#4a5568'/>")
            out.append(f"<text x='{x:.1f}' y='{self.mt + ph + 18}' text-anchor='middle' {FONT} font-size='11' fill='#4a5568'>{escape(self.xfmt(t))}</text>")
        out.append(f"<rect x='{self.ml}' y='{self.mt}' width='{pw}' height='{ph}' fill='none' stroke='#a0aec0'/>")
        for y, label, color, side in self.hlines:
            yy = fy(y)
            out.append(f"<line x1='{self.ml}' x2='{self.ml + pw}' y1='{yy:.1f}' y2='{yy:.1f}' stroke='{color}' stroke-dasharray='5,4'/>")
            if label:
                lx, anchor = (self.ml + pw - 4, "end") if side == "right" else (self.ml + 4, "start")
                out.append(f"<text x='{lx}' y='{yy - 5:.1f}' text-anchor='{anchor}' {FONT} font-size='11' fill='{color}' stroke='white' stroke-width='3' paint-order='stroke'>{escape(label)}</text>")
        legend = []
        for k, s in enumerate(self.series):
            c = s["color"] or PALETTE[k % len(PALETTE)]
            pts = [(fx(x), fy(y)) for x, y in zip(s["xs"], s["ys"])]
            if s["connect"] and len(pts) > 1:
                d = " ".join(f"{'M' if i == 0 else 'L'}{px:.1f},{py:.1f}" for i, (px, py) in enumerate(pts))
                dash = " stroke-dasharray='6,4'" if s["dashed"] else ""
                out.append(f"<path d='{d}' fill='none' stroke='{c}' stroke-width='2'{dash}/>")
            if s["yerr"]:
                for (px, _), y, e in zip(pts, s["ys"], s["yerr"]):
                    out.append(f"<line x1='{px:.1f}' x2='{px:.1f}' y1='{fy(y - e):.1f}' y2='{fy(y + e):.1f}' stroke='{c}' stroke-width='1.2'/>")
            if s["markers"]:
                out += [f"<circle cx='{px:.1f}' cy='{py:.1f}' r='3.5' fill='{c}'/>" for px, py in pts]
            if s["label"]:
                legend.append((s["label"], c))
        box_w = 30 + 6.5 * max((len(label) for label, _ in legend), default=0)          # approximate width of 12 px text
        lx = self.ml + pw - 12 - box_w if "right" in self.legend else self.ml + 12
        ly = self.mt + ph - 18 * len(legend) if "bottom" in self.legend else self.mt + 14
        for i, (label, c) in enumerate(legend):
            y = ly + 18 * i
            out.append(f"<rect x='{lx:.1f}' y='{y - 9}' width='12' height='12' fill='{c}'/>")
            out.append(f"<text x='{lx + 18:.1f}' y='{y + 1}' {FONT} font-size='12'>{escape(label)}</text>")
        out.append(f"<text x='{self.ml + pw / 2}' y='{self.H - 26}' text-anchor='middle' {FONT} font-size='12'>{escape(self.xlabel)}</text>")
        out.append(f"<text transform='translate(16,{self.mt + ph / 2}) rotate(-90)' text-anchor='middle' {FONT} font-size='12'>{escape(self.ylabel)}</text>")
        for i, n in enumerate(self.notes):
            out.append(f"<text x='{self.ml}' y='{self.H - 8 + 0 * i}' {FONT} font-size='10' fill='#718096'>{escape(n)}</text>")
        out.append("</svg>")
        return "\n".join(out)


def bar_chart(title, categories, groups, ylabel="", width=720, height=420, hlines=(), yfmt=_fmt, note=None, ylog=False):
    """groups: {name: [value per category]}; grouped vertical bars, values written above bars."""
    names = list(groups)
    vals = [v for g in groups.values() for v in g]
    if ylog:
        yt = log_ticks(min(vals), max(vals))
    else:
        top = max(0.0, max(vals), *[h[0] for h in hlines])
        yt = nice_ticks(min(0.0, min(vals), *[h[0] for h in hlines]), top * (1.15 if len(names) > 1 else 1.0))   # room for the legend
    ml, mr, mt, mb = 70, 24, 44, 70
    pw, ph = width - ml - mr, height - mt - mb
    ya, yb = yt[0], yt[-1]
    fy = (lambda v: mt + ph - (math.log10(v) - math.log10(ya)) / (math.log10(yb) - math.log10(ya)) * ph) if ylog else (lambda v: mt + ph - (v - ya) / (yb - ya) * ph)
    out = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
           "<rect width='100%' height='100%' fill='white'/>",
           f"<text x='{width / 2}' y='24' text-anchor='middle' {FONT} font-size='16' font-weight='bold'>{escape(title)}</text>"]
    for t in yt:
        y = fy(t)
        out.append(f"<line x1='{ml}' x2='{ml + pw}' y1='{y:.1f}' y2='{y:.1f}' stroke='#edf2f7'/>")
        out.append(f"<text x='{ml - 8}' y='{y + 4:.1f}' text-anchor='end' {FONT} font-size='11' fill='#4a5568'>{escape(yfmt(t))}</text>")
    base = fy(ya if ylog else 0.0)
    slot = pw / len(categories)
    bw = slot * 0.8 / len(names)
    for ci, cat in enumerate(categories):
        for gi, name in enumerate(names):
            v = groups[name][ci]
            x = ml + ci * slot + slot * 0.1 + gi * bw
            y = fy(v)
            top, h = min(y, base), abs(base - y)
            c = PALETTE[gi % len(PALETTE)]
            out.append(f"<rect x='{x:.1f}' y='{top:.1f}' width='{bw * 0.92:.1f}' height='{h:.1f}' fill='{c}'/>")
            out.append(f"<text x='{x + bw * 0.46:.1f}' y='{(top - 4) if v >= 0 or ylog else (top + h + 12):.1f}' text-anchor='middle' {FONT} font-size='10'>{escape(yfmt(v))}</text>")
        out.append(f"<text x='{ml + ci * slot + slot / 2:.1f}' y='{mt + ph + 18}' text-anchor='middle' {FONT} font-size='11'>{escape(cat)}</text>")
    out.append(f"<line x1='{ml}' x2='{ml + pw}' y1='{base:.1f}' y2='{base:.1f}' stroke='#4a5568'/>")
    for y, label, color in hlines:
        yy = fy(y)
        out.append(f"<line x1='{ml}' x2='{ml + pw}' y1='{yy:.1f}' y2='{yy:.1f}' stroke='{color}' stroke-dasharray='5,4'/>")
        out.append(f"<text x='{ml + pw - 4}' y='{yy - 5:.1f}' text-anchor='end' {FONT} font-size='11' fill='{color}' stroke='white' stroke-width='3' paint-order='stroke'>{escape(label)}</text>")
    if len(names) > 1:
        for gi, name in enumerate(names):
            x = ml + 12 + gi * 170
            out.append(f"<rect x='{x}' y='{mt + 6}' width='12' height='12' fill='{PALETTE[gi % len(PALETTE)]}'/>")
            out.append(f"<text x='{x + 18}' y='{mt + 16}' {FONT} font-size='12'>{escape(name)}</text>")
    out.append(f"<text transform='translate(16,{mt + ph / 2}) rotate(-90)' text-anchor='middle' {FONT} font-size='12'>{escape(ylabel)}</text>")
    if note:
        out.append(f"<text x='{ml}' y='{height - 10}' {FONT} font-size='10' fill='#718096'>{escape(note)}</text>")
    out.append("</svg>")
    return "\n".join(out)


def selftest():
    import xml.etree.ElementTree as ET
    check = Checker()
    t = nice_ticks(0.013, 0.087)
    check(f"nice ticks cover the range with round steps ({t})", t[0] <= 0.013 and t[-1] >= 0.087 and len(set(round(b - a, 10) for a, b in zip(t, t[1:]))) == 1)
    check("log ticks are powers of ten covering the range", log_ticks(3, 2000) == [1.0, 10.0, 100.0, 1000.0, 10000.0])
    s = Figure("t", "x", "y").line([1, 2, 3], [0.1, 0.4, 0.2], label="a").points([1, 2], [0.3, 0.2], yerr=[0.05, 0.05], label="b").hline(0.25, "ref").svg()
    root = ET.fromstring(s)
    ns = "{http://www.w3.org/2000/svg}"
    check("line chart is well-formed SVG with a path, markers, error bars and a legend",
          root.tag == ns + "svg" and len(root.findall(ns + "path")) == 1 and len(root.findall(ns + "circle")) == 5 and s.count("stroke-dasharray") >= 1)
    b = bar_chart("t", ["A", "B"], {"g1": [1.0, -0.5], "g2": [0.3, 0.7]}, hlines=[(0.5, "ref", "#c53030")])
    check("bar chart is well-formed with one bar per value (+ legend swatches)", len(ET.fromstring(b).findall(ns + "rect")) == 1 + 4 + 2)
    lg = Figure("t", xlog=True).line([1e5, 1e6, 1e9], [1, 2, 3]).svg()
    check("log-x chart renders finite coordinates", "nan" not in lg and "inf" not in lg)
    f = Figure("t", legend="bottom right").line([0, 1], [0, 1], label="a long legend label")
    sw = ET.fromstring(f.svg()).findall(ns + "rect")[-1]                              # the legend swatch
    x, y = float(sw.get("x")), float(sw.get("y"))
    check(f"legend 'bottom right' stays inside the plot area, in its lower-right quarter ({x:.0f}, {y:.0f})",
          f.ml + (f.W - f.ml - f.mr) / 2 < x < f.W - f.mr and f.mt + (f.H - f.mt - f.mb) / 2 < y < f.H - f.mb)
    return check.summary()


if __name__ == "__main__":
    import sys
    sys.exit(0 if selftest() else 1)
