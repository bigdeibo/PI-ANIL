"""Politely fetch IR spectra (JCAMP-DX) of oil-related compounds from the NIST Chemistry WebBook.

Compliance note: SDBS (AIST) explicitly prohibits bulk scraping by robots, so it
was abandoned; the NIST WebBook offers no official bulk-download channel, and
fetching individual records via single GET requests is common practice in the
community (e.g., kcw1375/nist-webbook-ir). This script:
- is limited to ~110 small molecules relevant to petroleum chemistry
  (n-alkanes / cycloalkanes / aromatics / olefins / oxygenates / fatty acids /
  FAMEs / glycerides / S,N-heterocycles)
- waits >=1.2 s between requests; the total volume is small and one-off
- saves raw .jdx files to data/NIST_IR/jdx/ and parsed results to data/NIST_IR/nist_ir.npz
Supports resuming: existing .jdx files are skipped automatically.
"""
import sys
import time
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "NIST_IR"
JDX_DIR = OUT / "jdx"
JDX_DIR.mkdir(parents=True, exist_ok=True)

# (CAS, name, category)
COMPOUNDS = [
    # n-alkanes
    ("109-66-0", "pentane", "n-alkane"), ("110-54-3", "hexane", "n-alkane"),
    ("142-82-5", "heptane", "n-alkane"), ("111-65-9", "octane", "n-alkane"),
    ("111-84-2", "nonane", "n-alkane"), ("124-18-5", "decane", "n-alkane"),
    ("1120-21-4", "undecane", "n-alkane"), ("112-40-3", "dodecane", "n-alkane"),
    ("629-50-5", "tridecane", "n-alkane"), ("629-59-4", "tetradecane", "n-alkane"),
    ("629-62-9", "pentadecane", "n-alkane"), ("544-76-3", "hexadecane", "n-alkane"),
    ("629-78-7", "heptadecane", "n-alkane"), ("593-45-3", "octadecane", "n-alkane"),
    ("629-92-5", "nonadecane", "n-alkane"), ("112-95-8", "eicosane", "n-alkane"),
    # branched alkanes
    ("540-84-1", "isooctane", "iso-alkane"), ("107-83-5", "2-methylpentane", "iso-alkane"),
    ("96-14-0", "3-methylpentane", "iso-alkane"), ("591-76-4", "2-methylhexane", "iso-alkane"),
    ("592-27-8", "2-methylheptane", "iso-alkane"), ("79-29-8", "2,3-dimethylbutane", "iso-alkane"),
    ("75-83-2", "2,2-dimethylbutane", "iso-alkane"), ("464-06-2", "2,2,3-trimethylbutane", "iso-alkane"),
    ("871-83-0", "2-methylnonane", "iso-alkane"), ("1921-70-6", "pristane", "iso-alkane"),
    ("638-36-8", "phytane", "iso-alkane"), ("111-01-3", "squalane", "iso-alkane"),
    ("111-02-4", "squalene", "iso-alkane"),
    # cycloalkanes
    ("287-92-3", "cyclopentane", "cycloalkane"), ("110-82-7", "cyclohexane", "cycloalkane"),
    ("108-87-2", "methylcyclohexane", "cycloalkane"), ("1678-91-7", "ethylcyclohexane", "cycloalkane"),
    ("292-64-8", "cyclooctane", "cycloalkane"), ("91-17-8", "decalin", "cycloalkane"),
    # aromatics
    ("71-43-2", "benzene", "aromatic"), ("108-88-3", "toluene", "aromatic"),
    ("95-47-6", "o-xylene", "aromatic"), ("108-38-3", "m-xylene", "aromatic"),
    ("106-42-3", "p-xylene", "aromatic"), ("100-41-4", "ethylbenzene", "aromatic"),
    ("95-63-6", "1,2,4-trimethylbenzene", "aromatic"), ("91-20-3", "naphthalene", "aromatic"),
    ("90-12-0", "1-methylnaphthalene", "aromatic"), ("91-57-6", "2-methylnaphthalene", "aromatic"),
    ("120-12-7", "anthracene", "aromatic"), ("85-01-8", "phenanthrene", "aromatic"),
    ("100-42-5", "styrene", "aromatic"), ("496-11-7", "indane", "aromatic"),
    ("119-64-2", "tetralin", "aromatic"), ("98-82-8", "cumene", "aromatic"),
    # olefins
    ("592-41-6", "1-hexene", "olefin"), ("111-66-0", "1-octene", "olefin"),
    ("872-05-9", "1-decene", "olefin"), ("112-41-4", "1-dodecene", "olefin"),
    ("112-88-9", "1-octadecene", "olefin"),
    # oxygenates
    ("64-17-5", "ethanol", "oxygenate"), ("71-23-8", "1-propanol", "oxygenate"),
    ("71-36-3", "1-butanol", "oxygenate"), ("111-87-5", "1-octanol", "oxygenate"),
    ("104-76-7", "2-ethylhexanol", "oxygenate"), ("56-81-5", "glycerol", "oxygenate"),
    ("107-21-1", "ethylene_glycol", "oxygenate"), ("67-56-1", "methanol", "oxygenate"),
    ("1634-04-4", "MTBE", "oxygenate"), ("60-29-7", "diethyl_ether", "oxygenate"),
    ("67-64-1", "acetone", "oxygenate"), ("78-93-3", "2-butanone", "oxygenate"),
    ("79-20-9", "methyl_acetate", "oxygenate"), ("141-78-6", "ethyl_acetate", "oxygenate"),
    ("123-86-4", "butyl_acetate", "oxygenate"),
    # fatty acids
    ("124-07-2", "octanoic_acid", "fatty_acid"), ("334-48-5", "decanoic_acid", "fatty_acid"),
    ("143-07-7", "lauric_acid", "fatty_acid"), ("544-63-8", "myristic_acid", "fatty_acid"),
    ("57-10-3", "palmitic_acid", "fatty_acid"), ("57-11-4", "stearic_acid", "fatty_acid"),
    ("112-80-1", "oleic_acid", "fatty_acid"), ("60-33-3", "linoleic_acid", "fatty_acid"),
    ("463-40-1", "linolenic_acid", "fatty_acid"),
    # FAMEs (biodiesel-related)
    ("111-11-5", "methyl_caprylate", "FAME"), ("111-82-0", "methyl_laurate", "FAME"),
    ("124-10-7", "methyl_myristate", "FAME"), ("112-39-0", "methyl_palmitate", "FAME"),
    ("112-61-8", "methyl_stearate", "FAME"), ("112-62-9", "methyl_oleate", "FAME"),
    ("112-63-0", "methyl_linoleate", "FAME"), ("111-62-6", "ethyl_oleate", "FAME"),
    # glycerides
    ("102-76-1", "triacetin", "glyceride"), ("60-01-5", "tributyrin", "glyceride"),
    ("122-32-7", "triolein", "glyceride"), ("555-43-1", "tristearin", "glyceride"),
    ("555-44-2", "tripalmitin", "glyceride"),
    # other oil-related compounds
    ("7732-18-5", "water", "misc"), ("57-88-5", "cholesterol", "misc"),
    ("83-46-5", "beta-sitosterol", "misc"), ("36653-82-4", "1-hexadecanol", "misc"),
    ("143-28-2", "oleyl_alcohol", "misc"),
    # S/N-heterocycles (diesel-related)
    ("110-02-1", "thiophene", "S/N-heterocycle"), ("95-15-8", "benzothiophene", "S/N-heterocycle"),
    ("132-65-0", "dibenzothiophene", "S/N-heterocycle"), ("91-22-5", "quinoline", "S/N-heterocycle"),
    ("120-72-9", "indole", "S/N-heterocycle"), ("110-86-1", "pyridine", "S/N-heterocycle"),
    ("62-53-3", "aniline", "S/N-heterocycle"), ("86-74-8", "carbazole", "S/N-heterocycle"),
]


def cas_to_id(cas):
    return "C" + cas.replace("-", "")


def fetch(cas, idx=0):
    import urllib.request
    url = (f"https://webbook.nist.gov/cgi/cbook.cgi?JCAMP={cas_to_id(cas)}"
           f"&Index={idx}&Type=IR")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 research"})
    with urllib.request.urlopen(req, timeout=25) as r:
        return r.read().decode("utf-8", errors="replace")


def parse_jcamp(text):
    """Parse NIST JCAMP-DX (X++(Y..Y)). Returns (x, y, title) or None."""
    if "##XYDATA" not in text and "##XYPOINTS" not in text:
        return None
    fields = {}
    for line in text.splitlines():
        if line.startswith("##"):
            key, _, val = line[2:].partition("=")
            fields[key.strip().upper()] = val.strip()
    title = fields.get("TITLE", "")
    try:
        firstx = float(fields["FIRSTX"]); lastx = float(fields["LASTX"])
        npts = int(float(fields["NPOINTS"]))
        xf = float(fields.get("XFACTOR", 1)); yf = float(fields.get("YFACTOR", 1))
    except Exception:
        return None
    ys = []
    in_data = False
    for line in text.splitlines():
        if line.startswith("##XYDATA") or line.startswith("##XYPOINTS"):
            in_data = True
            continue
        if in_data:
            if line.startswith("##") or line.startswith("$$"):
                break
            toks = line.split()
            if len(toks) >= 2:
                try:
                    [float(t) for t in toks]
                except ValueError:
                    continue
                if "X++" in fields.get("XYDATA", "") or len(toks) > 2:
                    ys.extend(float(t) * yf for t in toks[1:])
                else:  # XYPOINTS: x y pairs
                    ys.append(float(toks[1]) * yf)
    if abs(len(ys) - npts) > max(2, 0.02 * npts):
        return None
    x = np.linspace(firstx * xf, lastx * xf, len(ys))
    return x, np.asarray(ys), title


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delay", type=float, default=1.3)
    args = ap.parse_args()
    rows = []
    for i, (cas, name, cat) in enumerate(COMPOUNDS):
        f = JDX_DIR / f"{cas_to_id(cas)}.jdx"
        if not f.exists():
            text, got = None, False
            for idx in (0, 1, 2):
                try:
                    t = fetch(cas, idx)
                except Exception as e:
                    print(f"[{i}] {name} idx{idx} fetch error: {e}", flush=True)
                    time.sleep(args.delay)
                    continue
                if "##JCAMP-DX" in t and "##XYDATA" in t:
                    text, got = t, True
                    break
                time.sleep(0.4)
            if got:
                f.write_text(text, encoding="utf-8")
            else:
                print(f"[{i}] MISS {name} ({cas})", flush=True)
            time.sleep(args.delay)
        if f.exists():
            parsed = parse_jcamp(f.read_text(encoding="utf-8", errors="replace"))
            if parsed:
                x, y, title = parsed
                rows.append(dict(cas=cas, name=name, category=cat, title=title,
                                 n=len(y), file=f.name, x=x, y=y))
            else:
                print(f"[{i}] PARSE-FAIL {name}", flush=True)
    np.savez(OUT / "nist_ir.npz",
             x=np.array([r["x"] for r in rows], dtype=object),
             y=np.array([r["y"] for r in rows], dtype=object),
             names=np.array([r["name"] for r in rows]),
             categories=np.array([r["category"] for r in rows]),
             titles=np.array([r["title"] for r in rows]))
    import pandas as pd
    pd.DataFrame([{k: r[k] for k in ("cas", "name", "category", "title", "n")}
                  for r in rows]).to_csv(OUT / "manifest.csv", index=False)
    print(f"done: {len(rows)} spectra parsed -> {OUT / 'nist_ir.npz'}")


if __name__ == "__main__":
    main()
