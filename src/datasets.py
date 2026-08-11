"""Unified dataset loaders 
All loaders return a dict:
  X: (n_samples, n_wl) spectral matrix
  wavelengths: (n_wl,)
  targets: {property_name: (n_samples,) array, may contain NaN}
  ids: list of sample IDs
  meta: dataset information
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


def _read_eigenvector_csv(path):
    """Parse an Eigenvector CSV (leading rows are metadata)."""
    rows = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            rows.append(line.rstrip("\n").split(","))
    return rows


def _clean_id(s):
    return s.strip().strip('"').strip()


def load_diesel():
    """SWRI diesel NIR: 6+2 property labels."""
    spec_rows = _read_eigenvector_csv(DATA_ROOT / "SWRI_Diesel_NIR" / "diesel_spec.csv")
    prop_rows = _read_eigenvector_csv(DATA_ROOT / "SWRI_Diesel_NIR" / "diesel_prop.csv")

    # Find the Axisscale row
    axis = None
    data_start = None
    for i, r in enumerate(spec_rows):
        if r and _clean_id(r[0]) == "Axisscale":
            axis = np.array([float(x) for x in r[2:] if x.strip()])
            data_start = i + 1
            break
    n_wl = len(axis)
    ids, X = [], []
    for r in spec_rows[data_start:]:
        vals = [x for x in r[2:2 + n_wl]]
        if len(vals) < n_wl or not _clean_id(r[1]):
            continue
        try:
            X.append([float(x) if x.strip() else np.nan for x in vals])
            ids.append(_clean_id(r[1]))
        except ValueError:
            continue
    X = np.array(X, dtype=float)

    # Property table
    header_i = None
    for i, r in enumerate(prop_rows):
        if _clean_id(r[0]) == "Label" and any("CN" == _clean_id(c) for c in r):
            header_i = i
            break
    header = [_clean_id(c) for c in prop_rows[header_i]]
    prop_names = [h for h in header[2:] if h]
    pids, P = [], []
    for r in prop_rows[header_i + 1:]:
        if len(r) < 3 or not _clean_id(r[1]):
            continue
        pids.append(_clean_id(r[1]))
        row = []
        for x in r[2:2 + len(prop_names)]:
            x = x.strip()
            try:
                row.append(float(x))
            except ValueError:
                row.append(np.nan)
        P.append(row)
    P = np.array(P, dtype=float)
    # Align by spectrum ids
    pos = {pid: j for j, pid in enumerate(pids)}
    targets = {}
    for k, name in enumerate(prop_names):
        t = np.full(len(ids), np.nan)
        for i, sid in enumerate(ids):
            if sid in pos:
                t[i] = P[pos[sid], k]
        targets[name] = t
    return dict(X=X, wavelengths=axis, targets=targets, ids=ids,
                meta=dict(name="SWRI_Diesel_NIR", source="eigenvector.com",
                          desc="Diesel fuel NIR, SWRI"))


def load_gasoline():
    """Kalivas gasoline NIR + octane number (extracted from the R pls package)."""
    z = np.load(DATA_ROOT / "processed" / "gasoline_arrays.npz")
    octane = z["arr_0"].astype(float)
    nir = z["arr_1"].astype(float)
    if nir.shape[0] != 60:
        nir = nir.reshape(60, 401, order="F")
    wl = 900 + 2 * np.arange(nir.shape[1])
    return dict(X=nir, wavelengths=wl.astype(float), targets=dict(octane=octane),
                ids=[f"gas{i+1}" for i in range(len(octane))],
                meta=dict(name="Kalivas_Gasoline", source="R pls package",
                          desc="Gasoline NIR 900-1700nm, octane number"))


def load_mayonnaise():
    """Mayonnaise NIR + 6-class oil-type labels (extracted from the R pls package;
    no quantitative labels, treated as a qualitative dataset)."""
    z = np.load(DATA_ROOT / "processed" / "mayonnaise_arrays.npz")
    nir = z["arr_0"].astype(float)
    oil_type = z["arr_1"].astype(int)  # oil types 1-6
    wl = 1100 + 4 * np.arange(nir.shape[1], dtype=float)
    return dict(X=nir, wavelengths=wl, labels=oil_type, targets={},
                ids=[f"mayo{i+1}" for i in range(nir.shape[0])],
                meta=dict(name="Mayonnaise", source="R pls package",
                          desc="Mayonnaise NIR, 6-class oil type (classification)"))


def _dataset_struct_data(struct):
    """Extract data and axisscale from an Eigenvector DataSet struct (mat_struct)."""
    s = struct[0, 0] if isinstance(struct, np.ndarray) else struct
    data = np.asarray(getattr(s, "data"), dtype=float)
    axis = np.array([], dtype=float)
    try:
        raw = np.asarray(getattr(s, "axisscale", []), dtype=object).ravel()
        for elem in raw:
            e = np.asarray(elem, dtype=float).ravel() if not np.isscalar(elem) else np.array([float(elem)])
            if e.size == data.shape[1]:
                axis = e
                break
    except (ValueError, TypeError):
        pass
    if axis.size != data.shape[1]:
        axis = np.arange(data.shape[1], dtype=float)  # placeholder; callers may override
    return data, axis


def load_corn():
    """Cargill corn three-instrument NIR: moisture/oil/protein/starch."""
    import scipy.io
    m = scipy.io.loadmat(DATA_ROOT / "Corn_Cargill" / "corn.mat",
                         struct_as_record=False, squeeze_me=False)
    out = {}
    for inst in ["m5", "mp5", "mp6"]:
        key = f"{inst}spec"
        X, axis = _dataset_struct_data(m[key])
        if axis.size == X.shape[1] and axis[0] < 100:  # placeholder axis; override with the literature-known range
            axis = 1100 + 2 * np.arange(X.shape[1], dtype=float)
        out[inst] = (X, axis)
    P, _ = _dataset_struct_data(m["propvals"])
    names = ["moisture", "oil", "protein", "starch"]
    targets = {names[i]: P[:, i].astype(float) for i in range(min(4, P.shape[1]))}
    return dict(instruments=out, targets=targets,
                ids=[f"corn{i+1}" for i in range(P.shape[0])],
                meta=dict(name="Cargill_Corn_3inst", source="eigenvector.com",
                          desc="Corn NIR on 3 instruments, moisture/oil/protein/starch"))


def load_evoo():
    """EVOO adulteration NIR-HSI mean spectra (open-source on GitHub)."""
    df = pd.read_excel(DATA_ROOT / "EVOO_NIR_HSI" / "data" / "Raw_A.xlsx")
    meta_cols = [c for c in df.columns if not str(c).replace(".", "").isdigit()]
    spec_cols = [c for c in df.columns if str(c).replace(".", "").isdigit()]
    X = df[spec_cols].to_numpy(dtype=float)
    wl = np.array([float(c) for c in spec_cols])
    targets = dict(
        adulteration_level=pd.to_numeric(df["adulteration_level"], errors="coerce").to_numpy(dtype=float),
    )
    labels = df["class_1"].astype(str).to_numpy()
    return dict(X=X, wavelengths=wl, targets=targets, labels=labels,
                ids=df["sample_id"].astype(str).tolist(),
                meta=dict(name="EVOO_NIR_HSI", source="github.com/DNMalavi",
                          desc="EVOO adulteration NIR-HSI mean spectra"))


def load_selfmix():
    """Self-collected FT-NIR n-pentane/CCl4 binary-mixture quantification task
    (collected in-house on an ABB MB3600).

    See data/pentane_ccl4/processed/README.md for the cleaned data. The label
    phi_pentane is the n-pentane volume fraction (preparation ratio, parts/250).
    Replicate structure: spectra sharing the same bottle sample_id are replicate
    scans of the same physical sample (3 reps x 3 temperatures); groups are
    returned for group-aware leakage-safe splitting (same as EVOO). X is raw
    absorbance with columns sorted by ascending wavelength_nm; the ML pipeline
    then interpolates each spectrum to 512 points and applies SNV.
    """
    import json
    z = np.load(DATA_ROOT / "pentane_ccl4" / "processed" / "pentane_ccl4_FTIR.npz",
                allow_pickle=True)
    return dict(
        X=z["X"].astype(float),
        wavelengths=z["wavelength_nm"].astype(float),
        targets=dict(phi_pentane=z["y"].astype(float)),
        ids=[str(s) for s in z["name"]],
        groups=[str(s) for s in z["sample_id"]],
        rep=z["rep"].astype(int), temp_c=z["temp_c"].astype(int),
        kind=[str(s) for s in z["kind"]],
        meta=json.loads(str(z["meta"])) if "meta" in z else
        dict(name="pentane_ccl4_FTIR_selfcollected"),
    )


def load_edibleoil():
    """Edible oil FT-NIR peroxide value quantification (Lavine/Booksh group,
    Mendeley DOI 10.17632/ctgg7k4m5g.2).

    NIR24mm1A: 300 spectra = 100 edible oils x 3 replicate scans; the FT-NIR data
    are stored in **wavenumber** 3799-14999 cm^-1 (= 667-2632 nm) — covering all
    C-H overtone/combination bands (1100-1250/1350-1550/1650-1800/2100-2400 nm),
    i.e., the strongest domain for the physics prior (better than diesel). The
    target is peroxide value PV (meq O2/kg, 1.52-165). After the
    **wavenumber-to-nm conversion** (nm = 1e7/cm^-1), wavelengths are stored in
    nm so that the nm-based band_mask correctly hits the C-H bands.
    groups = oil identity (class, PV) (100 oils x 3 replicates; (class, PV) is
    unique) for group-aware leakage-safe splitting. X is absorbance with columns
    aligned in ascending nm order; the ML pipeline then interpolates each
    spectrum to 512 points and applies SNV.
    **npz cache**: the wide table (11618 columns) is prone to OOM with
    pd.read_csv in fresh processes; it is parsed once and cached as an npz, and
    only the small npz is read afterwards.
    """
    cache = DATA_ROOT / "EdibleOil_NIR_MIR" / "edibleoil_nir24mm.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return dict(X=z["X"].astype(float), wavelengths=z["wl"].astype(float),
                    targets=dict(peroxide_value=z["y"].astype(float)),
                    ids=[str(s) for s in z["ids"]], groups=[str(s) for s in z["groups"]],
                    meta=dict(name="EdibleOil_NIR_LavineBooksh",
                              source="Mendeley 10.17632/ctgg7k4m5g.2",
                              desc="Edible oil FT-NIR 667-2632nm, peroxide value, 100 oils x3 reps"))
    p = DATA_ROOT / "EdibleOil_NIR_MIR" / "NIR24mm1A.csv"
    raw = pd.read_csv(p, header=None, low_memory=False)
    wavenum = np.array([float(x) for x in raw.iloc[0, 2:]], dtype=float)  # cm⁻¹
    nm = 1e7 / wavenum                                                    # → nm
    body = raw.iloc[1:].reset_index(drop=True)
    cls = pd.to_numeric(body.iloc[:, 0], errors="coerce").to_numpy()
    pv = pd.to_numeric(body.iloc[:, 1], errors="coerce").to_numpy(dtype=float)
    spec = body.iloc[:, 2:].to_numpy(dtype=float)
    order = np.argsort(nm)            # align spectral columns in ascending nm order
    nm, spec = nm[order], spec[:, order]
    # Drop invalid rows: missing PV / non-numeric class / **rows with all-NaN
    # placeholder spectra** (the dataset uses all-NaN rows to pad the matrix;
    # 2 such rows; if not dropped, they would contaminate all methods via SNV).
    # After filtering: 298 clean spectra / 100 oils.
    m = ~np.isnan(pv) & np.isfinite(cls) & ~np.isnan(spec).any(axis=1)
    spec, pv, cls = spec[m], pv[m], cls[m]
    oil_id = [f"c{int(c)}_pv{round(float(v), 3)}" for c, v in zip(cls, pv)]  # 100 oils x 3 replicates
    ids = [f"oil{int(c)}_{i}" for i, c in enumerate(cls)]
    np.savez(cache, X=spec.astype(np.float32), wl=nm.astype(np.float32),
             y=pv.astype(np.float32), ids=np.array(ids), groups=np.array(oil_id))
    return dict(X=spec.astype(float), wavelengths=nm.astype(float),
                targets=dict(peroxide_value=pv.astype(float)),
                ids=ids, groups=oil_id,
                meta=dict(name="EdibleOil_NIR_LavineBooksh",
                          source="Mendeley 10.17632/ctgg7k4m5g.2",
                          desc="Edible oil FT-NIR 667-2632nm, peroxide value, 100 oils x3 reps"))


LOADERS = dict(diesel=load_diesel, gasoline=load_gasoline,
               mayonnaise=load_mayonnaise, corn=load_corn, evoo=load_evoo,
               selfmix=load_selfmix, edibleoil=load_edibleoil)

if __name__ == "__main__":
    d = load_diesel()
    print("diesel:", d["X"].shape, d["wavelengths"][:3], d["wavelengths"][-3:],
          {k: int(np.sum(~np.isnan(v))) for k, v in d["targets"].items()})
    g = load_gasoline()
    print("gasoline:", g["X"].shape, g["targets"]["octane"][:3])
    m = load_mayonnaise()
    print("mayonnaise:", m["X"].shape, "oil types:", sorted(set(m["labels"].tolist())))
    c = load_corn()
    print("corn:", {k: v[0].shape for k, v in c["instruments"].items()},
          {k: v[:2] for k, v in c["targets"].items()})
    e = load_evoo()
    print("evoo:", e["X"].shape, "adulteration:", np.nanmin(e["targets"]["adulteration_level"]),
          "-", np.nanmax(e["targets"]["adulteration_level"]), "classes:", len(set(e["labels"])))
