# app.py
# Streamlit LED Cross-Reference Tool (Competitor PN -> Google search -> download PDF -> parse specs
# -> search Seoul Semiconductor PDFs -> score similarity -> show top matches)
#
# Requirements:
#   pip install streamlit requests pypdf
#
# Run:
#   streamlit run app.py
#
# NOTE:
# - This version intentionally does NOT use st.secrets (avoids "No secrets found" crash).
# - Paste GOOGLE_API_KEY and GOOGLE_CSE_ID in the sidebar.

import os
import re
import time
import hashlib
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List

import requests
import streamlit as st
from pypdf import PdfReader


# ----------------------------
# Config
# ----------------------------
USER_AGENT = "Mozilla/5.0"
CACHE_DIR = "./cache"
PDF_DIR = os.path.join(CACHE_DIR, "pdfs")
os.makedirs(PDF_DIR, exist_ok=True)


# ----------------------------
# Models
# ----------------------------
@dataclass
class LEDSpec:
    part: str
    vendor: str
    source_url: str
    size_mm: Optional[Tuple[float, float, float]] = None   # (L,W,H)
    wavelength_nm: Optional[Tuple[Optional[float], Optional[float]]] = None  # (peak, dom)
    cct_k: Optional[float] = None
    flux_lm: Optional[float] = None
    intensity_cd: Optional[float] = None
    radiant_mw: Optional[float] = None
    vf_v: Optional[float] = None
    if_ma: Optional[float] = None
    view_deg: Optional[float] = None
    color_family: Optional[str] = None


# ----------------------------
# Helpers
# ----------------------------
def safe_filename(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s.strip())
    return s[:180] or "file"

def pick_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except Exception:
        return None

def download_file_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()

def pdf_to_text(pdf_path: str, max_pages: int = 6) -> str:
    reader = PdfReader(pdf_path)
    texts = []
    for page in reader.pages[:max_pages]:
        texts.append(page.extract_text() or "")
    return "\n".join(texts)


# ----------------------------
# Google CSE Search
# ----------------------------
def google_search(query: str, api_key: str, cse_id: str, max_results: int = 10) -> List[str]:
    """
    Google Custom Search JSON API.
    - num max = 10 per request
    - start must be <= 91
    """
    urls: List[str] = []
    start = 1

    while len(urls) < max_results and start <= 91:
        num = min(10, max_results - len(urls))

        r = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={"key": api_key, "cx": cse_id, "q": query, "num": num, "start": start},
            headers={"User-Agent": USER_AGENT},
            timeout=30,
        )

        if r.status_code != 200:
            # Provide readable error
            try:
                j = r.json()
            except Exception:
                j = {"raw": r.text[:500]}
            raise RuntimeError(f"Google CSE error {r.status_code}: {j}")

        data = r.json()
        items = data.get("items", [])
        if not items:
            break

        for item in items:
            link = item.get("link")
            if link:
                urls.append(link)

        start += len(items)

    # Deduplicate
    seen = set()
    out = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)

    return out[:max_results]


# ----------------------------
# PDF download (robust)
# ----------------------------
def download_pdf(url: str, out_path: str) -> None:
    """
    Downloads a PDF. Validates it by checking %PDF header.
    Many distributor links return HTML or 403 pages.
    """
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
    r.raise_for_status()
    content = r.content

    # Validate PDF header
    if not content.startswith(b"%PDF"):
        raise ValueError("Downloaded content is not a valid PDF (likely HTML/blocked page).")

    with open(out_path, "wb") as f:
        f.write(content)


# ----------------------------
# Spec extraction (basic heuristics)
# ----------------------------
def extract_specs_from_text(part: str, vendor: str, source_url: str, text: str) -> LEDSpec:
    t = " ".join(text.split())
    spec = LEDSpec(part=part, vendor=vendor, source_url=source_url)

    # Size like "3.2 x 1.6 x 1.1 mm"
    m = re.search(r'(\d+(?:\.\d+)?)\s*[x×\*]\s*(\d+(?:\.\d+)?)\s*(?:[x×\*]\s*(\d+(?:\.\d+)?))?\s*mm', t, re.I)
    if m:
        L = pick_float(m.group(1))
        W = pick_float(m.group(2))
        H = pick_float(m.group(3)) if m.group(3) else 0.0
        if L and W:
            spec.size_mm = (L, W, H if H is not None else 0.0)

    # Dominant / peak wavelength
    m = re.search(r'(dominant wavelength|lambda d|λd)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*nm', t, re.I)
    dom = pick_float(m.group(2)) if m else None
    m = re.search(r'(peak wavelength|lambda p|λp)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*nm', t, re.I)
    peak = pick_float(m.group(2)) if m else None
    if dom is not None or peak is not None:
        spec.wavelength_nm = (peak, dom)

    # CCT
    m = re.search(r'(\b[2-9]\d{3,4})\s*K\b', t)
    if m:
        spec.cct_k = pick_float(m.group(1))

    # Flux
    m = re.search(r'(luminous flux|flux)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*lm', t, re.I)
    if m:
        spec.flux_lm = pick_float(m.group(2))

    # Intensity (cd)
    m = re.search(r'(luminous intensity|intensity)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*cd', t, re.I)
    if m:
        spec.intensity_cd = pick_float(m.group(2))

    # Vf
    m = re.search(r'(forward voltage|vf)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*V\b', t, re.I)
    if m:
        spec.vf_v = pick_float(m.group(2))

    # If
    m = re.search(r'(forward current|if)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*mA\b', t, re.I)
    if m:
        spec.if_ma = pick_float(m.group(2))

    # Viewing angle
    m = re.search(r'(view angle|viewing angle)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*deg', t, re.I)
    if m:
        spec.view_deg = pick_float(m.group(2))

    # Color family inference
    if spec.cct_k:
        spec.color_family = "white"
    elif spec.wavelength_nm:
        w = spec.wavelength_nm[1] or spec.wavelength_nm[0]
        if w:
            if w < 420:
                spec.color_family = "uv/violet"
            elif w < 500:
                spec.color_family = "blue"
            elif w < 570:
                spec.color_family = "green"
            elif w < 700:
                spec.color_family = "red"
            else:
                spec.color_family = "ir"

    return spec


# ----------------------------
# Similarity score
# ----------------------------
def rel_diff(a: Optional[float], b: Optional[float], scale: float = 1.0) -> float:
    if a is None or b is None:
        return 1.0
    denom = max(abs(a), abs(b), scale)
    return abs(a - b) / denom

def score_match(target: LEDSpec, cand: LEDSpec) -> float:
    w_size, w_color, w_wave, w_flux, w_vf, w_if, w_view = 2.5, 3.0, 2.0, 2.0, 1.5, 1.0, 1.0
    score = 0.0
    wsum = 0.0

    # Size
    if target.size_mm and cand.size_mm:
        dl = rel_diff(target.size_mm[0], cand.size_mm[0], 1.0)
        dw = rel_diff(target.size_mm[1], cand.size_mm[1], 1.0)
        dh = rel_diff(target.size_mm[2], cand.size_mm[2], 1.0)
        d = (dl + dw + dh) / 3.0
        score += w_size * (1.0 - min(d, 1.0))
    wsum += w_size

    # Color family
    if target.color_family and cand.color_family:
        score += w_color * (1.0 if target.color_family == cand.color_family else 0.0)
    wsum += w_color

    # Wavelength
    tw = (target.wavelength_nm[1] or target.wavelength_nm[0]) if target.wavelength_nm else None
    cw = (cand.wavelength_nm[1] or cand.wavelength_nm[0]) if cand.wavelength_nm else None
    if tw and cw:
        d = rel_diff(tw, cw, 50.0)
        score += w_wave * (1.0 - min(d, 1.0))
    wsum += w_wave

    # Flux
    d = rel_diff(target.flux_lm, cand.flux_lm, 10.0)
    score += w_flux * (1.0 - min(d, 1.0))
    wsum += w_flux

    # Vf/If/View
    for wt, ta, ca, sc in [
        (w_vf, target.vf_v, cand.vf_v, 1.0),
        (w_if, target.if_ma, cand.if_ma, 20.0),
        (w_view, target.view_deg, cand.view_deg, 30.0),
    ]:
        d = rel_diff(ta, ca, sc)
        score += wt * (1.0 - min(d, 1.0))
        wsum += wt

    return 100.0 * (score / max(wsum, 1e-9))


# ----------------------------
# SSC candidate search
# ----------------------------
def find_seoulsemi_candidates(target: LEDSpec, api_key: str, cse_id: str, limit: int = 12) -> List[str]:
    terms = []
    if target.color_family:
        terms.append(target.color_family)
    if target.size_mm:
        terms.append(f"{target.size_mm[0]}x{target.size_mm[1]}mm")
    if target.cct_k:
        terms.append(f"{int(target.cct_k)}K")
    hint = " ".join(terms).strip()

    q = f"site:seoulsemicon.com filetype:pdf datasheet {hint}".strip()
    links = google_search(q, api_key, cse_id, max_results=40)
    pdfs = [l for l in links if ".pdf" in l.lower()]
    return pdfs[:limit]


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="LED Cross-Reference", layout="wide")
st.title("LED Cross-Reference Tool (Competitor → Seoul Semiconductor)")

with st.sidebar:
    st.header("Google Programmable Search (CSE)")
    st.caption("Paste your Google API Key and CSE ID here (no secrets file needed).")

    api_key = st.text_input("GOOGLE_API_KEY", value=os.environ.get("GOOGLE_API_KEY", ""), type="password")
    cse_id = st.text_input("GOOGLE_CSE_ID (cx)", value=os.environ.get("GOOGLE_CSE_ID", ""))

    api_ok = bool(api_key.strip()) and bool(cse_id.strip())
    if api_ok:
        st.success("Credentials provided.")
    else:
        st.warning("Enter both GOOGLE_API_KEY and GOOGLE_CSE_ID to enable the app.")

    max_seoul = st.slider("SSC candidate PDFs to analyze", 3, 25, 10, 1)
    max_pages = st.slider("PDF pages to parse", 1, 20, 6, 1)


with st.form("run_form"):
    competitor_part = st.text_input("Competitor part number", placeholder="e.g., LTST-C190KGKT / LUW HWQP / ...")
    vendor_hint = st.text_input("Vendor hint (optional)", placeholder="e.g., OSRAM / Nichia / Lumileds / LiteOn")
    submitted = st.form_submit_button("Find datasheet + Match to SeoulSemi", disabled=not api_ok)

if submitted:
    part = competitor_part.strip()
    vendor = vendor_hint.strip()

    if not part:
        st.error("Please enter a competitor part number.")
        st.stop()

    try:
        # 1) Competitor datasheet
        with st.status("Searching competitor datasheet PDF…", expanded=True) as status:
            query = f'"{part}" datasheet filetype:pdf'
            if vendor:
                query += f' "{vendor}"'

            links = google_search(query, api_key.strip(), cse_id.strip(), max_results=12)
            pdf_links = [l for l in links if ".pdf" in l.lower()]

            if not pdf_links:
                status.update(label="No PDF datasheet found.", state="error")
                st.error("Could not find a PDF datasheet. Try adding vendor hint or verifying PN suffix.")
                st.stop()

            competitor_pdf_url = pdf_links[0]
            h = hashlib.sha256(competitor_pdf_url.encode("utf-8")).hexdigest()[:10]
            competitor_pdf_path = os.path.join(PDF_DIR, safe_filename(f"{part}_{h}.pdf"))

            if not os.path.exists(competitor_pdf_path):
                download_pdf(competitor_pdf_url, competitor_pdf_path)

            status.update(label="Competitor datasheet downloaded.", state="complete")
            st.write("Competitor datasheet URL:", competitor_pdf_url)

        st.download_button(
            "Download competitor datasheet (cached PDF)",
            data=download_file_bytes(competitor_pdf_path),
            file_name=os.path.basename(competitor_pdf_path),
            mime="application/pdf",
        )

        # 2) Parse competitor
        with st.status("Extracting competitor specs…", expanded=False) as status:
            text = pdf_to_text(competitor_pdf_path, max_pages=max_pages)
            target = extract_specs_from_text(part, vendor or "Unknown", competitor_pdf_url, text)
            status.update(label="Target specs extracted.", state="complete")

        st.subheader("Extracted target specs")
        st.json(asdict(target))

        # 3) Find SSC candidates
        with st.status("Searching SSC datasheet PDFs…", expanded=True) as status:
            ssc_links = find_seoulsemi_candidates(target, api_key.strip(), cse_id.strip(), limit=max_seoul)
            if not ssc_links:
                status.update(label="No SSC candidates found.", state="error")
                st.error("No SSC PDF candidates found via search. Try a different PN or increase candidate count.")
                st.stop()
            status.update(label=f"Found {len(ssc_links)} SSC PDF links.", state="complete")

        # 4) Download/parse/score SSC
        results = []
        with st.status("Scoring SSC candidates…", expanded=True) as status:
            for i, link in enumerate(ssc_links, 1):
                try:
                    name = safe_filename(link.split("/")[-1].split("?")[0])
                    spath = os.path.join(PDF_DIR, f"SSC_{name}")

                    if not spath.lower().endswith(".pdf"):
                        spath += ".pdf"

                    if not os.path.exists(spath):
                        download_pdf(link, spath)

                    stext = pdf_to_text(spath, max_pages=max_pages)
                    cand = extract_specs_from_text(name.replace(".pdf", ""), "Seoul Semiconductor", link, stext)
                    s = score_match(target, cand)
                    results.append((s, cand, spath))

                    st.write(f"[{i}/{len(ssc_links)}] {cand.part} → score {s:.1f}")
                except Exception:
                    continue

            if not results:
                status.update(label="No SSC PDFs could be parsed.", state="error")
                st.error("All SSC PDFs failed parsing (often due to blocked/non-PDF links). Try again or increase pages.")
                st.stop()

            results.sort(key=lambda x: x[0], reverse=True)
            status.update(label="Matching complete.", state="complete")

        st.subheader("Top Seoul Semiconductor matches")
        top_n = min(5, len(results))
        for rank, (s, cand, spath) in enumerate(results[:top_n], 1):
            with st.expander(f"#{rank} — Score {s:.1f} — {cand.part}", expanded=(rank == 1)):
                st.write("SSC datasheet URL:", cand.source_url)
                st.json({k: v for k, v in asdict(cand).items() if v not in (None, "", (), [])})
                st.download_button(
                    f"Download SSC datasheet — {cand.part}",
                    data=download_file_bytes(spath),
                    file_name=os.path.basename(spath),
                    mime="application/pdf",
                    key=f"dl_{rank}_{hashlib.md5(spath.encode()).hexdigest()}",
                )

        st.success("Done.")

    except Exception as e:
        st.error("Run failed. Details below:")
        st.exception(e)

st.divider()
st.caption(
    "This is an MVP using heuristic PDF parsing. Accuracy improves significantly by indexing SSC PDFs into a local DB "
    "and using table-aware extraction for optical/electrical characteristic tables."
)
