"""
Core computation engine for restriction enzyme digest analysis.

All pure logic lives here — no I/O, no plotting, no Streamlit dependency.
Functions are grouped into:
  - FASTA / DNA helpers
  - Enzyme loading & filtering
  - Genome scanning & cut indexing
  - Fragment analysis & insertion simulation
  - Transgene (T-DNA) evaluation
  - Summary collation
"""

from __future__ import annotations

import bisect
import gzip
import io
import re
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import (
    BinaryIO,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

# ---------------------------------------------------------------------------
# IUPAC alphabet → regex mapping
# ---------------------------------------------------------------------------
IUPAC_TO_REGEX = {
    "A": "A", "C": "C", "G": "G", "T": "T",
    "R": "[AG]", "Y": "[CT]", "S": "[GC]", "W": "[AT]",
    "K": "[GT]", "M": "[AC]", "B": "[CGT]", "D": "[AGT]",
    "H": "[ACT]", "V": "[ACG]", "N": "[ACGT]",
}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EnzymeMetadata:
    enzyme: str
    site: str
    size: int
    palindromic: bool
    cut_type: str
    overhang_length: Optional[int]
    ovhgseq: str
    substrat: str
    freq: str
    suppl: str
    opt_temp: str
    inact_temp: str
    fst5: Optional[int]
    fst3: Optional[int]
    scd5: Optional[int]
    scd3: Optional[int]
    charac: str
    uri: str
    enzyme_id: str


@dataclass
class DigestMetrics:
    genome_site_count: int
    sites_per_mb: float
    sites_per_chromosome: Dict[str, int]
    n_fragments: int
    n_useful_fragments: int
    pct_useful_fragments: float
    median_fragment_bp: float
    p25_fragment_bp: float
    p75_fragment_bp: float
    n_insertions: int
    n_usable_insertions: int
    pct_usable_insertions: float
    insertion_median_bp: float
    insertion_p25_bp: float
    insertion_p75_bp: float


@dataclass
class TransgeneMetrics:
    tdna_length_bp: int
    tdna_cut_count: int
    tdna_cut_positions: List[int]
    left_border_zone_cut_count: int
    right_border_zone_cut_count: int
    left_border_ok: bool
    right_border_ok: bool
    tdna_no_cut: bool
    selected_border_ok: bool
    ranking_bucket: str

# ---------------------------------------------------------------------------
# DNA helpers
# ---------------------------------------------------------------------------

def clean_dna_sequence(seq: str) -> str:
    return "".join(b for b in seq.upper() if b in "ACGT")


def reverse_complement(seq: str) -> str:
    return seq.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def is_unambiguous_site(site: str) -> bool:
    return bool(site) and all(b in "ACGT" for b in site.upper())

# ---------------------------------------------------------------------------
# FASTA reading  (accepts a file path *or* an in-memory binary stream)
# ---------------------------------------------------------------------------

def read_fasta(source: Union[str, Path, bytes, BinaryIO]) -> Dict[str, str]:
    """Return ``{contig_name: uppercase_sequence}``.

    *source* may be:
      - a ``pathlib.Path`` or string path (handles ``.gz`` transparently)
      - raw ``bytes`` (gzip auto-detected via the magic number ``0x1f 0x8b``)
      - an open file-like object (binary or text)

    The function normalises the input to a single text iterator before
    parsing, so the gzip / decode logic only lives in one place.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        fh = gzip.open(path, "rt") if path.suffix == ".gz" else path.open("r")
    elif isinstance(source, bytes):
        if source[:2] == b"\x1f\x8b":
            text = gzip.decompress(source).decode()
        else:
            text = source.decode()
        fh = io.StringIO(text)
    else:
        raw = source.read()
        if isinstance(raw, bytes):
            if raw[:2] == b"\x1f\x8b":
                text = gzip.decompress(raw).decode()
            else:
                text = raw.decode()
            fh = io.StringIO(text)
        else:
            fh = io.StringIO(raw)

    genome: Dict[str, List[str]] = {}
    current: Optional[str] = None
    with fh as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                current = line[1:].split()[0]
                if current in genome:
                    raise ValueError(f"Duplicate contig name: {current}")
                genome[current] = []
            else:
                if current is None:
                    raise ValueError("FASTA format error: sequence before header")
                genome[current].append(line.upper())

    return {k: "".join(v) for k, v in genome.items()}

# ---------------------------------------------------------------------------
# Enzyme loading & filtering
# ---------------------------------------------------------------------------

def _safe_int(v: object) -> Optional[int]:
    """Return ``int(v)`` for any int / float / numeric-string input.

    Biopython occasionally stores fields like ``fst5`` as floats or numeric
    strings, so a strict ``isinstance(v, int)`` check loses real data. We
    coerce when possible and return ``None`` only when the value really
    cannot be turned into an integer.
    """
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v) if v == v else None
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            return int(float(s))
        except ValueError:
            return None
    return None


def _classify_cut(ovhg: Optional[int]) -> str:
    if ovhg is None:
        return "unknown"
    return "blunt" if ovhg == 0 else "sticky"


def load_biopython_enzymes() -> List[EnzymeMetadata]:
    from Bio.Restriction import Restriction_Dictionary

    enzymes: List[EnzymeMetadata] = []
    for name, info in Restriction_Dictionary.rest_dict.items():
        site = str(info.get("site") or "").upper()
        size = int(info.get("size") or len(site) or 0)
        ovhg = _safe_int(info.get("ovhg"))
        palindromic = (
            bool(site)
            and is_unambiguous_site(site)
            and site == reverse_complement(site)
        )
        enzymes.append(
            EnzymeMetadata(
                enzyme=str(name), site=site, size=size,
                palindromic=palindromic, cut_type=_classify_cut(ovhg),
                overhang_length=abs(ovhg) if ovhg is not None else None,
                ovhgseq=str(info.get("ovhgseq") or ""),
                substrat=str(info.get("substrat") or ""),
                freq=str(info.get("freq") or ""),
                suppl=str(info.get("suppl") or ""),
                opt_temp=str(info.get("opt_temp") or ""),
                inact_temp=str(info.get("inact_temp") or ""),
                fst5=_safe_int(info.get("fst5")),
                fst3=_safe_int(info.get("fst3")),
                scd5=_safe_int(info.get("scd5")),
                scd3=_safe_int(info.get("scd3")),
                charac=str(info.get("charac") or ""),
                uri=str(info.get("uri") or ""),
                enzyme_id=str(info.get("id") or ""),
            )
        )
    enzymes.sort(key=lambda x: x.enzyme)
    return enzymes


def filter_practical_enzymes(
    all_enzymes: List[EnzymeMetadata],
    allowed_sizes: Sequence[int] = (4, 5, 6),
    require_palindromic: bool = True,
    require_unambiguous: bool = True,
    exclude_unusual: bool = True,
    deduplicate_sites: bool = True,
) -> Tuple[List[EnzymeMetadata], Dict[str, int]]:
    kept: List[EnzymeMetadata] = []
    reasons: Dict[str, int] = defaultdict(int)

    for m in all_enzymes:
        if not m.site:
            reasons["missing_site"] += 1
        elif m.size not in allowed_sizes:
            reasons["wrong_size"] += 1
        elif require_palindromic and not m.palindromic:
            reasons["not_palindromic"] += 1
        elif require_unambiguous and not is_unambiguous_site(m.site):
            reasons["ambiguous_site"] += 1
        elif exclude_unusual and (m.cut_type == "unknown" or m.size <= 0):
            reasons["unusual_cut_behavior"] += 1
        else:
            kept.append(m)

    kept.sort(key=lambda x: x.enzyme)

    if deduplicate_sites:
        seen_sites: dict[str, str] = {}
        deduped: List[EnzymeMetadata] = []
        n_duplicates = 0
        for e in kept:
            if e.site in seen_sites:
                n_duplicates += 1
            else:
                seen_sites[e.site] = e.enzyme
                deduped.append(e)
        if n_duplicates:
            reasons["duplicate_site"] = n_duplicates
        kept = deduped

    return kept, dict(reasons)

# ---------------------------------------------------------------------------
# Motif scanning & cut-site indexing
# ---------------------------------------------------------------------------

def motif_to_regex(motif: str) -> re.Pattern[str]:
    regex = "".join(IUPAC_TO_REGEX[b] for b in motif.upper())
    return re.compile(rf"(?={regex})")


def find_cut_positions(seq: str, motif: str) -> List[int]:
    return [m.start() for m in motif_to_regex(motif).finditer(seq.upper())]


def build_motif_cut_index(
    genome: Dict[str, str],
    motifs: Iterable[str],
    progress_callback=None,
) -> Dict[str, Dict[str, List[int]]]:
    """Build {motif: {contig: [positions]}} for every unique motif.

    *progress_callback(current, total)* is called after each motif so that
    a Streamlit progress bar (or any other UI) can be updated.
    """
    unique = sorted(set(motifs))
    total = len(unique)
    index: Dict[str, Dict[str, List[int]]] = {}

    for i, motif in enumerate(unique, 1):
        pat = motif_to_regex(motif)
        index[motif] = {
            contig: [m.start() for m in pat.finditer(seq)]
            for contig, seq in genome.items()
        }
        if progress_callback:
            progress_callback(i, total)

    return index

# ---------------------------------------------------------------------------
# Fragment analysis
# ---------------------------------------------------------------------------

def compute_fragment_sizes(seq_len: int, cuts: List[int]) -> List[int]:
    bounds = [0] + cuts + [seq_len]
    return [bounds[i] - bounds[i - 1] for i in range(1, len(bounds)) if bounds[i] - bounds[i - 1] > 0]


def _percentile(vals: List[int], p: float) -> float:
    if not vals:
        return 0.0
    if p <= 0:
        return float(vals[0])
    if p >= 100:
        return float(vals[-1])
    rank = (p / 100.0) * (len(vals) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(vals) - 1)
    frac = rank - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac

# ---------------------------------------------------------------------------
# Insertion simulation
# ---------------------------------------------------------------------------

def sample_random_insertions(
    contig_lengths: Dict[str, int],
    n_insertions: int,
    insert_len_bp: int = 100,
    seed: int = 7,
) -> List[Tuple[str, int]]:
    rng = random.Random(seed)
    contigs = list(contig_lengths.keys())
    lengths = [contig_lengths[c] for c in contigs]
    total = sum(lengths)
    if total == 0:
        raise ValueError("Genome length is zero")

    half = insert_len_bp // 2
    insertions: List[Tuple[str, int]] = []
    for _ in range(n_insertions):
        r = rng.uniform(0, total)
        cum = 0
        chosen = contigs[-1]
        for c, L in zip(contigs, lengths):
            cum += L
            if r <= cum:
                chosen = c
                break
        L = contig_lengths[chosen]
        pos = L // 2 if L <= insert_len_bp + 2 else rng.randint(half, L - half - 1)
        insertions.append((chosen, pos))
    return insertions


def _flanking_fragment(contig_len: int, cuts: List[int], pos: int) -> int:
    idx = bisect.bisect_right(cuts, pos)
    up = cuts[idx - 1] if idx > 0 else 0
    down = cuts[idx] if idx < len(cuts) else contig_len
    return max(down - up, 0)

# ---------------------------------------------------------------------------
# Per-motif digest + insertion analysis
# ---------------------------------------------------------------------------

def analyze_motif(
    motif: str,
    cut_index: Dict[str, Dict[str, List[int]]],
    contig_lengths: Dict[str, int],
    insertions: List[Tuple[str, int]],
    useful_min: int,
    useful_max: int,
) -> Tuple[DigestMetrics, List[int]]:
    cuts_by_contig = cut_index[motif]

    chr_counts = {c: len(cuts_by_contig.get(c, [])) for c in contig_lengths}
    total_sites = sum(chr_counts.values())
    total_bp = sum(contig_lengths.values())
    sites_per_mb = total_sites / (total_bp / 1_000_000) if total_bp else 0.0

    all_frags: List[int] = []
    for chrom in contig_lengths:
        all_frags.extend(
            compute_fragment_sizes(contig_lengths[chrom], cuts_by_contig.get(chrom, []))
        )
    all_frags.sort()

    n_useful = sum(1 for x in all_frags if useful_min <= x <= useful_max)
    pct_useful = 100.0 * n_useful / len(all_frags) if all_frags else 0.0

    ins_sizes = [
        _flanking_fragment(contig_lengths[c], cuts_by_contig.get(c, []), p)
        for c, p in insertions
    ]
    ins_sorted = sorted(ins_sizes)
    n_usable = sum(1 for x in ins_sorted if useful_min <= x <= useful_max)
    pct_usable = 100.0 * n_usable / len(ins_sorted) if ins_sorted else 0.0

    return DigestMetrics(
        genome_site_count=total_sites,
        sites_per_mb=sites_per_mb,
        sites_per_chromosome=chr_counts,
        n_fragments=len(all_frags),
        n_useful_fragments=n_useful,
        pct_useful_fragments=pct_useful,
        median_fragment_bp=_percentile(all_frags, 50),
        p25_fragment_bp=_percentile(all_frags, 25),
        p75_fragment_bp=_percentile(all_frags, 75),
        n_insertions=len(ins_sorted),
        n_usable_insertions=n_usable,
        pct_usable_insertions=pct_usable,
        insertion_median_bp=_percentile(ins_sorted, 50),
        insertion_p25_bp=_percentile(ins_sorted, 25),
        insertion_p75_bp=_percentile(ins_sorted, 75),
    ), ins_sizes

# ---------------------------------------------------------------------------
# Transgene (T-DNA) evaluation
# ---------------------------------------------------------------------------

def evaluate_transgene(
    tdna_seq: str,
    motif: str,
    border: str = "right",
    protected_bp: int = 300,
) -> TransgeneMetrics:
    tdna_len = len(tdna_seq)
    cuts = find_cut_positions(tdna_seq, motif)
    mlen = len(motif)

    left_cuts = [p for p in cuts if (p + mlen) > 0 and p < protected_bp]
    right_start = max(0, tdna_len - protected_bp)
    right_cuts = [p for p in cuts if (p + mlen) > right_start]

    left_ok = len(left_cuts) == 0
    right_ok = len(right_cuts) == 0
    no_cut = len(cuts) == 0
    sel_ok = left_ok if border == "left" else right_ok

    if no_cut:
        bucket = "ideal_no_tdna_cut"
    elif sel_ok:
        bucket = "usable_selected_border_safe"
    else:
        bucket = "poor_selected_border_cut"

    return TransgeneMetrics(
        tdna_length_bp=tdna_len,
        tdna_cut_count=len(cuts),
        tdna_cut_positions=cuts,
        left_border_zone_cut_count=len(left_cuts),
        right_border_zone_cut_count=len(right_cuts),
        left_border_ok=left_ok,
        right_border_ok=right_ok,
        tdna_no_cut=no_cut,
        selected_border_ok=sel_ok,
        ranking_bucket=bucket,
    )

# ---------------------------------------------------------------------------
# Summary collation helpers
# ---------------------------------------------------------------------------

def _row_sort_key(row: dict) -> tuple:
    """Three-criterion ranking, easy to explain and biologically motivated:

    1. **Border safety** — enzymes that cut the selected T-DNA border
       zone are pushed to the bottom (the iPCR primer site is
       destroyed there).
    2. **% usable insertions** — higher is better. This is the actual
       experimental success rate: the fraction of simulated insertions
       whose flanking fragment falls inside the iPCR size window.
    3. **T-DNA cut count** — fewer is better. Used as a tiebreaker
       between enzymes with similar usable rates, since each extra
       internal cut shortens the T-DNA-derived flanking fragment.
    """
    border_unsafe = 0 if row["selected_border_ok"] else 1
    return (
        border_unsafe,
        -float(row["pct_usable_insertions"]),
        int(row["tdna_cut_count"]),
    )


def build_summary_table(
    enzymes: List[EnzymeMetadata],
    motif_metrics: Dict[str, DigestMetrics],
    tdna_metrics: Dict[str, TransgeneMetrics],
    border: str,
) -> List[dict]:
    """One row per *enzyme* (isoschizomers each get their own row)."""
    rows: List[dict] = []
    for e in enzymes:
        m = motif_metrics[e.site]
        t = tdna_metrics[e.site]
        rows.append({
            "enzyme": e.enzyme,
            "site": e.site,
            "size": e.size,
            "cut_type": e.cut_type,
            "border_of_interest": border,
            "ranking_bucket": t.ranking_bucket,
            "tdna_cut_count": t.tdna_cut_count,
            "tdna_no_cut": t.tdna_no_cut,
            "selected_border_ok": t.selected_border_ok,
            "genome_site_count": m.genome_site_count,
            "sites_per_mb": round(m.sites_per_mb, 4),
            "n_fragments": m.n_fragments,
            "n_useful_fragments": m.n_useful_fragments,
            "pct_useful_fragments": round(m.pct_useful_fragments, 2),
            "median_fragment_bp": round(m.median_fragment_bp, 1),
            "n_insertions": m.n_insertions,
            "n_usable_insertions": m.n_usable_insertions,
            "pct_usable_insertions": round(m.pct_usable_insertions, 2),
            "insertion_median_bp": round(m.insertion_median_bp, 1),
        })
    rows.sort(key=_row_sort_key)
    return rows


def build_collapsed_summary(
    enzymes: List[EnzymeMetadata],
    motif_metrics: Dict[str, DigestMetrics],
    tdna_metrics: Dict[str, TransgeneMetrics],
    border: str,
) -> List[dict]:
    """One row per *unique motif* — isoschizomers are collapsed."""
    by_motif: Dict[str, List[EnzymeMetadata]] = defaultdict(list)
    for e in enzymes:
        by_motif[e.site].append(e)

    rows: List[dict] = []
    for motif, group in sorted(by_motif.items()):
        names = sorted(e.enzyme for e in group)
        m = motif_metrics[motif]
        t = tdna_metrics[motif]
        rows.append({
            "representative_enzyme": names[0],
            "isoschizomers": " | ".join(names),
            "isoschizomer_count": len(names),
            "site": motif,
            "size": group[0].size,
            "cut_type": group[0].cut_type,
            "border_of_interest": border,
            "ranking_bucket": t.ranking_bucket,
            "tdna_cut_count": t.tdna_cut_count,
            "tdna_no_cut": t.tdna_no_cut,
            "selected_border_ok": t.selected_border_ok,
            "genome_site_count": m.genome_site_count,
            "sites_per_mb": round(m.sites_per_mb, 4),
            "n_fragments": m.n_fragments,
            "n_useful_fragments": m.n_useful_fragments,
            "pct_useful_fragments": round(m.pct_useful_fragments, 2),
            "median_fragment_bp": round(m.median_fragment_bp, 1),
            "n_insertions": m.n_insertions,
            "n_usable_insertions": m.n_usable_insertions,
            "pct_usable_insertions": round(m.pct_usable_insertions, 2),
            "insertion_median_bp": round(m.insertion_median_bp, 1),
        })
    rows.sort(key=_row_sort_key)
    return rows


def row_label(row: dict) -> str:
    enzyme = row.get("representative_enzyme") or row.get("enzyme", "")
    site = row.get("site", "")
    return f"{enzyme} ({site})" if site else enzyme
