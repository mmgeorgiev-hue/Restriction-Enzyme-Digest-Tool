"""
Streamlit front-end for the Brachypodium restriction-enzyme iPCR tool.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from engine import (
    clean_dna_sequence,
    load_biopython_enzymes,
    filter_practical_enzymes,
    read_fasta,
    build_motif_cut_index,
    sample_random_insertions,
    analyze_motif,
    evaluate_transgene,
    build_summary_table,
    build_collapsed_summary,
    row_label,
)
from plotting import (
    ranked_bar_chart,
    top_violin,
    best_mid_poor_histogram,
    sites_per_chromosome_heatmap,
)

# ── Default T-DNA ──────────────────────────────────────────────────────────
DEFAULT_TDNA = (
    "tacatggatcagcaatgagtatgatggtcaatatggagaaaaagaaagagtaattaccaat"
    "tttttttcaattcaaaaatgtagatgtccgcagcgttattataaaatgaaagtacattttg"
    "ataaaacgacaaattacgatccgtcgtatttataggcgaaagcaataaacaaattattcta"
    "attcggaaatctttatttcgacgtgtctacattcacgtccaaatgggggcttagatgagaa"
    "acttcacgatcggctctagtagtctgcagtgcagcgtgacccggtcgtgcccctctctag"
)

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Restriction Enzyme Digest Tool",
    page_icon="🧬",
    layout="wide",
)

st.title("Restriction Enzyme Digest & iPCR Prioritization")
st.markdown(
    "Upload a reference genome (FASTA), configure digest parameters, "
    "and get a ranked summary of restriction enzymes for iPCR experiments."
)

# ── Sidebar inputs ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Inputs")

    fasta_file = st.file_uploader(
        "Reference genome FASTA (.fa / .fa.gz)",
        type=["fa", "fasta", "gz"],
        help="Upload your genome assembly. Gzipped files are supported.",
    )

    st.subheader("Enzyme source")
    enzyme_source = st.radio(
        "Load enzymes from",
        ["Biopython (built-in)", "CSV file"],
        index=0,
    )
    enzyme_csv = None
    if enzyme_source == "CSV file":
        enzyme_csv = st.file_uploader(
            "Enzyme file (columns: Enzyme, Site — CSV or TXT)",
            type=["csv", "txt"],
        )

    st.subheader("Filter settings")
    allowed_sizes = st.multiselect(
        "Recognition site lengths (bp)",
        options=[4, 5, 6, 7, 8],
        default=[4, 5, 6],
    )
    require_palindromic = st.checkbox("Require palindromic", value=True)
    exclude_unusual = st.checkbox("Exclude unusual cut behavior", value=True)
    deduplicate_sites = st.checkbox(
        "Remove duplicate recognition sites (isoschizomers)",
        value=True,
        help="When enabled, only one representative enzyme is kept per unique "
             "recognition sequence. Disable to see all isoschizomers.",
    )

    st.subheader("Fragment size range")
    col_min, col_max = st.columns(2)
    useful_min = col_min.number_input("Min (bp)", value=500, step=100)
    useful_max = col_max.number_input("Max (bp)", value=5000, step=500)

    st.subheader("Insertion simulation")
    n_insertions = st.number_input("Number of random insertions", value=1000, step=100, min_value=10)
    seed = st.number_input("Random seed", value=7, step=1)

    st.subheader("T-DNA / transgene")
    border = st.selectbox("Border of interest", ["right", "left"])
    protected_bp = st.number_input("Protected border zone (bp)", value=300, step=50)

    with st.expander("T-DNA sequence (editable)"):
        tdna_raw = st.text_area(
            "Paste or edit the T-DNA sequence",
            value=DEFAULT_TDNA,
            height=180,
        )

    top_n_violin = st.slider("Top-N enzymes for violin plot", 5, 30, 12)

    run_btn = st.button("Run analysis", type="primary", use_container_width=True)

# ── Main panel ─────────────────────────────────────────────────────────────

if not run_btn:
    st.info("Configure parameters in the sidebar, then press **Run analysis**.")
    st.stop()

if fasta_file is None:
    st.error("Please upload a FASTA file.")
    st.stop()

if useful_min >= useful_max:
    st.error("Fragment min must be smaller than max.")
    st.stop()

# ── Step 1: Load enzymes ───────────────────────────────────────────────────
with st.status("Loading enzymes...", expanded=True) as status:
    if enzyme_source == "CSV file" and enzyme_csv is not None:
        from engine import EnzymeMetadata, reverse_complement, is_unambiguous_site
        import io as _io

        raw_bytes = enzyme_csv.read()
        for enc in ("utf-8-sig", "utf-16", "utf-8", "latin-1"):
            try:
                raw_text = raw_bytes.decode(enc)
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        else:
            raw_text = raw_bytes.decode("utf-8", errors="replace")
        raw_text = raw_text.replace('"', '').replace('\r', '')
        df_enz = pd.read_csv(_io.StringIO(raw_text))

        df_enz.columns = df_enz.columns.str.strip().str.lower()
        if "enzyme" not in df_enz.columns and len(df_enz.columns) >= 2:
            df_enz.columns = ["enzyme", "site"] + list(df_enz.columns[2:])

        required_cols = {"enzyme", "site"}
        if not required_cols.issubset(df_enz.columns):
            st.error(
                f"Could not find columns **enzyme** and **site** in your file. "
                f"Found columns: {list(df_enz.columns)}"
            )
            st.stop()

        all_enzymes = []
        for _, r in df_enz.iterrows():
            site = str(r["site"]).strip().upper()
            size = int(r.get("size", len(site))) if "size" in df_enz.columns else len(site)
            palindromic = (
                bool(site)
                and is_unambiguous_site(site)
                and site == reverse_complement(site)
            )
            all_enzymes.append(
                EnzymeMetadata(
                    enzyme=str(r["enzyme"]).strip(),
                    site=site,
                    size=size,
                    palindromic=palindromic,
                    cut_type=str(r.get("cut_type", "unknown")).strip() if "cut_type" in df_enz.columns else "unknown",
                    overhang_length=None, ovhgseq="", substrat="",
                    freq="", suppl="", opt_temp="", inact_temp="",
                    fst5=None, fst3=None, scd5=None, scd3=None,
                    charac="", uri="", enzyme_id="",
                )
            )
        st.write(f"Loaded **{len(all_enzymes)}** enzymes from file.")
    else:
        all_enzymes = load_biopython_enzymes()
        st.write(f"Loaded **{len(all_enzymes):,}** enzymes from Biopython.")

    filtered, reason_counts = filter_practical_enzymes(
        all_enzymes,
        allowed_sizes=tuple(allowed_sizes),
        require_palindromic=require_palindromic,
        require_unambiguous=True,
        exclude_unusual=exclude_unusual,
        deduplicate_sites=deduplicate_sites,
    )
    st.write(f"**{len(filtered):,}** enzymes pass filters.")
    if reason_counts:
        with st.expander("Filter exclusion details"):
            st.json(reason_counts)
    status.update(label=f"{len(filtered):,} enzymes ready", state="complete")

if not filtered:
    st.warning("No enzymes passed the filters. Try relaxing the criteria.")
    st.stop()

# ── Step 2: Read genome ───────────────────────────────────────────────────
with st.status("Reading FASTA genome...", expanded=True) as status:
    genome = read_fasta(fasta_file)
    contig_lengths = {c: len(s) for c, s in genome.items()}
    total_bp = sum(contig_lengths.values())
    st.write(f"**{len(genome)}** contigs, **{total_bp:,}** bp total.")
    status.update(label="Genome loaded", state="complete")

# ── Step 3: Scan genome for cut sites ─────────────────────────────────────
unique_motifs = sorted({e.site for e in filtered})
st.write(f"Scanning **{len(unique_motifs)}** unique motifs across the genome...")
scan_bar = st.progress(0, text="Scanning motifs...")


def _scan_progress(current: int, total: int):
    scan_bar.progress(current / total, text=f"Motif {current}/{total}")


cut_index = build_motif_cut_index(genome, unique_motifs, progress_callback=_scan_progress)
scan_bar.progress(1.0, text="Scan complete.")

# ── Step 4: Simulate insertions ───────────────────────────────────────────
insertions = sample_random_insertions(
    contig_lengths, int(n_insertions), seed=int(seed),
)

# ── Step 5: Analyze every motif ───────────────────────────────────────────
tdna_seq = clean_dna_sequence(tdna_raw)
motif_metrics = {}
motif_insertion_sizes = {}
tdna_metrics = {}

analysis_bar = st.progress(0, text="Analyzing motifs...")
for i, motif in enumerate(unique_motifs, 1):
    m, ins = analyze_motif(
        motif, cut_index, contig_lengths, insertions, int(useful_min), int(useful_max),
    )
    motif_metrics[motif] = m
    motif_insertion_sizes[motif] = ins
    tdna_metrics[motif] = evaluate_transgene(
        tdna_seq, motif, border, int(protected_bp),
    )
    analysis_bar.progress(i / len(unique_motifs), text=f"Motif {i}/{len(unique_motifs)}")

analysis_bar.progress(1.0, text="Analysis complete.")

# ── Step 6: Build summaries ───────────────────────────────────────────────
summary_rows = build_summary_table(filtered, motif_metrics, tdna_metrics, border)
collapsed_rows = build_collapsed_summary(filtered, motif_metrics, tdna_metrics, border)

insertion_sizes_by_label = {
    row_label(r): motif_insertion_sizes[r["site"]]
    for r in collapsed_rows
}

# ── Results ───────────────────────────────────────────────────────────────

st.divider()
st.header("Results")

# Summary table with isoschizomer toggle
show_iso = st.toggle(
    "Show isoschizomers",
    value=False,
    help="When enabled, every isoschizomer (enzyme sharing the same recognition "
         "site) gets its own row. When disabled, only one representative per "
         "unique site is shown.",
)

if show_iso:
    df_display = pd.DataFrame(summary_rows)
    csv_name = "enzyme_summary_all.csv"
else:
    df_display = pd.DataFrame(collapsed_rows)
    csv_name = "enzyme_summary.csv"

st.dataframe(df_display, use_container_width=True, height=450)
st.download_button(
    "Download CSV",
    df_display.to_csv(index=False).encode(),
    csv_name,
    "text/csv",
)

# Plots
st.subheader("Plots")
plot_tabs = st.tabs([
    "Ranked bar chart",
    "Violin (top N)",
    "Best / Median / Poor",
    "Cut-site heatmap",
])

with plot_tabs[0]:
    fig = ranked_bar_chart(collapsed_rows)
    st.pyplot(fig)

with plot_tabs[1]:
    fig = top_violin(collapsed_rows, insertion_sizes_by_label, top_n=top_n_violin)
    st.pyplot(fig)

with plot_tabs[2]:
    fig = best_mid_poor_histogram(
        collapsed_rows, insertion_sizes_by_label,
        useful_min=int(useful_min), useful_max=int(useful_max),
    )
    st.pyplot(fig)

with plot_tabs[3]:
    fig = sites_per_chromosome_heatmap(collapsed_rows, motif_metrics, top_n=15)
    st.pyplot(fig)

st.divider()
st.caption(
    "**Ranking logic:** enzymes with zero T-DNA cuts rank highest, "
    "then those preserving the selected border zone, then the rest. "
    "Within each tier, enzymes are sorted by % usable insertions descending."
)
