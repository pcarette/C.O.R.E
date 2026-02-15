import json
import os
import sys
import time
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from pybiomart import Server
from openai import OpenAI

st.set_page_config(
    page_title="C.O.R.E | Precision CRISPR Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }

    div[data-testid="stMetric"] {
        background-color: #1a1c24;
        border: 1px solid #2d2f36;
        padding: 15px;
        border-radius: 8px;
        transition: all 0.2s ease-in-out;
    }

    div[data-testid="stMetric"]:hover {
        border-color: #00C9FF;
        box-shadow: 0 0 15px rgba(0, 201, 255, 0.1);
    }

    h1, h2, h3 { font-family: 'Helvetica Neue', sans-serif; font-weight: 700; }
    h1 {
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem !important;
    }

    div[data-testid="stDataFrame"] { border: 1px solid #2d2f36; border-radius: 8px; }

    .risk-high { color: #ff4b4b; background: rgba(255, 75, 75, 0.1); border: 1px solid #ff4b4b; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
    .risk-med { color: #ffa700; background: rgba(255, 167, 0, 0.1); border: 1px solid #ffa700; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
    .risk-low { color: #00cc96; background: rgba(0, 204, 150, 0.1); border: 1px solid #00cc96; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

POSSIBLE_BUILD_DIRS = [
    os.path.join(os.getcwd(), "cmake-build-release"),
    os.path.join(os.getcwd(), "build"),
    os.getcwd()
]

CORE_LOADED = False
for d in POSSIBLE_BUILD_DIRS:
    if not os.path.exists(d):
        continue
    for f in os.listdir(d):
        if f.startswith("core_engine") and f.endswith(".so"):
            sys.path.append(d)
            try:
                import core_engine

                CORE_LOADED = True
                break
            except ImportError:
                continue
    if CORE_LOADED:
        break

if not CORE_LOADED:
    st.error("CRITICAL: C++ Engine (`core_engine.so`) not found. Compile the project first.")
    st.stop()


def get_chrom_rank(chrom_name):
    raw = str(chrom_name).replace("chr", "")
    if raw.isdigit(): return int(raw)
    if raw == "X": return 23
    if raw == "Y": return 24
    if raw in ["M", "MT"]: return 25
    return 99


class CoreService:
    @staticmethod
    @st.cache_resource(show_spinner=False)
    def get_engine(fasta_path: str, epi_path: str):
        if not os.path.exists(fasta_path):
            st.error(f"Genome file not found: {fasta_path}")
            return None
        has_epi = os.path.exists(epi_path)
        print(f"[CORE] Loading Genome: {fasta_path} | Epigenome: {epi_path if has_epi else 'None'}")
        try:
            return core_engine.Engine(fasta_path, epi_path if has_epi else "")
        except Exception as e:
            st.error(f"Engine Initialization Failed: {e}")
            return None


class BiodataService:
    def __init__(self):
        self.server = None
        self.dataset = None
        self.connected = False

    def connect(self):
        if self.connected:
            return
        try:
            self.server = Server(host='http://www.ensembl.org')
            self.dataset = self.server.marts['ENSEMBL_MART_ENSEMBL'].datasets['hsapiens_gene_ensembl']
            self.connected = True
        except Exception as e:
            print(f"[WARN] Biomart connection issues: {e}")

    def annotate_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        if not self.connected:
            self.connect()
        if not self.connected:
            df["Gene"] = "Connection Error"
            return df

        annotated = df.copy()
        annotated['Gene'] = "."
        annotated["Description"] = "."

        try:
            for idx, row in annotated.iterrows():
                chrom = str(row['chrom']).replace("chr", "")
                start = int(row['pos'])
                end = start + 25

                res = self.dataset.query(
                    attributes=['external_gene_name', 'description'],
                    filters={'chromosome_name': chrom, 'start': start, 'end': end}
                )

                if not res.empty:
                    genes = res["Gene name"].dropna().unique()
                    descs = res["Gene description"].dropna().unique()
                    if len(genes) > 0:
                        annotated.at[idx, 'Gene'] = ", ".join(genes)
                    if len(descs) > 0:
                        annotated.at[idx, "Description"] = descs[0].split("[")[0].strip()
        except Exception as e:
            print(f"[ERR] Annotation loop error: {e}")

        return annotated


class RiskAnalystService:
    def __init__(self, api_key):
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key
        )
        self.model = "nvidia/llama-3.3-nemotron-super-49b-v1.5"

    def analyze_clinical_risk(self, hits_list: list):
        data_str = json.dumps(hits_list, indent=2)
        system_prompt = (
            "You are a Clinical Bioinformatics AI specializing in CRISPR safety. "
            "Analyze these off-target sites based on mismatches, gene function, and EPIGENETIC CONTEXT. "
            "Context 'Open Chromatin' implies high accessibility and HIGH RISK of cleavage. "
            "Context 'Blacklist' implies unreliable genomic region (likely artifact). "
            "Return a JSON object with key 'analysis' containing a list of objects. "
            "Each object: 'id', 'gene_symbol', 'risk_level' (High/Medium/Low), 'rationale'."
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze this dataset:\n{data_str}"}
                ],
                temperature=0.2,
                max_tokens=2048,
                extra_body={"response_format": {"type": "json_object"}}
            )
            return response.choices[0].message.content
        except Exception as e:
            return None


def main():
    with st.sidebar:
        st.header("🎛️ Control Panel")
        st.subheader("1. Genomic Data")
        fasta_path = st.text_input("Genome Path (.fa)", "./data/hg38_full.fa")
        epi_path = st.text_input("Epigenome Path (.epi)", "./data/hg38_k562.epi")
        config_path = st.selectbox("Enzyme Model", ["./config/spcas9_config.json", "./config/cas12a_config.json"])
        st.subheader("2. Filters & Heuristics")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            req_atac = st.checkbox("Require Open Chromatin", value=False, help="Show only targets in ATAC-seq peaks")
        with col_f2:
            hide_bl = st.checkbox("Hide Blacklisted", value=True, help="Remove ENCODE blacklist regions")
        st.subheader("3. Intelligence")
        api_key = st.text_input("NVIDIA API Key", type="password")
        st.markdown("---")
        st.caption("C.O.R.E v1.2 (Heuristic Enhanced)")

    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.title("C.O.R.E")
        st.markdown("### CRISPR Off-target Real-time Engine")
    with col2:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/21/Nvidia_logo.svg/1200px-Nvidia_logo.svg.png",
                 width=120)

    engine = None
    if os.path.exists(fasta_path):
        with st.spinner("🚀 Booting C++ Hyper-Engine (AVX2/CUDA)..."):
            engine = CoreService.get_engine(fasta_path, epi_path)
    else:
        st.error(f"❌ Genome file missing at {fasta_path}")

    st.markdown("<br>", unsafe_allow_html=True)
    query = st.text_input("🧬 Enter Guide Sequence (20-23nt 5'->3')", "TTTGGGGTGATCAGACCCAACAGCAGG")
    run = st.button("RUN DEEP SCAN", type="primary", use_container_width=True, disabled=(engine is None))

    if run and engine:
        t_start = time.time()
        status = st.status("Processing Pipeline...", expanded=True)
        status.write("Running CUDA Kernel on HG38 Genome...")

        try:
            hits = engine.search(query, config_path)
            t_search = time.time() - t_start
            status.write(f"Genome scan complete in {t_search:.4f}s")
        except Exception as e:
            st.error(f"Kernel Error: {e}")
            st.stop()

        data = []
        for h in hits:
            data.append({
                "chrom": h.chrom,
                "pos": h.pos,
                "seq": h.sequence,
                "mm": h.mismatches,
                "strand": h.strand,
                "epi_code": h.epi_code
            })
        df = pd.DataFrame(data)

        if df.empty:
            status.update(label="No hits found.", state="complete")
            st.success("Target is highly specific. No off-targets found.")
            return

        status.write("Applying Epigenetic Masks & Heuristics...")
        MASK_ATAC = 1
        MASK_BLACKLIST = 2

        df['is_atac'] = (df['epi_code'] & MASK_ATAC) > 0
        df['is_blacklist'] = (df['epi_code'] & MASK_BLACKLIST) > 0

        if req_atac:
            df = df[df['is_atac']]
        if hide_bl:
            df = df[~df['is_blacklist']]

        if df.empty:
            status.update(label="All hits filtered out.", state="complete")
            st.warning("Hits found, but all were filtered by epigenetic criteria.")
            return

        df['base_score'] = 100.0 / (df['mm'] + 1.0)
        df['heuristic_score'] = df['base_score']

        df['heuristic_score'] = np.where(df['is_atac'], df['heuristic_score'] * 1.5, df['heuristic_score'])
        df['heuristic_score'] = np.where(df['is_blacklist'], df['heuristic_score'] * 0.5, df['heuristic_score'])
        df = df.sort_values("heuristic_score", ascending=False)

        status.write("Connecting to Ensembl BioMart...")
        annotator = BiodataService()
        df_top = df.head(50).copy()
        df_annotated = annotator.annotate_batch(df_top)

        status.update(label="Analysis Ready", state="complete", expanded=False)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Engine Latency", f"{t_search * 1000:.1f} ms")
        k2.metric("Total Hits", len(df))

        n_critical = len(df[(df['mm'] <= 2) | ((df['mm'] <= 3) & df['is_atac'])])
        k3.metric("Critical Risks", n_critical, delta_color="inverse", delta=n_critical)

        spec_score = max(0, 100 - (n_critical * 3))
        k4.metric("Safety Score", f"{spec_score}/100", delta=spec_score - 85)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Risk Distribution (Heuristic)")
            fig_hist = px.histogram(df, x="heuristic_score", nbins=20,
                                    color_discrete_sequence=["#00C9FF"], template="plotly_dark",
                                    title="Danger Score Distribution")
            fig_hist.update_layout(bargap=0.1, height=300)
            st.plotly_chart(fig_hist, use_container_width=True)

        with c2:
            st.subheader("Epigenetic Context")
            atac_count = df['is_atac'].sum()
            closed_count = len(df) - atac_count
            fig_pie = px.pie(names=["Open Chromatin", "Closed/Unknown"], values=[atac_count, closed_count],
                             color_discrete_sequence=["#00CC96", "#333"], template="plotly_dark", hole=0.4)
            fig_pie.update_layout(height=300)
            st.plotly_chart(fig_pie, use_container_width=True)

        st.subheader("Top Candidates Analysis")
        st.dataframe(
            df_annotated[['chrom', 'pos', 'Gene', 'seq', 'mm', 'is_atac', 'is_blacklist', 'heuristic_score']],
            column_config={
                "heuristic_score": st.column_config.ProgressColumn(
                    "Risk Score", format="%.1f", min_value=0, max_value=150,
                    help="Calculated based on Mismatches + Epigenetics"
                ),
                "is_atac": st.column_config.CheckboxColumn("Open Access", default=False),
                "is_blacklist": st.column_config.CheckboxColumn("Blacklisted", default=False),
                "mm": st.column_config.NumberColumn("Mismatches", format="%d"),
                "Gene": st.column_config.TextColumn("Gene", width="medium"),
                "seq": st.column_config.TextColumn("Sequence", width="large")
            },
            use_container_width=True,
            height=400
        )

        if api_key:
            st.markdown("---")
            st.subheader("🤖 AI Clinical Risk Assessment")
            candidates_for_ai = df_annotated[
                (df_annotated['Gene'] != ".") &
                (df_annotated['heuristic_score'] < 75)
                ].head(20)

            if candidates_for_ai.empty:
                st.info("No high-risk gene-associated targets found for AI analysis.")
            else:
                payload = []
                for idx, row in candidates_for_ai.iterrows():
                    context_str = "Closed Chromatin"
                    if row['is_atac']:
                        context_str = "OPEN CHROMATIN (High Accessibility)"
                    if row['is_blacklist']:
                        context_str += ", BLACKLISTED REGION"

                    payload.append({
                        "id": idx,
                        "location": f"{row['chrom']}:{row['pos']}",
                        "gene": row['Gene'],
                        "mismatches": row['mm'],
                        "epigenetic_context": context_str
                    })

                analyst = RiskAnalystService(api_key)
                with st.spinner(f"Analyzing {len(payload)} critical targets with Nemotron-49b..."):
                    json_response = analyst.analyze_clinical_risk(payload)

                if json_response:
                    try:
                        parsed = json.loads(json_response)
                        analysis_list = parsed.get("analysis", [])
                        ac1, ac2 = st.columns(2)
                        for i, item in enumerate(analysis_list):
                            col = ac1 if i % 2 == 0 else ac2
                            r_lvl = item.get('risk_level', 'Unknown').lower()

                            border_c = "#00CC96"
                            if "high" in r_lvl:
                                border_c = "#FF4B4B"
                            elif "medium" in r_lvl:
                                border_c = "#FFAA00"

                            with col:
                                st.markdown(f"""
                                <div style="background-color: #262730; padding: 15px; border-radius: 8px; margin-bottom: 10px; border-left: 5px solid {border_c};">
                                    <div style="display: flex; justify-content: space-between;">
                                        <h4 style="margin:0; color:white;">{item.get('gene_symbol', 'Unknown')}</h4>
                                        <span style="color:{border_c}; font-weight:bold;">{item.get('risk_level', 'UNKNOWN').upper()}</span>
                                    </div>
                                    <p style="color:#ccc; font-size:0.9em; margin-top:5px;">{item.get('rationale', 'No rationale')}</p>
                                </div>
                                """, unsafe_allow_html=True)

                    except json.JSONDecodeError:
                        st.warning("AI Response Parse Error")
                else:
                    st.error("AI Service Unavailable")


if __name__ == "__main__":
    main()
