import json
import os
import sys
import time
import pandas as pd
import plotly.express as px
import streamlit as st
from pybiomart import Server
from openai import OpenAI

st.set_page_config(
    page_title="C.O.R.E | Precision CRISPR Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }

    div[data-testid="stMetric"] {
        background-color: #1a1c24;
        border: 1px solid #2d2f36;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: #00C9FF;
    }

    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
    }

    h1 {
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem !important;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #2d2f36;
        border-radius: 8px;
        overflow: hidden;
    }

    .risk-high { color: #FF4B4B; font-weight: bold; border: 1px solid #FF4B4B; padding: 2px 8px; border-radius: 4px; }
    .risk-med { color: #FFAA00; font-weight: bold; border: 1px solid #FFAA00; padding: 2px 8px; border-radius: 4px; }
    .risk-low { color: #00CC96; font-weight: bold; border: 1px solid #00CC96; padding: 2px 8px; border-radius: 4px; }
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
    st.error("CRITICAL: C++ Engine not found. Please compile the project first.")
    st.stop()


def get_chrom_rank(chrom_name):
    raw = str(chrom_name).replace("chr", "")
    if raw.isdigit():
        return int(raw)
    if raw == "X": return 23
    if raw == "Y": return 24
    if raw == "M" or raw == "MT": return 25
    return 99


class CoreService:
    @staticmethod
    @st.cache_resource(show_spinner=False)
    def get_engine(fasta_path: str):
        if not os.path.exists(fasta_path):
            st.error(f"Genome file not found: {fasta_path}")
            return None
        print(f"[CORE] Loading Genome: {fasta_path}")
        try:
            return core_engine.Engine(fasta_path)
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
                    attributes=['external_gene_name', 'description', 'gene_biotype'],
                    filters={'chromosome_name': chrom, 'start': start, 'end': end}
                )

                if not res.empty:
                    genes = res["Gene name"].dropna().unique()
                    descs = res["Gene description"].dropna().unique()
                    if len(genes) > 0:
                        annotated.at[idx, 'Gene'] = ", ".join(genes)
                    if len(descs) > 0:
                        clean_desc = descs[0].split("[")[0].strip()
                        annotated.at[idx, "Description"] = clean_desc
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
            "You are a Clinical Bioinformatics AI. "
            "Analyze these CRISPR off-target sites. "
            "Return a JSON object with a key 'analysis' containing a list of objects. "
            "Each object must have: 'id', 'gene_symbol', 'region_type' (Exon/Intron/Promoter), "
            "'risk_level' (High/Medium/Low), and 'rationale' (Short clinical explanation). "
            "NO MARKDOWN, ONLY JSON."
        )
        user_prompt = f"Analyze this dataset:\n{data_str}"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=4096,
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
        config_path = st.selectbox("Enzyme Model", ["./config/cas12a_config.json", "./config/spcas9_config.json"])
        st.subheader("2. Intelligence")
        api_key = st.text_input("NVIDIA API Key", type="password", help="Required for Clinical Analysis")
        st.markdown("---")
        st.caption("C.O.R.E v1.0")
        st.caption("Powered by NVIDIA CUDA & Nemotron")

    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.title("C.O.R.E")
        st.markdown("### CRISPR Off-target Real-time Engine")
    with col2:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/21/Nvidia_logo.svg/1200px-Nvidia_logo.svg.png",
                 width=150)

    engine = None
    if os.path.exists(fasta_path):
        with st.spinner("🚀 Booting C++ Hyper-Engine (AVX2/CUDA)..."):
            engine = CoreService.get_engine(fasta_path)
    else:
        st.error(f"❌ Genome file missing at {fasta_path}")

    st.markdown("<br>", unsafe_allow_html=True)
    query = st.text_input("🧬 Enter Guide Sequence (5' -> 3')", "TTTGGGGTGATCAGACCCAACAGCAGG")
    run = st.button("RUN DEEP SCAN", type="primary", width='stretch', disabled=(engine is None))

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
                "strand": h.strand
            })
        df = pd.DataFrame(data)

        if df.empty:
            status.update(label="No hits found.", state="complete")
            st.success("Target is highly specific. No off-targets found.")
            return

        df['chr_rank'] = df['chrom'].apply(get_chrom_rank)
        df = df.sort_values(["mm", "chr_rank", "pos"])
        df = df.drop(columns=['chr_rank'])
        status.write("Connecting to Ensembl BioMart...")
        annotator = BiodataService()
        df_top = df.head(50).copy()
        df_annotated = annotator.annotate_batch(df_top)

        status.update(label="Analysis Ready", state="complete", expanded=False)

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Engine Latency", f"{t_search * 1000:.1f} ms")
        kpi2.metric("Total Hits", len(df))
        kpi3.metric("Critical Off-Targets (0-2mm)", len(df[df['mm'] <= 2]))
        spec_score = max(0, 100 - (len(df[df['mm'] <= 3]) * 2))
        kpi4.metric("Specificity Score", f"{spec_score}/100", delta=spec_score - 80)

        col_chart_1, col_chart_2 = st.columns(2)
        with col_chart_1:
            st.subheader("Mismatch Distribution")
            fig_hist = px.histogram(df, x="mm", nbins=6, color_discrete_sequence=["#00C9FF"], template="plotly_dark")
            fig_hist.update_layout(bargap=0.1, height=300)
            st.plotly_chart(fig_hist, width="stretch")

        with col_chart_2:
            st.subheader("Chromosomal Spread")
            df_chart = df.copy()
            df_chart['chr_num'] = df_chart['chrom'].apply(get_chrom_rank)
            df_chart = df_chart.sort_values('chr_num')
            fig_scatter = px.scatter(
                df_chart, x="chrom", y="pos", color="mm",
                color_continuous_scale="Reds_r",
                template="plotly_dark",
                size_max=10,
                hover_data=["seq"]
            )
            fig_scatter.update_xaxes(categoryorder='array', categoryarray=df_chart['chrom'].unique())
            fig_scatter.update_layout(height=300)
            st.plotly_chart(fig_scatter, width="stretch")

        st.subheader("Genomic Candidates")
        st.dataframe(
            df_annotated[['chrom', 'pos', 'Gene', 'Description', 'seq', 'mm', 'strand']],
            column_config={
                "mm": st.column_config.NumberColumn(
                    "Mismatches",
                    help="Lower is more dangerous",
                    format="%d"
                ),
                "Gene": st.column_config.TextColumn("Gene Symbol", width="medium"),
                "seq": st.column_config.TextColumn("Sequence", width="large"),
                "Description": st.column_config.TextColumn("Function", width="large")
            },
            width="stretch",
            height=400
        )

        csv_data = df_annotated.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full Report (CSV)",
            data=csv_data,
            file_name=f"core_report_{int(time.time())}.csv",
            mime="text/csv"
        )

        if api_key:
            st.markdown("---")
            st.subheader("AI Clinical Risk Assessment")
            st.caption("Powered by NVIDIA LLAMA-3.3-Nemotron-Super-49b")

            candidates_for_ai = df_annotated[
                (df_annotated['Gene'] != ".") & (df_annotated['mm'] <= 5)
                ].head(50)

            if candidates_for_ai.empty:
                st.info("No gene-associated risks found in top candidates. AI analysis skipped.")
            else:
                payload = []
                for idx, row in candidates_for_ai.iterrows():
                    payload.append({
                        "id": idx,
                        "location": f"{row['chrom']}:{row['pos']}",
                        "sequence": row['seq'],
                        "gene": row['Gene'],
                        "mismatches": row['mm']
                    })

                analyst = RiskAnalystService(api_key)
                with st.spinner("Nemotron is analyzing gene functions and toxicity risks..."):
                    json_response = analyst.analyze_clinical_risk(payload)

                if json_response:
                    try:
                        parsed = json.loads(json_response)
                        analysis_list = parsed.get("analysis", [])
                        cols = st.columns(2)
                        for i, item in enumerate(analysis_list):
                            with cols[i % 2]:
                                r_lvl = item.get('risk_level', 'Unknown').lower()
                                risk_class = "risk-low"
                                border_color = "#00CC96"
                                if r_lvl == "high":
                                    risk_class = "risk-high"
                                    border_color = "#FF4B4B"
                                elif r_lvl == "medium":
                                    risk_class = "risk-med"
                                    border_color = "#FFAA00"

                                with st.container():
                                    st.markdown(f"""
                                    <div style="background-color: #262730; padding: 15px; border-radius: 8px; margin-bottom: 10px; border-left: 5px solid {border_color};">
                                        <div style="display: flex; justify-content: space-between; align-items: center;">
                                            <h4 style="margin: 0; color: white;">{item.get('gene_symbol', 'Unknown')}</h4>
                                            <span class="{risk_class}">{item.get('risk_level', 'Unknown').upper()}</span>
                                        </div>
                                        <p style="color: #bbb; font-size: 0.9em; margin-top: 5px;"><b>Region:</b> {item.get('region_type', 'N/A')}</p>
                                        <p style="color: #ddd; margin-top: 10px;">{item.get('rationale', 'No rationale provided.')}</p>
                                    </div>
                                    """, unsafe_allow_html=True)

                    except json.JSONDecodeError:
                        st.warning("AI generated non-JSON response.")
                else:
                    st.error("AI Analysis failed.")

        st.toast(f"Analysis complete. {len(df)} hits processed.", icon="✅")


if __name__ == "__main__":
    main()
