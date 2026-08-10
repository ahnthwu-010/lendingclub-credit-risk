import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import xgboost as xgb

st.set_page_config(page_title="LendingClub Credit Risk", layout="wide")

# ---------- Design tokens ----------
BG = "#0B0E14"
CARD_BG = "#141821"
TEXT = "#E8EAED"
TEXT_MUTED = "#8B92A0"
GRID = "#1F2430"
LINE = "#2A3040"
ACCENT = "#6366F1"

GRADE_ORDER = ["A", "B", "C", "D", "E", "F", "G"]
GRADE_COLORS = {"A": "#2DD4BF", "B": "#5EEAD4", "C": "#FBBF24", "D": "#F59E0B",
                "E": "#FB923C", "F": "#F87171", "G": "#EF4444"}
OUTCOME_COLORS = {"default_rate": "#EF4444", "prepaid_rate": "#2DD4BF", "censored_rate": "#6366F1"}
OUTCOME_LABELS = {"default_rate": "Default", "prepaid_rate": "Prepaid", "censored_rate": "Censored"}

# ---------- Custom Plotly template ----------
custom_template = go.layout.Template()
custom_template.layout = go.Layout(
    paper_bgcolor=BG, plot_bgcolor=BG,
    font=dict(family="IBM Plex Sans, sans-serif", color=TEXT, size=13),
    title=dict(font=dict(family="IBM Plex Sans, sans-serif", size=16, color=TEXT), x=0.02),
    colorway=list(GRADE_COLORS.values()),
    xaxis=dict(gridcolor=GRID, zerolinecolor=LINE, linecolor=LINE, tickfont=dict(color=TEXT_MUTED)),
    yaxis=dict(gridcolor=GRID, zerolinecolor=LINE, linecolor=LINE, tickfont=dict(color=TEXT_MUTED)),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_MUTED)),
    hoverlabel=dict(bgcolor=CARD_BG, font=dict(family="IBM Plex Mono, monospace", color=TEXT), bordercolor=LINE),
    margin=dict(t=50, l=10, r=10, b=10),
)
pio.templates["credit_dark"] = custom_template
pio.templates.default = "credit_dark"

# ---------- CSS injection ----------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
html, body, [class*="css"] {{ font-family: 'IBM Plex Sans', sans-serif; }}
[data-testid="stMetricValue"] {{ font-family: 'IBM Plex Mono', monospace; font-weight: 600; }}
[data-testid="stMetricLabel"] {{
    font-family: 'IBM Plex Sans', sans-serif; color: {TEXT_MUTED};
    font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em;
}}
[data-testid="stMetric"] {{
    background-color: {CARD_BG}; border: 1px solid {GRID};
    border-radius: 8px; padding: 1rem 1.2rem;
}}
h1, h2, h3 {{ font-family: 'IBM Plex Sans', sans-serif; letter-spacing: -0.02em; }}
[data-testid="stDataFrame"] {{ font-family: 'IBM Plex Mono', monospace; }}
</style>
""", unsafe_allow_html=True)

DATA = "data"
DASH = "data/dashboard_data"

@st.cache_data
def load_eda(): return pd.read_csv(f"{DASH}/eda_summary.csv")
@st.cache_data
def load_curves(): return pd.read_csv(f"{DASH}/survival_curves.csv")
@st.cache_data
def load_cif_grade(): return pd.read_csv(f"{DASH}/cif_by_grade.csv")
@st.cache_data
def load_hr(): return pd.read_csv(f"{DASH}/cox_hazard_ratios.csv")
@st.cache_data
def load_sample(): return pd.read_csv(f"{DASH}/sample_for_viz.csv")
@st.cache_data
def load_cate(): return pd.read_csv(f"{DATA}/lendingclub_with_cate.csv")
@st.cache_data
def load_optimization(): return pd.read_csv(f"{DATA}/lendingclub_final_optimization.csv")

st.title("LendingClub Credit Risk — Survival, Causal & Policy Dashboard")
st.caption("2.25M khoản vay · Competing Risks (Default vs Prepaid) · X-learner CATE · Verification Policy Optimization")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Tổng quan danh mục", "⏳ Rủi ro theo thời gian",
    "🎯 Yếu tố ảnh hưởng (Cox)", "🔬 Tác động xác minh (Causal)", "💰 Tối ưu hóa chính sách"
])

# ================= TAB 1: EDA =================
with tab1:
    eda, sample = load_eda(), load_sample()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tổng số khoản vay", "2,250,076")
    c2.metric("Default rate TB", f"{eda['default_rate'].mean():.1%}")
    c3.metric("Prepaid rate TB", f"{eda['prepaid_rate'].mean():.1%}")
    c4.metric("Lãi suất TB", f"{eda['avg_int_rate'].mean():.1f}%")

    st.subheader("Rủi ro theo Grade")
    col1, col2 = st.columns(2)
    with col1:
        long_eda = eda.melt(id_vars="grade", value_vars=list(OUTCOME_COLORS.keys()),
                             var_name="outcome", value_name="rate")
        long_eda["outcome_label"] = long_eda["outcome"].map(OUTCOME_LABELS)
        fig = px.bar(long_eda, x="grade", y="rate", color="outcome",
                     category_orders={"grade": GRADE_ORDER, "outcome": list(OUTCOME_COLORS.keys())},
                     color_discrete_map=OUTCOME_COLORS, barmode="stack",
                     custom_data=["outcome_label"], title="Phân bổ outcome theo Grade")
        fig.update_traces(hovertemplate="Grade %{x}<br>%{customdata[0]}: %{y:.1%}<extra></extra>")
        fig.update_yaxes(tickformat=".0%", title="Tỷ lệ")
        fig.update_layout(legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(eda, x="grade", y="avg_int_rate", category_orders={"grade": GRADE_ORDER},
                     color="grade", color_discrete_map=GRADE_COLORS, title="Lãi suất trung bình theo Grade")
        fig.update_traces(hovertemplate="Grade %{x}<br>Lãi suất TB: %{y:.2f}%<extra></extra>")
        fig.update_yaxes(title="Lãi suất (%)")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Phân phối FICO & DTI (mẫu 40k khoản vay)")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(sample, x="fico_range_low", color="grade",
                            category_orders={"grade": GRADE_ORDER}, color_discrete_map=GRADE_COLORS,
                            nbins=50, title="FICO score theo Grade")
        fig.update_traces(hovertemplate="FICO ~%{x}<br>Số khoản vay: %{y}<extra></extra>")
        fig.update_xaxes(title="FICO score"); fig.update_yaxes(title="Số khoản vay")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.box(sample, x="grade", y="dti", category_orders={"grade": GRADE_ORDER},
                     color="grade", color_discrete_map=GRADE_COLORS, title="DTI theo Grade")
        fig.update_traces(hovertemplate="Grade %{x}<br>DTI: %{y:.1f}<extra></extra>")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(eda.style.format({
        "avg_loan_amnt": "${:,.0f}", "avg_int_rate": "{:.2f}%", "avg_dti": "{:.1f}",
        "avg_fico": "{:.0f}", "default_rate": "{:.1%}", "prepaid_rate": "{:.1%}", "censored_rate": "{:.1%}"
    }), use_container_width=True)

# ================= TAB 2: Survival / Competing Risks =================
with tab2:
    curves, cif_grade = load_curves(), load_cif_grade()

    st.subheader("KM 'sai' vs Aalen-Johansen CIF đúng")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curves["month"], y=1 - curves["km_naive_survival"],
                              name="KM naive (SAI)", line=dict(dash="dash", color=TEXT_MUTED, width=2),
                              hovertemplate="Tháng %{x}<br>KM naive: %{y:.1%}<extra></extra>"))
    fig.add_trace(go.Scatter(x=curves["month"], y=curves["aj_cif_default"],
                              name="AJ CIF Default (ĐÚNG)", line=dict(color=OUTCOME_COLORS["default_rate"], width=3),
                              hovertemplate="Tháng %{x}<br>CIF Default: %{y:.1%}<extra></extra>"))
    fig.add_trace(go.Scatter(x=curves["month"], y=curves["aj_cif_prepaid"],
                              name="AJ CIF Prepaid", line=dict(color=OUTCOME_COLORS["prepaid_rate"], width=3),
                              hovertemplate="Tháng %{x}<br>CIF Prepaid: %{y:.1%}<extra></extra>"))

    gap_36 = curves.loc[curves["month"] == 36]
    if not gap_36.empty:
        naive_36 = float(1 - gap_36["km_naive_survival"].values[0])
        correct_36 = float(gap_36["aj_cif_default"].values[0])
        fig.add_vline(x=36, line_dash="dot", line_color=ACCENT, opacity=0.6)
        fig.add_annotation(x=36, y=naive_36, text=f"Naive: {naive_36:.1%}",
                            showarrow=True, arrowhead=2, ax=40, ay=-30,
                            font=dict(color=TEXT_MUTED, size=11))
        fig.add_annotation(x=36, y=correct_36, text=f"Đúng: {correct_36:.1%}",
                            showarrow=True, arrowhead=2, ax=40, ay=30,
                            font=dict(color=OUTCOME_COLORS["default_rate"], size=11))

    fig.update_layout(xaxis_title="Tháng", yaxis_title="Xác suất tích lũy",
                       title="Competing Risks: Default vs Prepaid theo thời gian")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    if not gap_36.empty:
        st.info(f"Tại tháng 36: chênh lệch **{naive_36-correct_36:.1%} điểm phần trăm** giữa "
                f"ước lượng naive và CIF đúng — do bỏ qua cạnh tranh với Prepaid.")

    st.subheader("CIF tại tháng 36 theo Grade")
    col1, col2 = st.columns([2, 1])
    with col1:
        long_cif = cif_grade.melt(id_vars=["grade", "n_loans"],
                                   value_vars=["cif_default_36m", "cif_prepaid_36m"],
                                   var_name="outcome", value_name="cif")
        long_cif["outcome_label"] = long_cif["outcome"].map(
            {"cif_default_36m": "Default", "cif_prepaid_36m": "Prepaid"})
        color_map = {"cif_default_36m": OUTCOME_COLORS["default_rate"], "cif_prepaid_36m": OUTCOME_COLORS["prepaid_rate"]}
        fig = px.bar(long_cif, x="grade", y="cif", color="outcome", barmode="group",
                     category_orders={"grade": GRADE_ORDER}, color_discrete_map=color_map,
                     custom_data=["outcome_label"], title="CIF Default vs Prepaid theo Grade (36 tháng)")
        fig.update_traces(hovertemplate="Grade %{x}<br>%{customdata[0]}: %{y:.1%}<extra></extra>")
        fig.update_yaxes(tickformat=".0%", title="CIF")
        fig.update_layout(legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.dataframe(cif_grade.style.format({
            "cif_default_36m": "{:.1%}", "cif_prepaid_36m": "{:.1%}", "n_loans": "{:,}"
        }), use_container_width=True)

# ================= TAB 3: Cox Hazard Ratios =================
with tab3:
    hr = load_hr()
    model_choice = st.radio("Chọn model", ["charged_off", "fully_paid"], horizontal=True,
                             format_func=lambda x: "Charged Off (Default)" if x == "charged_off" else "Fully Paid (Prepaid)")

    hr_m = hr[hr["model"] == model_choice].sort_values("hazard_ratio").copy()
    hr_m["significant"] = hr_m["p_value"] < 0.05
    marker_color = np.where(hr_m["significant"], OUTCOME_COLORS["default_rate"], TEXT_MUTED)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hr_m["hazard_ratio"], y=hr_m["covariate"], mode="markers",
        error_x=dict(type="data", symmetric=False,
                      array=hr_m["ci_upper"] - hr_m["hazard_ratio"],
                      arrayminus=hr_m["hazard_ratio"] - hr_m["ci_lower"],
                      color=LINE, thickness=1.5),
        marker=dict(color=marker_color, size=9, line=dict(color=BG, width=1)),
        customdata=np.stack([hr_m["ci_lower"], hr_m["ci_upper"], hr_m["p_value"]], axis=-1),
        hovertemplate="<b>%{y}</b><br>HR: %{x:.3f}<br>95%% CI: [%{customdata[0]:.3f}, %{customdata[1]:.3f}]"
                      "<br>p-value: %{customdata[2]:.2e}<extra></extra>"
    ))
    fig.add_vline(x=1, line_dash="dash", line_color=TEXT_MUTED)
    fig.update_layout(title=f"Hazard Ratios — {model_choice} (đỏ = p<0.05)",
                       xaxis_title="Hazard Ratio (95% CI)", height=700)
    fig.update_layout(title=f"Hazard Ratios — {model_choice} (đỏ = p<0.05)",
                   xaxis_title="Hazard Ratio (95% CI, log scale)", height=700)
    fig.update_xaxes(type="log")
    st.plotly_chart(fig, use_container_width=True)

    st.caption("HR > 1: tăng rủi ro. int_rate đã bị loại khỏi feature set do đa cộng tuyến với grade.")
    st.dataframe(hr_m.style.format({
        "hazard_ratio": "{:.3f}", "ci_lower": "{:.3f}", "ci_upper": "{:.3f}", "p_value": "{:.2e}"
    }), use_container_width=True)

# ================= TAB 4: Causal (CATE) =================
with tab4:
    cate_df = load_cate()
    ate = cate_df["CATE"].mean()
    pct_positive = (cate_df["CATE"] > 0).mean()
    raw_diff = cate_df[cate_df['treatment']==1]['default_flag'].mean() - cate_df[cate_df['treatment']==0]['default_flag'].mean()

    c1, c2, c3 = st.columns(3)
    c1.metric("ATE trung bình", f"{ate:.4f}", help="Dương = xác minh làm giảm xác suất vỡ nợ")
    c2.metric("% khoản vay có CATE dương", f"{pct_positive:.1%}")
    c3.metric("So sánh thô (chưa điều chỉnh)", f"{-raw_diff:.4f}", help="Chưa loại confounding")

    st.subheader("CATE trung bình theo Grade")
    cate_grade = cate_df.groupby("grade")["CATE"].agg(["mean", "std", "count"]).reindex(GRADE_ORDER).reset_index()
    fig = px.bar(cate_grade, x="grade", y="mean", error_y="std", color="grade",
                 color_discrete_map=GRADE_COLORS, title="CATE trung bình theo Grade (dương = xác minh có lợi)")
    fig.update_traces(hovertemplate="Grade %{x}<br>CATE TB: %{y:.4f}<extra></extra>")
    fig.add_hline(y=0, line_dash="dash", line_color=TEXT_MUTED)
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("CATE theo tứ phân vị DTI")
    cate_df["dti_quartile"] = pd.qcut(cate_df["dti"], q=4, labels=["Q1_thấp", "Q2", "Q3", "Q4_cao"])
    cate_dti = cate_df.groupby("dti_quartile", observed=True)["CATE"].mean().reset_index()
    fig = px.bar(cate_dti, x="dti_quartile", y="CATE", color_discrete_sequence=[ACCENT],
                 title="CATE trung bình theo mức DTI")
    fig.update_traces(hovertemplate="%{x}<br>CATE TB: %{y:.4f}<extra></extra>")
    fig.add_hline(y=0, line_dash="dash", line_color=TEXT_MUTED)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Phân phối CATE")
    fig = px.histogram(cate_df, x="CATE", nbins=80, color_discrete_sequence=[ACCENT],
                        title="Phân phối CATE toàn danh mục")
    fig.update_traces(hovertemplate="CATE ~%{x:.3f}<br>Số khoản vay: %{y}<extra></extra>")
    fig.add_vline(x=0, line_dash="dash", line_color=TEXT_MUTED)
    st.plotly_chart(fig, use_container_width=True)

# ================= TAB 5: Optimization =================
with tab5:
    opt = load_optimization()
    VERIFICATION_COST = 20

    n_worth = (opt["expected_value_verify"] > 0).sum()
    current_policy_value = opt["expected_value_verify"].sum() + len(opt) * VERIFICATION_COST
    current_policy_net = current_policy_value - len(opt) * VERIFICATION_COST
    selective_net = opt.loc[opt["expected_value_verify"] > 0, "expected_value_verify"].sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Số khoản đáng xác minh", f"{n_worth:,}", f"{n_worth/len(opt):.1%} tổng danh mục")
    c2.metric("Net value — đại trà", f"${current_policy_net:,.0f}")
    c3.metric("Net value — chọn lọc", f"${selective_net:,.0f}", f"+${selective_net - current_policy_net:,.0f}")

    st.subheader("% khoản vay đáng xác minh theo Grade")
    worth_by_grade = (opt.groupby("grade")
                       .apply(lambda g: (g["expected_value_verify"] > 0).mean(), include_groups=False)
                       .reindex(GRADE_ORDER).reset_index())
    worth_by_grade.columns = ["grade", "pct_worth_verifying"]
    fig = px.bar(worth_by_grade, x="grade", y="pct_worth_verifying", color="grade",
                 color_discrete_map=GRADE_COLORS, title="% đáng xác minh theo Grade")
    fig.update_traces(hovertemplate="Grade %{x}<br>%{y:.1%} đáng xác minh<extra></extra>")
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sensitivity Analysis — độ tin tưởng vào CATE")
    rows = []
    for s in [1.0, 0.75, 0.5, 0.25, 0.1]:
        ev_adj = opt["CATE"] * s * opt["potential_loss"] - VERIFICATION_COST
        rows.append({"Mức tin tưởng CATE": f"{int(s*100)}%", "Số khoản đáng xác minh": (ev_adj > 0).sum(),
                     "% tổng": (ev_adj > 0).mean(), "Net value chọn lọc ($)": ev_adj[ev_adj > 0].sum()})
    sens_df = pd.DataFrame(rows)
    fig = px.line(sens_df, x="Mức tin tưởng CATE", y="Net value chọn lọc ($)", markers=True,
                  color_discrete_sequence=[ACCENT], title="Net value theo độ tin tưởng vào model CATE")
    fig.update_traces(hovertemplate="Tin tưởng %{x}<br>Net value: $%{y:,.0f}<extra></extra>",
                       marker=dict(size=10))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(sens_df.style.format({"% tổng": "{:.1%}", "Net value chọn lọc ($)": "${:,.0f}"}),
                 use_container_width=True)

    st.caption(f"Chi phí xác minh giả định: ${VERIFICATION_COST}/hồ sơ (benchmark $15–25).")