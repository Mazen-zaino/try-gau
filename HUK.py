import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import anthropic
import json
from datetime import timedelta

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Fitout Optimization Engine",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;700&family=DM+Mono:wght@400;500&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .kpi-card {
        background: #161b22; border: 1px solid #21262d;
        border-radius: 10px; padding: 1.1rem 1.3rem; margin-bottom: 0.5rem;
    }
    .kpi-label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1.5px; color: #8b949e; margin-bottom: 0.3rem; }
    .kpi-value { font-size: 1.75rem; font-weight: 700; color: #f0f6fc; font-family: 'DM Mono', monospace; }
    .kpi-sub   { font-size: 0.75rem; color: #8b949e; margin-top: 0.2rem; }
    .good { color: #3fb950; } .warn { color: #d29922; } .bad { color: #f85149; }
    .ai-box {
        background: #161b22; border: 1px solid #30363d;
        border-left: 3px solid #58a6ff; border-radius: 10px;
        padding: 1.2rem 1.5rem; font-size: 0.88rem; line-height: 1.75; color: #c9d1d9;
    }
    [data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #21262d; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; background: #161b22; padding: 4px; border-radius: 8px; }
    .stTabs [data-baseweb="tab"] { background: transparent; border-radius: 6px; color: #8b949e; font-weight: 500; }
    .stTabs [aria-selected="true"] { background: #21262d !important; color: #f0f6fc !important; }
    .phase-header {
        font-size: 0.8rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 1px; color: #8b949e; padding: 0.4rem 0; margin-top: 0.5rem;
    }
    .stakeholder-tag {
        display: inline-block; background: #21262d; color: #c9d1d9;
        border-radius: 4px; padding: 2px 8px; font-size: 0.75rem; margin: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPER FUNCTIONS  (defined before any UI so
# topo_sort can call parse_deps safely)
# ─────────────────────────────────────────────
def parse_deps(dep):
    if pd.isna(dep) or str(dep).strip() == "":
        return []
    return [int(x.strip()) for x in str(dep).split(",") if x.strip().isdigit()]

def topo_sort(df):
    dep_map = {row["ID"]: parse_deps(row["Dependencies"]) for _, row in df.iterrows()}
    visited, order = set(), []
    def visit(tid):
        if tid in visited:
            return
        visited.add(tid)
        for d in dep_map.get(tid, []):
            visit(d)
        order.append(tid)
    for tid in df["ID"].tolist():
        visit(tid)
    id_to_pos = {tid: i for i, tid in enumerate(order)}
    return (df.assign(_sort=df["ID"].map(id_to_pos))
              .sort_values("_sort")
              .drop(columns="_sort")
              .reset_index(drop=True))

def compute_schedule(ids, durations, dep_lists):
    dur_map  = dict(zip(ids, durations))
    deps_map = dict(zip(ids, dep_lists))
    schedule = {}
    for tid in ids:
        deps  = deps_map[tid]
        start = max((schedule[d][1] for d in deps if d in schedule), default=0.0)
        schedule[tid] = (start, start + dur_map[tid])
    return schedule

def find_critical_path(ids, durations, dep_lists, base_sched, total):
    dur_map  = dict(zip(ids, durations))
    succ = {tid: [] for tid in ids}
    for tid, deps in zip(ids, dep_lists):
        for d in deps:
            if d in succ:
                succ[d].append(tid)
    late_start, late_finish = {}, {}
    for tid in reversed(ids):
        late_finish[tid] = total if not succ[tid] else min(late_start[s] for s in succ[tid])
        late_start[tid]  = late_finish[tid] - dur_map[tid]
    return [tid for tid in ids if abs(late_start[tid] - base_sched[tid][0]) < 0.01]

# ─────────────────────────────────────────────
# DEFAULT TASK DATA  (Stakeholders replaces Resources)
# ─────────────────────────────────────────────
DEFAULT_DF = pd.DataFrame({
    "ID": list(range(1, 26)),
    "Phase": [
        "Phase 0","Phase 0","Phase 0","Phase 0","Phase 0","Phase 0",
        "Design","Design","Design",
        "Procurement","Procurement",
        "Authority","Authority","Authority","Authority",
        "Pre-Execution",
        "Execution","Execution","Execution","Execution","Execution","Execution",
        "Testing","Authority","Opening",
    ],
    "Task": [
        "Business Approval","Site Feasibility","Concept Layout",
        "MEP Load Assessment","QS Estimate","Go Decision",
        "Concept Design","Detailed Design","IFC Drawings",
        "Long Lead Procurement","Local Procurement",
        "Civil Defense Approval","Mall Approval","DEWA Approval","Authority NOC",
        "Mobilization",
        "Demolition","MEP First Fix","Civil Works",
        "Second Fix","Equipment Install","Refrigeration Install",
        "Commissioning","Final Authority Approval","Store Opening",
    ],
    "Duration": [
        1,5,7,5,5,2,
        7,12,5,
        60,25,
        10,10,7,5,
        3,
        5,12,10,6,5,7,
        7,5,1,
    ],
    "Uncertainty": [
        1,1,2,2,2,1,
        2,3,2,
        5,3,
        4,3,3,2,
        1,
        2,3,3,2,2,2,
        2,2,0.5,
    ],
    "Cost_AED": [
        0,5000,15000,10000,8000,0,
        45000,95000,25000,
        380000,120000,
        12000,8000,9000,5000,
        18000,
        55000,180000,220000,
        95000,310000,145000,
        22000,6000,0,
    ],
    "Actual_Cost_AED": [0] * 25,
    "Stakeholders": [
        "CEO, CFO, Board",
        "Property Team, Project Manager",
        "Design Consultant, Project Manager",
        "MEP Engineer, Project Manager",
        "QS Consultant, CFO",
        "CEO, CFO, Board",
        "Design Consultant, Project Manager, Operator",
        "Design Consultant, MEP Engineer, Structural Engineer",
        "Design Consultant, MEP Engineer, QS",
        "Procurement Manager, Suppliers, Project Manager",
        "Procurement Manager, Local Suppliers",
        "Civil Defense, Consultant",
        "Mall Management, Leasing Team",
        "DEWA, MEP Consultant",
        "DM, Consultant, Project Manager",
        "Main Contractor, Site Manager, Project Manager",
        "Main Contractor, Site Manager",
        "MEP Subcontractor, Site Manager",
        "Main Contractor, Civil Sub, Site Manager",
        "Main Contractor, MEP Sub, Site Manager",
        "Equipment Supplier, Main Contractor, Site Manager",
        "Refrigeration Specialist, MEP Sub",
        "Commissioning Engineer, MEP Consultant, Operator",
        "Civil Defense, Mall Management, Project Manager",
        "Store Manager, Operations, Mall Management, Marketing",
    ],
    "Dependencies": [
        "","1","2","2","3","5",
        "6","7","8",
        "9","9",
        "9","9","9","9",
        "12,13,14,15",
        "16","17","18","19","20","18",
        "21,22","23","24",
    ],
})

# ─────────────────────────────────────────────
# SESSION STATE  — persists data across reruns
# ─────────────────────────────────────────────
if "project_data" not in st.session_state:
    st.session_state["project_data"] = DEFAULT_DF.copy()
if "data_version" not in st.session_state:
    st.session_state["data_version"] = 0

PHASE_COLORS = {
    "Phase 0":       "#58a6ff",
    "Design":        "#a371f7",
    "Procurement":   "#d29922",
    "Authority":     "#f0883e",
    "Pre-Execution": "#79c0ff",
    "Execution":     "#3fb950",
    "Testing":       "#56d364",
    "Opening":       "#ff7b72",
}
DARK = dict(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            font=dict(color="#c9d1d9", family="DM Sans"))

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:

    # ── Save / Load ───────────────────────────
    st.markdown("### 💾 Save & Load")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        save_data = st.session_state["project_data"].drop(
            columns=["Dep_List"], errors="ignore"
        ).to_json(orient="records", indent=2)
        st.download_button(
            "⬇️ Save JSON",
            data=save_data,
            file_name="fitout_project.json",
            mime="application/json",
            use_container_width=True,
        )
    with col_s2:
        uploaded = st.file_uploader("", type="json", label_visibility="collapsed")
        if uploaded:
            try:
                loaded_df = pd.read_json(uploaded)
                # Fill any missing columns with defaults
                for col in DEFAULT_DF.columns:
                    if col not in loaded_df.columns:
                        loaded_df[col] = DEFAULT_DF[col]
                st.session_state["project_data"] = loaded_df
                st.session_state["data_version"] += 1
                st.success("Loaded!")
                st.rerun()
            except Exception as e:
                st.error(f"Load error: {e}")

    st.divider()

    # ── Project Settings ──────────────────────
    st.markdown("### ⚙️ Project Settings")
    opening_date    = st.date_input("🎯 Target Opening Date",
                                    pd.to_datetime("2027-01-01"))
    buffer_days     = st.number_input("Buffer Days (before opening)", 0, 60, 7, step=1)
    budget          = st.number_input("Total Budget (AED)", value=2_500_000,
                                      step=50_000, format="%d")
    runs            = st.slider("Monte Carlo Runs", 200, 2000, 500, step=100)
    contingency_pct = st.slider("Contingency %", 5, 30, 15)

    st.divider()

    # ── Duration Sliders ──────────────────────
    st.markdown("### ⏱️ Adjust Durations")
    st.caption("Drag to update — all charts refresh instantly")
    BASE_DF = st.session_state["project_data"]
    dur_overrides = {}
    for _, r in BASE_DF.iterrows():
        default_val = int(r["Duration"])
        dur_overrides[int(r["ID"])] = st.slider(
            label=r["Task"],
            min_value=1,
            max_value=max(120, default_val * 3),
            value=default_val,
            step=1,
            key=f"dur_{int(r['ID'])}_{st.session_state['data_version']}",
        )

    st.divider()
    st.markdown("<small style='color:#8b949e;'>AI analysis uses Claude Sonnet</small>",
                unsafe_allow_html=True)

# ─────────────────────────────────────────────
# APPLY DURATION OVERRIDES
# ─────────────────────────────────────────────
ACTIVE_DF = st.session_state["project_data"].copy()
for tid, dur in dur_overrides.items():
    ACTIVE_DF.loc[ACTIVE_DF["ID"] == tid, "Duration"] = dur

# ─────────────────────────────────────────────
# SORT TOGGLE + TASK EDITOR
# ─────────────────────────────────────────────
st.markdown("# 🏗️ Fitout Optimization Engine")
st.markdown(
    "<span style='color:#8b949e;font-size:0.85rem;'>"
    "Reverse-Scheduled · Monte Carlo · Critical Path · AI Analysis</span>",
    unsafe_allow_html=True,
)
st.markdown("<br>", unsafe_allow_html=True)

sort_col, _ = st.columns([3, 5])
with sort_col:
    sort_mode = st.radio(
        "Task order",
        ["By ID", "Logical Flow (dependencies)"],
        horizontal=True,
        label_visibility="collapsed",
    )

DISPLAY_DF = topo_sort(ACTIVE_DF) if "Logical" in sort_mode else ACTIVE_DF.copy()

with st.expander("📋 Edit Task Register", expanded=True):
    tasks = st.data_editor(
        DISPLAY_DF,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Cost_AED":        st.column_config.NumberColumn("Budget (AED)",         format="%d"),
            "Actual_Cost_AED": st.column_config.NumberColumn("Actual Cost (AED)",    format="%d"),
            "Uncertainty":     st.column_config.NumberColumn("Uncertainty (σ days)", format="%.1f"),
            "Stakeholders":    st.column_config.TextColumn("Stakeholders"),
        },
        disabled=["ID", "Phase"],
        key=f"task_editor_{st.session_state['data_version']}",
    )

# Persist edits back to session state
st.session_state["project_data"] = tasks.drop(columns=["Dep_List"], errors="ignore")
tasks = tasks.reset_index(drop=True)

# ─────────────────────────────────────────────
# SCHEDULE COMPUTATION  (REVERSE from opening date)
# ─────────────────────────────────────────────
tasks["Dep_List"] = tasks["Dependencies"].apply(parse_deps)
ids       = tasks["ID"].tolist()
durations = tasks["Duration"].tolist()
dep_lists = tasks["Dep_List"].tolist()

base_sched = compute_schedule(ids, durations, dep_lists)
base_duration = max(v[1] for v in base_sched.values())

# Reverse: project start = opening_date - base_duration - buffer
project_start = pd.to_datetime(opening_date) - timedelta(days=int(base_duration) + int(buffer_days))

# Build gantt dataframe
gantt_df = tasks.copy()
gantt_df["Start_Day"]  = [base_sched[i][0] for i in ids]
gantt_df["End_Day"]    = [base_sched[i][1] for i in ids]
gantt_df["Start_Date"] = project_start + pd.to_timedelta(gantt_df["Start_Day"], unit="D")
gantt_df["End_Date"]   = project_start + pd.to_timedelta(gantt_df["End_Day"],   unit="D")
gantt_df["Float"]      = (gantt_df["End_Day"] - gantt_df["Start_Day"] - gantt_df["Duration"]).round(1)

# Critical path
critical_ids = find_critical_path(ids, durations, dep_lists, base_sched, base_duration)
gantt_df["Critical"] = gantt_df["ID"].isin(critical_ids)

# Days from today to opening
today = pd.Timestamp.today().normalize()
days_to_opening = (pd.to_datetime(opening_date) - today).days
days_to_start   = (project_start - today).days

# ─────────────────────────────────────────────
# MONTE CARLO
# ─────────────────────────────────────────────
durations_arr   = tasks["Duration"].values.astype(float)
uncertainty_arr = tasks["Uncertainty"].values.astype(float)
rng = np.random.default_rng(seed=42)

mc_totals    = np.empty(runs)
task_ends_mc = np.empty((len(ids), runs))

for r in range(runs):
    sim_durs = np.maximum(rng.normal(durations_arr, uncertainty_arr), 0.5)
    sched_r  = compute_schedule(ids, sim_durs.tolist(), dep_lists)
    mc_totals[r] = max(v[1] for v in sched_r.values())
    for k, tid in enumerate(ids):
        task_ends_mc[k, r] = sched_r[tid][1]

available_days = int(base_duration) + int(buffer_days)
mean_days   = float(mc_totals.mean())
p80         = float(np.percentile(mc_totals, 80))
p95         = float(np.percentile(mc_totals, 95))
on_time_pct = float(np.mean(mc_totals <= available_days) * 100)

# ─────────────────────────────────────────────
# SENSITIVITY (Tornado)
# ─────────────────────────────────────────────
sens_rows = []
for k, tid in enumerate(ids):
    row  = tasks[tasks["ID"] == tid].iloc[0]
    corr = float(np.corrcoef(task_ends_mc[k], mc_totals)[0, 1])
    sens_rows.append({"Task": row["Task"], "Phase": row["Phase"], "ID": tid, "Correlation": corr})
sens_df = pd.DataFrame(sens_rows).sort_values("Correlation", ascending=False).reset_index(drop=True)

# ─────────────────────────────────────────────
# COST SUMMARY
# ─────────────────────────────────────────────
total_cost   = int(tasks["Cost_AED"].sum())
total_actual = int(tasks["Actual_Cost_AED"].sum()) if "Actual_Cost_AED" in tasks.columns else 0
contingency  = int(total_cost * contingency_pct / 100)
total_needed = total_cost + contingency
budget_status = "under" if total_needed <= int(budget) else "over"

# ─────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────
ot_cls  = "good" if on_time_pct >= 70 else ("warn" if on_time_pct >= 40 else "bad")
bud_cls = "good" if budget_status == "under" else "bad"
start_cls = "warn" if days_to_start < 0 else "good"
start_label = f"{abs(days_to_start)}d {'overdue' if days_to_start < 0 else 'to start'}"

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">Opening Date</div>
        <div class="kpi-value" style="font-size:1.3rem">{pd.to_datetime(opening_date).strftime('%d %b %Y')}</div>
        <div class="kpi-sub">{days_to_opening} days from today</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">Implied Start Date</div>
        <div class="kpi-value {start_cls}" style="font-size:1.2rem">{project_start.strftime('%d %b %Y')}</div>
        <div class="kpi-sub">{start_label}</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">On-Time Probability</div>
        <div class="kpi-value {ot_cls}">{on_time_pct:.0f}%</div>
        <div class="kpi-sub">P80: {p80:.0f}d · P95: {p95:.0f}d · Mean: {mean_days:.0f}d</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">Budget w/ Contingency</div>
        <div class="kpi-value {bud_cls}">AED {total_needed/1e6:.2f}M</div>
        <div class="kpi-sub">Spent: AED {total_actual/1e6:.2f}M · {contingency_pct}% contingency</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Master Gantt",
    "🔍 Phase Breakdown",
    "⚠️ Risk Analysis",
    "💰 Cost Tracking",
    "👥 Stakeholders",
    "🤖 AI Analysis",
])

today_str = pd.Timestamp.today().strftime("%Y-%m-%d")
opening_str = pd.to_datetime(opening_date).strftime("%Y-%m-%d")

def add_reference_lines(fig):
    """Add Today and Opening Date reference lines to any timeline figure."""
    fig.add_shape(type="line", x0=today_str, x1=today_str, y0=0, y1=1,
                  xref="x", yref="paper", line=dict(color="#58a6ff", dash="dot", width=1))
    fig.add_annotation(x=today_str, y=1, xref="x", yref="paper",
                       text="Today", showarrow=False,
                       font=dict(color="#58a6ff", size=10), yanchor="bottom")
    fig.add_shape(type="line", x0=opening_str, x1=opening_str, y0=0, y1=1,
                  xref="x", yref="paper", line=dict(color="#ff7b72", dash="dash", width=1.5))
    fig.add_annotation(x=opening_str, y=0.95, xref="x", yref="paper",
                       text="🎯 Opening", showarrow=False,
                       font=dict(color="#ff7b72", size=10), yanchor="bottom")
    return fig

# ── TAB 1: MASTER GANTT (phase-level) ─────────
with tab1:
    st.markdown("#### Master Project Timeline — Phase Overview")
    st.caption("Each bar represents a full phase. Drill into Phase Breakdown tab for task detail.")

    # Aggregate to phase level
    phase_agg = (
        gantt_df.groupby("Phase")
        .agg(Start_Date=("Start_Date", "min"), End_Date=("End_Date", "max"))
        .reset_index()
    )
    # Preserve phase order by first appearance
    phase_order = gantt_df.drop_duplicates("Phase").set_index("Phase")["Start_Day"]
    phase_agg["_order"] = phase_agg["Phase"].map(phase_order)
    phase_agg = phase_agg.sort_values("_order").drop(columns="_order")
    phase_agg["Duration_Days"] = (phase_agg["End_Date"] - phase_agg["Start_Date"]).dt.days
    phase_agg["Color"] = phase_agg["Phase"].map(PHASE_COLORS)

    # Count tasks per phase and critical tasks
    phase_task_count = gantt_df.groupby("Phase")["Task"].count()
    phase_crit_count = gantt_df[gantt_df["Critical"]].groupby("Phase")["Task"].count()
    phase_cost       = tasks.groupby("Phase")["Cost_AED"].sum()

    fig_master = px.timeline(
        phase_agg,
        x_start="Start_Date",
        x_end="End_Date",
        y="Phase",
        color="Phase",
        color_discrete_map=PHASE_COLORS,
        custom_data=["Duration_Days", "Phase"],
    )
    fig_master.update_traces(
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Start: %{base|%d %b %Y}<br>"
            "End: %{x|%d %b %Y}<br>"
            "Duration: %{customdata[0]} days<extra></extra>"
        )
    )
    fig_master.update_yaxes(autorange="reversed")
    fig_master.update_layout(
        **DARK,
        height=380,
        margin=dict(l=10, r=10, t=30, b=30),
        xaxis=dict(gridcolor="#21262d", tickformat="%b %Y"),
        yaxis=dict(gridcolor="#21262d"),
        showlegend=False,
    )
    add_reference_lines(fig_master)
    st.plotly_chart(fig_master, use_container_width=True)

    # Phase summary table
    st.markdown("#### Phase Summary")
    phase_summary = phase_agg[["Phase","Start_Date","End_Date","Duration_Days"]].copy()
    phase_summary["Tasks"]          = phase_summary["Phase"].map(phase_task_count).fillna(0).astype(int)
    phase_summary["Critical Tasks"] = phase_summary["Phase"].map(phase_crit_count).fillna(0).astype(int)
    phase_summary["Budget (AED)"]   = phase_summary["Phase"].map(phase_cost).fillna(0).apply(lambda x: f"{x:,.0f}")
    phase_summary["Start_Date"]     = phase_summary["Start_Date"].dt.strftime("%d %b %Y")
    phase_summary["End_Date"]       = phase_summary["End_Date"].dt.strftime("%d %b %Y")
    phase_summary = phase_summary.rename(columns={
        "Start_Date": "Start", "End_Date": "End", "Duration_Days": "Days"
    })
    st.dataframe(phase_summary, use_container_width=True, hide_index=True)

    # Critical path callout
    cp_names = tasks[tasks["ID"].isin(critical_ids)]["Task"].tolist()
    st.caption("🔴 Critical path: " + " → ".join(cp_names))

# ── TAB 2: PHASE BREAKDOWN ────────────────────
with tab2:
    st.markdown("#### Phase Breakdown — Task-Level Detail")
    st.caption("Expand each phase to see individual task Gantt and details.")

    phases_ordered = gantt_df.drop_duplicates("Phase").sort_values("Start_Day")["Phase"].tolist()

    for phase in phases_ordered:
        phase_tasks = gantt_df[gantt_df["Phase"] == phase].copy()
        phase_color = PHASE_COLORS.get(phase, "#58a6ff")

        with st.expander(
            f"**{phase}** — {len(phase_tasks)} tasks · "
            f"{int(phase_tasks['Start_Day'].min())}–{int(phase_tasks['End_Day'].max())} days",
            expanded=False,
        ):
            # Task-level Gantt for this phase
            pt = phase_tasks.sort_values("Start_Day").copy()
            pt["Label"] = pt.apply(
                lambda r: f"🔴 {r['Task']}" if r["Critical"] else r["Task"], axis=1
            )
            pt["Bar_Color"] = pt["Critical"].apply(
                lambda c: "#f85149" if c else phase_color
            )
            pt["Hover"] = pt.apply(lambda r: (
                f"<b>{r['Task']}</b><br>"
                f"Start: {r['Start_Date'].strftime('%d %b %Y')}<br>"
                f"End: {r['End_Date'].strftime('%d %b %Y')}<br>"
                f"Duration: {int(r['Duration'])} days<br>"
                f"Stakeholders: {r['Stakeholders']}<br>"
                + ("⚠️ CRITICAL PATH" if r["Critical"] else f"Float: {r['Float']} days")
            ), axis=1)

            # Zoom to this phase's date window so bars are always visible
            ph_min = pt["Start_Date"].min() - pd.Timedelta(days=3)
            ph_max = pt["End_Date"].max()   + pd.Timedelta(days=3)

            fig_ph = px.timeline(
                pt,
                x_start="Start_Date",
                x_end="End_Date",
                y="Label",
                color="Bar_Color",
                color_discrete_map="identity",
                custom_data=["Hover"],
            )
            fig_ph.update_traces(hovertemplate="%{customdata[0]}<extra></extra>")
            fig_ph.update_yaxes(autorange="reversed")
            fig_ph.update_layout(
                **DARK,
                height=max(200, len(pt) * 38 + 60),
                margin=dict(l=10, r=20, t=20, b=20),
                xaxis=dict(
                    gridcolor="#21262d", tickformat="%d %b '%y",
                    range=[ph_min, ph_max], tickangle=-20,
                    automargin=True,
                ),
                yaxis=dict(gridcolor="#21262d", automargin=True),
                showlegend=False,
            )
            # Only draw ref lines if they fall inside this phase window
            _today_ts   = pd.Timestamp(today_str)
            _opening_ts = pd.Timestamp(opening_str)
            if ph_min <= _today_ts <= ph_max:
                fig_ph.add_shape(type="line", x0=today_str, x1=today_str, y0=0, y1=1,
                                 xref="x", yref="paper",
                                 line=dict(color="#58a6ff", dash="dot", width=1))
                fig_ph.add_annotation(x=today_str, y=1, xref="x", yref="paper",
                                      text="Today", showarrow=False,
                                      font=dict(color="#58a6ff", size=10), yanchor="bottom")
            if ph_min <= _opening_ts <= ph_max:
                fig_ph.add_shape(type="line", x0=opening_str, x1=opening_str, y0=0, y1=1,
                                 xref="x", yref="paper",
                                 line=dict(color="#ff7b72", dash="dash", width=1.5))
                fig_ph.add_annotation(x=opening_str, y=0.92, xref="x", yref="paper",
                                      text="Opening", showarrow=False,
                                      font=dict(color="#ff7b72", size=10), yanchor="bottom")
            st.plotly_chart(fig_ph, use_container_width=True)

            # Task detail table for this phase
            detail_cols = ["Task","Duration","Uncertainty","Stakeholders","Cost_AED","Actual_Cost_AED"]
            avail_cols  = [c for c in detail_cols if c in pt.columns]
            st.dataframe(
                pt[avail_cols].rename(columns={
                    "Cost_AED": "Budget (AED)", "Actual_Cost_AED": "Actual (AED)"
                }),
                use_container_width=True,
                hide_index=True,
            )

# ── TAB 3: RISK ANALYSIS ──────────────────────
with tab3:
    col_r1, col_r2 = st.columns(2)

    with col_r1:
        st.markdown("#### Completion Probability (S-Curve)")
        sorted_mc = np.sort(mc_totals)
        prob = np.linspace(0, 1, len(sorted_mc))
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=project_start + pd.to_timedelta(sorted_mc.astype(int), unit="D"),
            y=prob,
            fill="tozeroy", fillcolor="rgba(88,166,255,0.07)",
            line=dict(color="#58a6ff", width=2),
            hovertemplate="Finish: %{x|%d %b %Y}<br>Probability: %{y:.1%}<extra></extra>",
        ))
        for val, lbl, col in [
            (mean_days, "Mean",    "#3fb950"),
            (p80,       "P80",     "#d29922"),
            (p95,       "P95",     "#f85149"),
        ]:
            x_date = (project_start + timedelta(days=int(val))).strftime("%Y-%m-%d")
            fig_sc.add_shape(type="line", x0=x_date, x1=x_date, y0=0, y1=1,
                             xref="x", yref="paper", line=dict(color=col, dash="dash", width=1.5))
            fig_sc.add_annotation(x=x_date, y=0.5 if lbl=="Mean" else (0.7 if lbl=="P80" else 0.9),
                                  xref="x", yref="paper", text=f"{lbl} {val:.0f}d",
                                  showarrow=False, font=dict(color=col, size=10))
        # Opening date line
        fig_sc.add_shape(type="line", x0=opening_str, x1=opening_str, y0=0, y1=1,
                         xref="x", yref="paper", line=dict(color="#ff7b72", dash="dot", width=1.5))
        fig_sc.add_annotation(x=opening_str, y=0.15, xref="x", yref="paper",
                               text="🎯 Opening", showarrow=False,
                               font=dict(color="#ff7b72", size=10))
        fig_sc.update_layout(
            **DARK, height=320,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis=dict(title="Projected Finish Date", gridcolor="#21262d", tickformat="%b %Y"),
            yaxis=dict(title="Cumulative Probability", gridcolor="#21262d", tickformat=".0%"),
            showlegend=False,
        )
        st.plotly_chart(fig_sc, use_container_width=True)

    with col_r2:
        st.markdown("#### Tornado Chart — Schedule Sensitivity")
        top_sens   = sens_df.head(12)
        bar_colors = [
            "#f85149" if c > 0.6 else "#d29922" if c > 0.3 else "#3fb950"
            for c in top_sens["Correlation"]
        ]
        fig_tor = go.Figure(go.Bar(
            y=top_sens["Task"], x=top_sens["Correlation"],
            orientation="h", marker_color=bar_colors,
            hovertemplate="%{y}: r = %{x:.2f}<extra></extra>",
        ))
        fig_tor.update_layout(
            **DARK, height=320,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis=dict(title="Pearson r with total duration", gridcolor="#21262d", range=[0, 1]),
            yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
        )
        st.plotly_chart(fig_tor, use_container_width=True)

    st.markdown("#### Risk Register")
    risk_reg = tasks.copy()
    risk_reg["Risk_Score"] = (risk_reg["Duration"] * risk_reg["Uncertainty"]).round(1)
    risk_reg["Impact"]     = risk_reg["Risk_Score"].apply(
        lambda x: "🔴 High" if x > 50 else ("🟠 Medium" if x > 20 else "🟢 Low")
    )
    risk_reg["Critical"]    = risk_reg["ID"].isin(critical_ids).map({True: "✅ Yes", False: "—"})
    risk_reg["P95 Finish"]  = [
        (project_start + timedelta(days=int(np.percentile(task_ends_mc[k], 95)))).strftime("%d %b %Y")
        for k in range(len(ids))
    ]
    st.dataframe(
        risk_reg[["Phase","Task","Duration","Uncertainty","Risk_Score","Impact","Critical","P95 Finish","Stakeholders"]
                 ].sort_values("Risk_Score", ascending=False),
        use_container_width=True, hide_index=True,
    )

# ── TAB 4: COST TRACKING ──────────────────────
with tab4:
    col_c1, col_c2 = st.columns(2)

    with col_c1:
        st.markdown("#### Budget by Phase")
        cost_phase = tasks[tasks["Cost_AED"] > 0].groupby("Phase")["Cost_AED"].sum().reset_index()
        fig_pie = px.pie(
            cost_phase, names="Phase", values="Cost_AED", hole=0.55,
            color_discrete_sequence=list(PHASE_COLORS.values()),
        )
        fig_pie.update_traces(
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>AED %{value:,.0f}<extra></extra>",
        )
        fig_pie.update_layout(**DARK, height=280, margin=dict(l=10, r=10, t=10, b=10),
                               legend=dict(font=dict(size=11)))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_c2:
        st.markdown("#### Budget Summary")
        rows = [
            ("Base Cost",                     total_cost),
            (f"Contingency ({contingency_pct}%)", contingency),
            ("Total Required",                total_needed),
            ("Approved Budget",               int(budget)),
            ("Variance",                      int(budget) - total_needed),
            ("Total Spent",                   total_actual),
            ("Remaining to Spend",            total_cost - total_actual),
        ]
        for label, val in rows:
            color = "#f85149" if (label == "Variance" and val < 0) else \
                    "#3fb950" if label in ("Variance", "Remaining to Spend") else "#c9d1d9"
            sign  = "+" if (label == "Variance" and val > 0) else ""
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"padding:7px 0;border-bottom:1px solid #21262d;'>"
                f"<span style='color:#8b949e'>{label}</span>"
                f"<span style='font-family:DM Mono,monospace;color:{color}'>{sign}{val:,.0f}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("#### Budget vs Actual — by Task")
    bva = tasks[tasks["Cost_AED"] > 0].sort_values("Cost_AED", ascending=False).head(15)
    fig_bva = go.Figure()
    fig_bva.add_trace(go.Bar(name="Budget", x=bva["Task"], y=bva["Cost_AED"],
                             marker_color="#58a6ff",
                             hovertemplate="<b>%{x}</b><br>Budget: AED %{y:,.0f}<extra></extra>"))
    fig_bva.add_trace(go.Bar(name="Actual", x=bva["Task"], y=bva["Actual_Cost_AED"],
                             marker_color="#3fb950",
                             hovertemplate="<b>%{x}</b><br>Actual: AED %{y:,.0f}<extra></extra>"))
    fig_bva.update_layout(**DARK, barmode="group", height=320,
                          margin=dict(l=10, r=10, t=10, b=10),
                          xaxis=dict(gridcolor="#21262d", tickangle=-35),
                          yaxis=dict(gridcolor="#21262d", title="AED"),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    st.plotly_chart(fig_bva, use_container_width=True)

    st.markdown("#### Cost Tracker")
    cost_track = tasks[["Phase","Task","Cost_AED","Actual_Cost_AED"]].copy()
    cost_track["Variance"] = cost_track["Cost_AED"] - cost_track["Actual_Cost_AED"]
    cost_track["% Spent"]  = np.where(
        cost_track["Cost_AED"] > 0,
        (cost_track["Actual_Cost_AED"] / cost_track["Cost_AED"] * 100).round(1), 0.0
    )
    cost_track["Status"] = cost_track.apply(lambda r: (
        "⬜ Not started" if r["Actual_Cost_AED"] == 0 else
        "🔴 Over budget" if r["Actual_Cost_AED"] > r["Cost_AED"] else "🟢 On/under budget"
    ), axis=1)
    cost_track = cost_track.rename(columns={"Cost_AED": "Budget (AED)", "Actual_Cost_AED": "Actual (AED)"})
    st.dataframe(cost_track.sort_values("Variance"), use_container_width=True, hide_index=True)

# ── TAB 5: STAKEHOLDERS ───────────────────────
with tab5:
    st.markdown("#### Stakeholder Engagement Timeline")
    st.caption("When is each stakeholder active across the project?")

    # Parse stakeholders → expand to one row per stakeholder per task
    stk_rows = []
    for _, row in gantt_df.iterrows():
        for stk in str(row["Stakeholders"]).split(","):
            stk = stk.strip()
            if stk:
                stk_rows.append({
                    "Stakeholder": stk,
                    "Task":        row["Task"],
                    "Phase":       row["Phase"],
                    "Start_Date":  row["Start_Date"],
                    "End_Date":    row["End_Date"],
                })
    stk_df = pd.DataFrame(stk_rows)

    if not stk_df.empty:
        # Engagement timeline per stakeholder
        stk_agg = stk_df.groupby("Stakeholder").agg(
            First_On=("Start_Date", "min"),
            Last_On=("End_Date", "max"),
            Tasks=("Task", "count"),
            Phases=("Phase", lambda x: ", ".join(sorted(set(x)))),
        ).reset_index().sort_values("First_On")

        fig_stk = px.timeline(
            stk_agg,
            x_start="First_On",
            x_end="Last_On",
            y="Stakeholder",
            color="Stakeholder",
            custom_data=["Tasks", "Phases"],
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
        fig_stk.update_traces(
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Active: %{base|%d %b %Y} → %{x|%d %b %Y}<br>"
                "Tasks involved: %{customdata[0]}<br>"
                "Phases: %{customdata[1]}<extra></extra>"
            )
        )
        fig_stk.update_yaxes(autorange="reversed")
        fig_stk.update_layout(
            **DARK,
            height=max(350, len(stk_agg) * 28 + 80),
            margin=dict(l=10, r=10, t=30, b=30),
            xaxis=dict(gridcolor="#21262d", tickformat="%b %Y"),
            yaxis=dict(gridcolor="#21262d"),
            showlegend=False,
        )
        add_reference_lines(fig_stk)
        st.plotly_chart(fig_stk, use_container_width=True)

        # Phase-level stakeholder matrix
        st.markdown("#### Stakeholder × Phase Matrix")
        stk_matrix = stk_df.groupby(["Stakeholder","Phase"]).size().unstack(fill_value=0)
        stk_matrix = stk_matrix.reindex(
            columns=[p for p in PHASE_COLORS.keys() if p in stk_matrix.columns]
        )
        fig_heat = px.imshow(
            stk_matrix,
            color_continuous_scale=[[0,"#0d1117"],[0.01,"#1c2a1c"],[1,"#3fb950"]],
            aspect="auto",
            labels=dict(color="Task Count"),
        )
        fig_heat.update_layout(
            **DARK, height=max(300, len(stk_matrix) * 22 + 80),
            margin=dict(l=10, r=10, t=30, b=30),
            xaxis=dict(side="top"),
            coloraxis_showscale=False,
        )
        fig_heat.update_traces(
            hovertemplate="<b>%{y}</b> in <b>%{x}</b><br>Tasks: %{z}<extra></extra>"
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # Stakeholder list per phase
        st.markdown("#### Stakeholders by Phase")
        for phase in phases_ordered:
            phase_stk = stk_df[stk_df["Phase"] == phase]["Stakeholder"].unique()
            tags = " ".join([f'<span class="stakeholder-tag">{s}</span>' for s in sorted(phase_stk)])
            st.markdown(
                f"<div class='phase-header'>{phase}</div>{tags}",
                unsafe_allow_html=True,
            )
        st.markdown("<br>", unsafe_allow_html=True)

# ── TAB 6: AI ANALYSIS ────────────────────────
with tab6:
    st.markdown("#### 🤖 AI-Powered Strategic Analysis")

    cp_names  = tasks[tasks["ID"].isin(critical_ids)]["Task"].tolist()
    risk_t    = tasks.copy()
    risk_t["Risk"] = (risk_t["Duration"] * risk_t["Uncertainty"]).round(1)
    top5_risk = risk_t.nlargest(5,"Risk")[["Task","Phase","Duration","Uncertainty","Risk"]].to_dict("records")

    project_context = {
        "opening_date":           str(opening_date),
        "implied_start_date":     project_start.strftime("%Y-%m-%d"),
        "days_to_opening":        days_to_opening,
        "days_to_start":          days_to_start,
        "base_duration_days":     int(base_duration),
        "buffer_days":            int(buffer_days),
        "mean_duration_days":     round(mean_days, 1),
        "p80_days":               round(p80, 1),
        "p95_days":               round(p95, 1),
        "on_time_probability_pct":round(on_time_pct, 1),
        "critical_path_tasks":    cp_names,
        "top_risk_tasks":         top5_risk,
        "top_sensitive_tasks":    sens_df.head(5)[["Task","Correlation"]].to_dict("records"),
        "total_cost_aed":         total_cost,
        "total_spent_aed":        total_actual,
        "budget_aed":             int(budget),
        "budget_status":          budget_status,
        "highest_uncertainty_phase": tasks.groupby("Phase")["Uncertainty"].mean().idxmax(),
    }

    if st.button("⚡ Generate AI Analysis", type="primary"):
        with st.spinner("Analyzing with Claude…"):
            try:
                client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
                msg = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1400,
                    system=(
                        "You are a senior retail fitout project manager with 20+ years in UAE/GCC. "
                        "The project is reverse-scheduled from a fixed opening date. "
                        "Give sharp, specific recommendations. Use markdown. "
                        "Focus on what risks could cause the opening date to be missed."
                    ),
                    messages=[{
                        "role": "user",
                        "content": (
                            "Analyze this reverse-scheduled fitout project:\n\n"
                            f"```json\n{json.dumps(project_context, indent=2)}\n```\n\n"
                            "Structure:\n"
                            "1. **Executive Summary** (2–3 sentences, opening date focus)\n"
                            "2. **Top 3 Risks to Opening Date** with specific mitigation\n"
                            "3. **Critical Path Strategy** — what to protect\n"
                            "4. **Budget Risk Assessment**\n"
                            "5. **Immediate Actions** — 3 things to do this week\n"
                        ),
                    }],
                )
                st.markdown(f'<div class="ai-box">{msg.content[0].text}</div>',
                            unsafe_allow_html=True)
            except Exception as exc:
                st.error(f"AI error: {exc}")

    st.markdown("#### 📌 Automated Insights")
    delay_spread  = p95 - mean_days
    worst_phase   = tasks.groupby("Phase")["Uncertainty"].mean().idxmax()
    top_sens_name = sens_df.iloc[0]["Task"]

    col_i1, col_i2 = st.columns(2)
    with col_i1:
        st.info(
            f"**Schedule Buffer:** {int(buffer_days)} days built into plan\n\n"
            f"**P95 Overrun Risk:** {delay_spread:.0f} days beyond mean\n\n"
            f"**Most Sensitive Task:** `{top_sens_name}` — delays here cascade to opening date\n\n"
            f"**Critical Path:** {len(critical_ids)} tasks — 1 day slip = 1 day delay on opening"
        )
    with col_i2:
        bud_line = "✅ Under budget" if budget_status == "under" else "⚠️ Over budget"
        ot_line  = ("healthy ✅" if on_time_pct >= 70 else
                    "needs attention ⚠️" if on_time_pct >= 40 else
                    "CRITICAL — re-plan required 🔴")
        start_warn = (f"⚠️ Project should have started {abs(days_to_start)} days ago!"
                      if days_to_start < 0 else
                      f"✅ Project starts in {days_to_start} days")
        st.warning(
            f"**{start_warn}**\n\n"
            f"**Highest Uncertainty Phase:** {worst_phase}\n\n"
            f"**Budget:** {bud_line} by AED {abs(int(budget) - total_needed):,.0f}\n\n"
            f"**On-Time Probability:** {on_time_pct:.0f}% — {ot_line}"
        )

# ─────────────────────────────────────────────
# PROFESSIONAL MANAGEMENT REPORT
# ─────────────────────────────────────────────
st.divider()
st.markdown("### 🖨️ Management Report")
st.caption("Professional report ready for leadership. Download → open in Chrome → Ctrl+P to print or Save as PDF.")

if st.button("📄 Generate Management Report", type="secondary"):
    with st.spinner("Building report…"):

        PRINT_STYLE = dict(
            paper_bgcolor="white", plot_bgcolor="#f9fafb",
            font=dict(color="#1f2937", family="Arial, sans-serif"),
        )
        PRINT_PHASE = {
            "Phase 0": "#2563eb", "Design": "#7c3aed",
            "Procurement": "#d97706", "Authority": "#ea580c",
            "Pre-Execution": "#0891b2", "Execution": "#16a34a",
            "Testing": "#15803d", "Opening": "#dc2626",
        }

        def fig_to_html(fig, h=320):
            fig.update_layout(height=h)
            return fig.to_html(full_html=False, include_plotlyjs=False)

        def add_print_lines(fig):
            fig.add_shape(type="line", x0=today_str, x1=today_str, y0=0, y1=1,
                          xref="x", yref="paper",
                          line=dict(color="#2563eb", dash="dot", width=1.5))
            fig.add_annotation(x=today_str, y=1, xref="x", yref="paper",
                               text="Today", showarrow=False,
                               font=dict(color="#2563eb", size=10), yanchor="bottom")
            fig.add_shape(type="line", x0=opening_str, x1=opening_str, y0=0, y1=1,
                          xref="x", yref="paper",
                          line=dict(color="#dc2626", dash="dash", width=2))
            fig.add_annotation(x=opening_str, y=0.92, xref="x", yref="paper",
                               text="Opening", showarrow=False,
                               font=dict(color="#dc2626", size=10), yanchor="bottom")
            return fig

        # RAG status
        if on_time_pct >= 70:
            rag_color = "#059669"; rag_bg = "#ecfdf5"; rag_text = "ON TRACK"
            rag_icon = "✅"
            rag_desc = "Project is progressing as planned. Monitor critical path closely."
        elif on_time_pct >= 40:
            rag_color = "#d97706"; rag_bg = "#fffbeb"; rag_text = "AT RISK"
            rag_icon = "⚠️"
            rag_desc = "Schedule pressure detected. Immediate management attention required."
        else:
            rag_color = "#dc2626"; rag_bg = "#fef2f2"; rag_text = "CRITICAL"
            rag_icon = "🔴"
            rag_desc = "Opening date is at serious risk. Escalation and recovery plan needed."

        budget_rag  = "#059669" if budget_status == "under" else "#dc2626"
        budget_icon = "✅" if budget_status == "under" else "🔴"
        start_rag   = "#dc2626" if days_to_start < 0 else "#059669"
        start_icon  = "⚠️" if days_to_start < 0 else "✅"
        start_note  = (f"Should have started {abs(days_to_start)} days ago"
                       if days_to_start < 0 else f"Starts in {days_to_start} days")
        budget_variance_note = ("Under" if budget_status == "under" else "Over")

        # Master Gantt (light)
        fig_gantt_p = px.timeline(
            phase_agg, x_start="Start_Date", x_end="End_Date",
            y="Phase", color="Phase", color_discrete_map=PRINT_PHASE,
        )
        fig_gantt_p.update_yaxes(autorange="reversed")
        fig_gantt_p.update_layout(
            **PRINT_STYLE, height=300, showlegend=False,
            margin=dict(l=10, r=10, t=20, b=20),
            xaxis=dict(gridcolor="#e5e7eb", tickformat="%b '%y", linecolor="#d1d5db"),
            yaxis=dict(gridcolor="#e5e7eb", linecolor="#d1d5db"),
        )
        add_print_lines(fig_gantt_p)

        # Budget vs Actual (light)
        bva_p = tasks[tasks["Cost_AED"] > 0].sort_values("Cost_AED", ascending=False).head(12)
        fig_cost_p = go.Figure()
        fig_cost_p.add_trace(go.Bar(name="Budget", x=bva_p["Task"], y=bva_p["Cost_AED"],
                                    marker_color="#93c5fd"))
        fig_cost_p.add_trace(go.Bar(name="Actual", x=bva_p["Task"], y=bva_p["Actual_Cost_AED"],
                                    marker_color="#1d4ed8"))
        fig_cost_p.update_layout(
            **PRINT_STYLE, barmode="group", height=280,
            margin=dict(l=10, r=10, t=10, b=60),
            xaxis=dict(gridcolor="#e5e7eb", tickangle=-30, linecolor="#d1d5db"),
            yaxis=dict(gridcolor="#e5e7eb", title="AED", linecolor="#d1d5db"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )

        # Phase detail charts
        # Fix: zoom each chart to its own date range so short tasks are visible
        phase_detail_html = ""
        for phase in phases_ordered:
            pt_p = gantt_df[gantt_df["Phase"] == phase].copy()
            if pt_p.empty:
                continue
            pt_p = pt_p.copy()
            pt_p["Color"] = pt_p["Critical"].apply(
                lambda c: "#dc2626" if c else PRINT_PHASE.get(phase, "#2563eb")
            )
            pt_p["Label"] = pt_p.apply(
                lambda r: f"* {r['Task']}" if r["Critical"] else r["Task"], axis=1
            )

            # Zoom x-axis to THIS phase only (+ small padding)
            ph_x_min = pt_p["Start_Date"].min() - pd.Timedelta(days=5)
            ph_x_max = pt_p["End_Date"].max()   + pd.Timedelta(days=5)

            fp = px.timeline(pt_p, x_start="Start_Date", x_end="End_Date",
                             y="Label", color="Color", color_discrete_map="identity")
            fp.update_yaxes(autorange="reversed")
            fp.update_layout(
                **PRINT_STYLE,
                height=max(180, len(pt_p) * 38 + 70),
                # Large left margin so full task names are never clipped
                margin=dict(l=220, r=20, t=15, b=15),
                showlegend=False,
                xaxis=dict(
                    gridcolor="#e5e7eb", tickformat="%d %b '%y",
                    linecolor="#d1d5db",
                    range=[ph_x_min, ph_x_max],   # KEY: zoom to phase window
                    tickangle=-30,
                ),
                yaxis=dict(
                    gridcolor="#e5e7eb", linecolor="#d1d5db",
                    tickfont=dict(size=11, color="#111827"),
                    automargin=True,
                ),
            )

            # Only draw Today / Opening lines if they fall inside this phase window
            today_ts   = pd.Timestamp(today_str)
            opening_ts = pd.Timestamp(opening_str)
            if ph_x_min <= today_ts <= ph_x_max:
                fp.add_shape(type="line", x0=today_str, x1=today_str, y0=0, y1=1,
                             xref="x", yref="paper",
                             line=dict(color="#2563eb", dash="dot", width=1.5))
                fp.add_annotation(x=today_str, y=1, xref="x", yref="paper",
                                  text="Today", showarrow=False,
                                  font=dict(color="#2563eb", size=9), yanchor="bottom")
            if ph_x_min <= opening_ts <= ph_x_max:
                fp.add_shape(type="line", x0=opening_str, x1=opening_str, y0=0, y1=1,
                             xref="x", yref="paper",
                             line=dict(color="#dc2626", dash="dash", width=2))
                fp.add_annotation(x=opening_str, y=0.92, xref="x", yref="paper",
                                  text="Opening", showarrow=False,
                                  font=dict(color="#dc2626", size=9), yanchor="bottom")
            cost_sum   = int(tasks[tasks["Phase"] == phase]["Cost_AED"].sum())
            actual_sum = int(tasks[tasks["Phase"] == phase]["Actual_Cost_AED"].sum())
            stk_list   = ", ".join(sorted(set(
                s.strip()
                for row in tasks[tasks["Phase"] == phase]["Stakeholders"].tolist()
                for s in str(row).split(",") if s.strip()
            )))
            ph_color = PRINT_PHASE.get(phase, "#2563eb")
            chart_h  = max(160, len(pt_p) * 32 + 60)
            phase_detail_html += (
                f'<div class="phase-block">'
                f'<div class="phase-title" style="border-left:4px solid {ph_color}">'
                f'<span>{phase}</span>'
                f'<span class="phase-meta">{len(pt_p)} tasks &nbsp;&middot;&nbsp; '
                f'Budget AED {cost_sum:,.0f} &nbsp;&middot;&nbsp; '
                f'Spent AED {actual_sum:,.0f}</span></div>'
                + fig_to_html(fp, chart_h)
                + f'<div class="stk-row"><b>Stakeholders:</b> {stk_list}</div>'
                + '</div>'
            )

        # Top risk rows
        risk_t2 = tasks.copy()
        risk_t2["Risk_Score"] = risk_t2["Duration"] * risk_t2["Uncertainty"]
        top_risks_rows = ""
        for _, r in risk_t2.sort_values("Risk_Score", ascending=False).head(8).iterrows():
            cp_flag = "Yes — Critical Path" if r["ID"] in critical_ids else "No"
            impact  = "High" if r["Risk_Score"] > 50 else "Medium" if r["Risk_Score"] > 20 else "Low"
            row_cls = "risk-high" if impact == "High" else "risk-med" if impact == "Medium" else "risk-low"
            corr_row = sens_df[sens_df["ID"] == r["ID"]]
            sensitivity = f"{corr_row.iloc[0]['Correlation']:.2f}" if not corr_row.empty else "-"
            top_risks_rows += (
                f'<tr class="{row_cls}">'
                f"<td>{r['Phase']}</td>"
                f"<td><b>{r['Task']}</b></td>"
                f'<td style="text-align:center">{int(r["Duration"])}d</td>'
                f'<td style="text-align:center">&plusmn;{r["Uncertainty"]}d</td>'
                f'<td style="text-align:center;font-weight:700">{impact}</td>'
                f'<td style="text-align:center">{sensitivity}</td>'
                f"<td>{cp_flag}</td>"
                "</tr>"
            )

        # Actions
        actions_html = ""
        for i, (_, row) in enumerate(sens_df.head(5).iterrows(), 1):
            urgency = "Immediate" if i <= 2 else "This week" if i == 3 else "Monitor"
            u_color = "#dc2626" if urgency == "Immediate" else "#d97706" if urgency == "This week" else "#059669"
            cp_note = " (Critical Path)" if row["ID"] in critical_ids else ""
            actions_html += (
                "<tr>"
                f'<td><span style="color:{u_color};font-weight:700">{urgency}</span></td>'
                f"<td><b>{row['Task']}</b>{cp_note}</td>"
                f"<td>Sensitivity {row['Correlation']:.2f} — delays here directly impact the opening date. "
                "Confirm schedule and escalate if at risk.</td>"
                "</tr>"
            )

        # Cost table
        cost_rows_html = ""
        for _, r in tasks[tasks["Cost_AED"] > 0].sort_values("Cost_AED", ascending=False).iterrows():
            variance = int(r["Cost_AED"]) - int(r["Actual_Cost_AED"])
            pct = (r["Actual_Cost_AED"] / r["Cost_AED"] * 100) if r["Cost_AED"] > 0 else 0
            status = ("Not started" if r["Actual_Cost_AED"] == 0
                      else "Over budget" if r["Actual_Cost_AED"] > r["Cost_AED"]
                      else "On track")
            s_color = "#6b7280" if status == "Not started" else "#dc2626" if status == "Over budget" else "#059669"
            v_color = "#dc2626" if variance < 0 else "#059669"
            cost_rows_html += (
                "<tr>"
                f"<td>{r['Phase']}</td>"
                f"<td><b>{r['Task']}</b></td>"
                f'<td style="text-align:right">AED {int(r["Cost_AED"]):,.0f}</td>'
                f'<td style="text-align:right">AED {int(r["Actual_Cost_AED"]):,.0f}</td>'
                f'<td style="text-align:right;color:{v_color}">AED {variance:,.0f}</td>'
                f'<td style="text-align:center">{pct:.0f}%</td>'
                f'<td style="color:{s_color};font-weight:600">{status}</td>'
                "</tr>"
            )
        spent_pct = (total_actual / total_cost * 100) if total_cost > 0 else 0
        var_total = total_cost - total_actual
        v_total_color = "#dc2626" if var_total < 0 else "#059669"

        # Stakeholder table
        stk_summary_rows = ""
        all_stk: dict = {}
        for _, row in gantt_df.iterrows():
            for s in str(row["Stakeholders"]).split(","):
                s = s.strip()
                if not s:
                    continue
                if s not in all_stk:
                    all_stk[s] = {"phases": set(), "first": row["Start_Date"], "last": row["End_Date"]}
                all_stk[s]["phases"].add(row["Phase"])
                all_stk[s]["first"] = min(all_stk[s]["first"], row["Start_Date"])
                all_stk[s]["last"]  = max(all_stk[s]["last"],  row["End_Date"])
        for stk, info in sorted(all_stk.items(), key=lambda x: x[1]["first"]):
            stk_summary_rows += (
                "<tr>"
                f"<td><b>{stk}</b></td>"
                f"<td>{info['first'].strftime('%d %b %Y')}</td>"
                f"<td>{info['last'].strftime('%d %b %Y')}</td>"
                f"<td>{', '.join(sorted(info['phases']))}</td>"
                "</tr>"
            )

        # Critical path flow
        cp_flow_html = " &rarr; ".join(
            f'<span class="cp-task">{t}</span>' for t in cp_names
        )

        report_date = pd.Timestamp.today().strftime("%d %B %Y")
        ot_risk_label = (
            "Low risk to opening date" if on_time_pct >= 70
            else "Moderate risk — action required" if on_time_pct >= 40
            else "High risk — immediate escalation needed"
        )
        ot_kpi_color = "#059669" if on_time_pct >= 70 else "#d97706" if on_time_pct >= 40 else "#dc2626"
        ot_box_class = "green" if on_time_pct >= 70 else "amber" if on_time_pct >= 40 else "red"
        start_box_class = "red" if days_to_start < 0 else "green"
        start_status_note = ("Start date has passed — schedule at risk"
                             if days_to_start < 0 else "On schedule to begin on time")
        budget_box_class = "green" if budget_status == "under" else "red"
        top_sens_name = sens_df.iloc[0]["Task"]
        top_sens_corr = sens_df.iloc[0]["Correlation"]
        highest_unc_phase = tasks.groupby("Phase")["Uncertainty"].mean().idxmax()

        # Phase summary table rows
        phase_summary_rows = ""
        for _, row in phase_agg.iterrows():
            s_date = row["Start_Date"].strftime("%d %b %Y") if hasattr(row["Start_Date"], "strftime") else str(row["Start_Date"])
            e_date = row["End_Date"].strftime("%d %b %Y") if hasattr(row["End_Date"], "strftime") else str(row["End_Date"])
            tc = int(phase_task_count.get(row["Phase"], 0))
            bc = int(phase_cost.get(row["Phase"], 0))
            phase_summary_rows += (
                "<tr>"
                f"<td><b>{row['Phase']}</b></td>"
                f"<td>{s_date}</td><td>{e_date}</td>"
                f'<td style="text-align:center">{row["Duration_Days"]} days</td>'
                f'<td style="text-align:center">{tc}</td>'
                f'<td style="text-align:right">AED {bc:,.0f}</td>'
                "</tr>"
            )

        # ─── Build HTML ─────────────────────────
        html = (
            "<!DOCTYPE html>\n<html lang='en'>\n<head>\n"
            "<meta charset='UTF-8'>\n"
            f"<title>Fitout Project — Management Report — {pd.to_datetime(opening_date).strftime('%d %b %Y')}</title>\n"
            "<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>\n"
            "<style>\n"
            "*{box-sizing:border-box;margin:0;padding:0}\n"
            "body{background:#fff;color:#1f2937;font-family:Arial,sans-serif;font-size:13px;line-height:1.5}\n"
            ".cover{background:linear-gradient(135deg,#1e3a8a 0%,#1d4ed8 100%);color:#fff;padding:2.5rem 2.5rem 2rem;position:relative}\n"
            ".cover h1{font-size:1.9rem;font-weight:700;margin-bottom:0.25rem;letter-spacing:-0.5px}\n"
            ".cover .sub{font-size:0.85rem;opacity:0.8}\n"
            ".cover .meta{margin-top:1rem;font-size:0.78rem;opacity:0.7}\n"
            ".opening-badge{position:absolute;right:2.5rem;top:2rem;"
            "background:rgba(255,255,255,0.15);border:2px solid rgba(255,255,255,0.35);"
            "border-radius:12px;padding:1rem 1.5rem;text-align:center}\n"
            ".ob-label{font-size:0.65rem;text-transform:uppercase;letter-spacing:1px;opacity:0.8}\n"
            ".ob-date{font-size:1.45rem;font-weight:700;margin-top:0.2rem}\n"
            ".ob-days{font-size:0.75rem;opacity:0.8;margin-top:0.1rem}\n"
            ".status-banner{display:flex;align-items:center;gap:1rem;padding:0.85rem 2rem;"
            "border-left-width:0}\n"
            ".status-badge{font-weight:700;font-size:0.85rem;padding:0.3rem 1rem;border-radius:20px;white-space:nowrap;color:#fff}\n"
            ".status-desc{font-size:0.85rem;color:#374151}\n"
            ".section{padding:1.4rem 2.5rem;border-bottom:1px solid #f3f4f6}\n"
            ".section h2{font-size:0.9rem;font-weight:700;color:#111827;text-transform:uppercase;"
            "letter-spacing:0.5px;margin-bottom:1rem;padding-bottom:0.4rem;border-bottom:2px solid #e5e7eb}\n"
            ".section h3{font-size:0.82rem;font-weight:700;color:#374151;margin:1rem 0 0.5rem}\n"
            ".kpi-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:0.9rem;margin-bottom:1rem}\n"
            ".kpi-box{border:1px solid #e5e7eb;border-radius:8px;padding:0.9rem 1.1rem}\n"
            ".kpi-box.green{border-left:4px solid #059669}\n"
            ".kpi-box.amber{border-left:4px solid #d97706}\n"
            ".kpi-box.red{border-left:4px solid #dc2626}\n"
            ".kpi-box.blue{border-left:4px solid #2563eb}\n"
            ".kpi-label{font-size:0.62rem;text-transform:uppercase;letter-spacing:1px;color:#6b7280;margin-bottom:0.25rem}\n"
            ".kpi-value{font-size:1.5rem;font-weight:700;color:#111827;line-height:1}\n"
            ".kpi-sub{font-size:0.7rem;color:#6b7280;margin-top:0.25rem}\n"
            ".kpi-note{font-size:0.7rem;margin-top:0.25rem;font-weight:600}\n"
            ".exec-box{background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;"
            "padding:1rem 1.2rem;font-size:0.83rem;color:#374151;line-height:1.75}\n"
            ".cp-flow{display:flex;flex-wrap:wrap;gap:0.3rem;align-items:center;margin-top:0.5rem}\n"
            ".cp-task{background:#fef2f2;border:1px solid #fecaca;color:#991b1b;"
            "padding:3px 10px;border-radius:4px;font-size:0.75rem;font-weight:600}\n"
            "table{width:100%;border-collapse:collapse;font-size:0.78rem;margin-top:0.4rem}\n"
            "th{background:#f9fafb;color:#374151;font-weight:700;padding:7px 10px;"
            "text-align:left;border-bottom:2px solid #e5e7eb;font-size:0.68rem;"
            "text-transform:uppercase;letter-spacing:0.3px}\n"
            "td{padding:6px 10px;border-bottom:1px solid #f3f4f6;vertical-align:top}\n"
            "tr:hover td{background:#f9fafb}\n"
            ".risk-high td{background:#fef2f2}\n"
            ".risk-med td{background:#fffbeb}\n"
            ".risk-low td{background:#f0fdf4}\n"
            ".phase-block{margin-bottom:1.5rem}\n"
            ".phase-title{display:flex;justify-content:space-between;align-items:center;"
            "padding:0.5rem 0.8rem;background:#f9fafb;margin-bottom:0.4rem;border-radius:4px}\n"
            ".phase-title span:first-child{font-weight:700;font-size:0.83rem;color:#111827}\n"
            ".phase-meta{font-size:0.7rem;color:#6b7280}\n"
            ".stk-row{font-size:0.72rem;color:#6b7280;margin-top:0.3rem;padding:0 0.3rem}\n"
            ".footer{padding:0.8rem 2.5rem;background:#f9fafb;font-size:0.7rem;color:#9ca3af;text-align:center}\n"
            ".page-break{page-break-before:always}\n"
            "@media print{\n"
            "@page{margin:1cm;size:A4 landscape}\n"
            "body{font-size:11px}\n"
            ".cover{-webkit-print-color-adjust:exact;print-color-adjust:exact}\n"
            ".status-banner{-webkit-print-color-adjust:exact;print-color-adjust:exact}\n"
            ".kpi-box{-webkit-print-color-adjust:exact;print-color-adjust:exact}\n"
            ".risk-high td,.risk-med td,.risk-low td{-webkit-print-color-adjust:exact;print-color-adjust:exact}\n"
            "}\n"
            "</style>\n</head>\n<body>\n"

            # COVER
            "<div class='cover'>\n"
            "<div class='opening-badge'>\n"
            "<div class='ob-label'>Target Opening</div>\n"
            f"<div class='ob-date'>{pd.to_datetime(opening_date).strftime('%d %b %Y')}</div>\n"
            f"<div class='ob-days'>{days_to_opening} days from today</div>\n"
            "</div>\n"
            "<h1>Fitout Project</h1>\n"
            "<div class='sub'>Management Report &nbsp;&middot;&nbsp; Confidential</div>\n"
            f"<div class='meta'>Prepared: {report_date} &nbsp;&middot;&nbsp; "
            f"Implied Start: {project_start.strftime('%d %b %Y')} &nbsp;&middot;&nbsp; "
            f"Base Duration: {int(base_duration)} days + {int(buffer_days)}d buffer &nbsp;&middot;&nbsp; "
            f"Monte Carlo: {runs:,} simulations</div>\n"
            "</div>\n"

            # STATUS BANNER
            f"<div class='status-banner' style='background:{rag_bg};border-left:6px solid {rag_color}'>\n"
            f"<div class='status-badge' style='background:{rag_color}'>{rag_icon} {rag_text}</div>\n"
            f"<div class='status-desc'>{rag_desc} &nbsp;&middot;&nbsp; "
            f"<b>On-Time Probability: {on_time_pct:.0f}%</b> &nbsp;&middot;&nbsp; "
            f"{len(critical_ids)} tasks on critical path</div>\n"
            "</div>\n"

            # PAGE 1: EXECUTIVE DASHBOARD
            "<div class='section'>\n"
            "<h2>Executive Dashboard</h2>\n"
            "<div class='kpi-grid'>\n"

            f"<div class='kpi-box {ot_box_class}'>"
            "<div class='kpi-label'>On-Time Probability</div>"
            f"<div class='kpi-value'>{on_time_pct:.0f}%</div>"
            f"<div class='kpi-sub'>Mean: {mean_days:.0f}d &nbsp;&middot;&nbsp; P80: {p80:.0f}d &nbsp;&middot;&nbsp; P95: {p95:.0f}d</div>"
            f"<div class='kpi-note' style='color:{ot_kpi_color}'>{ot_risk_label}</div>"
            "</div>\n"

            "<div class='kpi-box blue'>"
            "<div class='kpi-label'>Target Opening Date</div>"
            f"<div class='kpi-value' style='font-size:1.15rem'>{pd.to_datetime(opening_date).strftime('%d %b %Y')}</div>"
            f"<div class='kpi-sub'>{days_to_opening} days from today</div>"
            f"<div class='kpi-note' style='color:#2563eb'>Implied start: {project_start.strftime('%d %b %Y')}</div>"
            "</div>\n"

            f"<div class='kpi-box {start_box_class}'>"
            "<div class='kpi-label'>Project Start Status</div>"
            f"<div class='kpi-value' style='font-size:1.05rem'>{start_icon} {start_note}</div>"
            f"<div class='kpi-sub'>Implied start: {project_start.strftime('%d %b %Y')}</div>"
            f"<div class='kpi-note' style='color:{start_rag}'>{start_status_note}</div>"
            "</div>\n"

            f"<div class='kpi-box {budget_box_class}'>"
            "<div class='kpi-label'>Budget (with Contingency)</div>"
            f"<div class='kpi-value' style='font-size:1.15rem'>AED {total_needed/1e6:.2f}M</div>"
            f"<div class='kpi-sub'>Base AED {total_cost/1e6:.2f}M + {contingency_pct}% contingency</div>"
            f"<div class='kpi-note' style='color:{budget_rag}'>{budget_icon} {budget_variance_note} approved budget by AED {abs(int(budget)-total_needed):,.0f}</div>"
            "</div>\n"

            "<div class='kpi-box blue'>"
            "<div class='kpi-label'>Expenditure to Date</div>"
            f"<div class='kpi-value' style='font-size:1.15rem'>AED {total_actual/1e6:.2f}M</div>"
            f"<div class='kpi-sub'>of AED {total_cost/1e6:.2f}M total budget</div>"
            f"<div class='kpi-note' style='color:#2563eb'>{spent_pct:.1f}% of base budget spent</div>"
            "</div>\n"

            "<div class='kpi-box amber'>"
            "<div class='kpi-label'>Critical Path Tasks</div>"
            f"<div class='kpi-value'>{len(critical_ids)}</div>"
            "<div class='kpi-sub'>Any slip = direct delay to opening</div>"
            f"<div class='kpi-note' style='color:#d97706'>Highest risk: {top_sens_name}</div>"
            "</div>\n"

            "</div>\n"  # end kpi-grid

            "<h3>Executive Summary</h3>\n"
            "<div class='exec-box'>"
            f"This project is targeting an opening date of <b>{pd.to_datetime(opening_date).strftime('%d %b %Y')}</b>, "
            f"with an implied start date of <b>{project_start.strftime('%d %b %Y')}</b> based on a "
            f"<b>{int(base_duration)}-day base schedule</b> plus <b>{int(buffer_days)} days buffer</b>. "
            f"Monte Carlo simulation across <b>{runs:,} scenarios</b> shows an on-time probability of "
            f"<b>{on_time_pct:.0f}%</b>, with a P80 completion of <b>{p80:.0f} days</b> and P95 of <b>{p95:.0f} days</b>. "
            f"The project has <b>{len(critical_ids)} tasks on the critical path</b> — any delay to these directly pushes the opening date. "
            f"The highest schedule sensitivity is <b>{top_sens_name}</b> (sensitivity score: {top_sens_corr:.2f}). "
            f"The highest uncertainty phase is <b>{highest_unc_phase}</b>. "
            f"Budget stands at <b>AED {total_needed/1e6:.2f}M</b> including contingency, "
            f"which is <b>{'within' if budget_status=='under' else 'over'}</b> the approved budget of <b>AED {int(budget)/1e6:.2f}M</b>."
            "</div>\n"

            "<h3 style='margin-top:1rem'>Critical Path</h3>\n"
            f"<div class='cp-flow'>{cp_flow_html}</div>\n"
            "</div>\n"  # end section

            # PAGE 2: MASTER TIMELINE
            "<div class='page-break'></div>\n"
            "<div class='section'>\n"
            "<h2>Master Project Timeline</h2>\n"
            + fig_to_html(fig_gantt_p, 300)
            + "\n<h3 style='margin-top:1rem'>Phase Schedule Summary</h3>\n"
            "<table><tr>"
            "<th>Phase</th><th>Start</th><th>End</th>"
            "<th style='text-align:center'>Days</th>"
            "<th style='text-align:center'>Tasks</th>"
            "<th style='text-align:right'>Budget (AED)</th>"
            "</tr>\n"
            + phase_summary_rows
            + "</table>\n</div>\n"

            # PAGE 3: PHASE BREAKDOWN
            "<div class='page-break'></div>\n"
            "<div class='section'>\n"
            "<h2>Phase Breakdown &mdash; Task Detail</h2>\n"
            + phase_detail_html
            + "</div>\n"

            # PAGE 4: SCHEDULE RISK
            "<div class='page-break'></div>\n"
            "<div class='section'>\n"
            "<h2>Schedule Risk Assessment</h2>\n"
            "<p style='color:#6b7280;font-size:0.78rem;margin-bottom:0.8rem'>"
            "Tasks ranked by risk score (duration &times; uncertainty). "
            "Sensitivity shows impact on total project duration &mdash; higher = more impact on opening date.</p>\n"
            "<table><tr>"
            "<th>Phase</th><th>Task</th>"
            "<th style='text-align:center'>Duration</th>"
            "<th style='text-align:center'>Uncertainty</th>"
            "<th style='text-align:center'>Impact</th>"
            "<th style='text-align:center'>Sensitivity</th>"
            "<th>Critical Path</th></tr>\n"
            + top_risks_rows
            + "</table>\n"
            "<h3 style='margin-top:1.5rem'>Recommended Actions</h3>\n"
            "<table><tr><th>Priority</th><th>Task</th><th>Action Required</th></tr>\n"
            + actions_html
            + "</table>\n</div>\n"

            # PAGE 5: FINANCIAL
            "<div class='page-break'></div>\n"
            "<div class='section'>\n"
            "<h2>Financial Summary</h2>\n"
            + fig_to_html(fig_cost_p, 280)
            + "\n<h3 style='margin-top:1rem'>Cost Tracker by Task</h3>\n"
            "<table><tr>"
            "<th>Phase</th><th>Task</th>"
            "<th style='text-align:right'>Budget</th>"
            "<th style='text-align:right'>Actual Spent</th>"
            "<th style='text-align:right'>Variance</th>"
            "<th style='text-align:center'>% Spent</th>"
            "<th>Status</th></tr>\n"
            + cost_rows_html
            + f"<tr style='font-weight:700;background:#f3f4f6'>"
            f"<td colspan='2'>TOTAL</td>"
            f"<td style='text-align:right'>AED {total_cost:,.0f}</td>"
            f"<td style='text-align:right'>AED {total_actual:,.0f}</td>"
            f"<td style='text-align:right;color:{v_total_color}'>AED {var_total:,.0f}</td>"
            f"<td style='text-align:center'>{spent_pct:.1f}%</td>"
            f"<td></td></tr>\n"
            "</table>\n</div>\n"

            # PAGE 6: STAKEHOLDERS
            "<div class='page-break'></div>\n"
            "<div class='section'>\n"
            "<h2>Stakeholder Engagement</h2>\n"
            "<table><tr>"
            "<th>Stakeholder</th>"
            "<th>Engagement From</th>"
            "<th>Engagement To</th>"
            "<th>Phases Involved</th></tr>\n"
            + stk_summary_rows
            + "</table>\n</div>\n"

            # FOOTER
            "<div class='footer'>"
            f"Fitout Optimization Engine &nbsp;&middot;&nbsp; Generated {report_date} "
            "&nbsp;&middot;&nbsp; Confidential &mdash; Management Use Only"
            "</div>\n"

            "</body>\n</html>"
        )

        st.download_button(
            label="⬇️ Download Management Report",
            data=html,
            file_name=f"fitout_report_{pd.to_datetime(opening_date).strftime('%Y%m%d')}.html",
            mime="text/html",
            type="primary",
        )
        st.success("✅ Report ready — open in Chrome or Edge, then Ctrl+P → Save as PDF for the cleanest output.")
