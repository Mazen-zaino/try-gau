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
                margin=dict(l=10, r=10, t=20, b=20),
                xaxis=dict(gridcolor="#21262d", tickformat="%d %b"),
                yaxis=dict(gridcolor="#21262d"),
                showlegend=False,
            )
            add_reference_lines(fig_ph)
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
# PRINT / EXPORT HTML REPORT
# ─────────────────────────────────────────────
st.divider()
st.markdown("### 🖨️ Export & Print")
st.caption("Generates a self-contained HTML report. Open in browser → Ctrl+P to print.")

if st.button("📄 Generate Printable Report", type="secondary"):
    with st.spinner("Building report…"):

        def fig_html(fig, height=400):
            fig.update_layout(height=height)
            return fig.to_html(full_html=False, include_plotlyjs=False)

        # Re-build key figures for the report
        fig_m2 = px.timeline(phase_agg, x_start="Start_Date", x_end="End_Date",
                              y="Phase", color="Phase", color_discrete_map=PHASE_COLORS)
        fig_m2.update_yaxes(autorange="reversed")
        fig_m2.update_layout(**DARK, height=320, margin=dict(l=10,r=10,t=20,b=20),
                              showlegend=False,
                              xaxis=dict(gridcolor="#21262d", tickformat="%b %Y"),
                              yaxis=dict(gridcolor="#21262d"))
        add_reference_lines(fig_m2)

        # Cost chart
        fig_bva2 = go.Figure()
        bva2 = tasks[tasks["Cost_AED"] > 0].sort_values("Cost_AED", ascending=False).head(15)
        fig_bva2.add_trace(go.Bar(name="Budget", x=bva2["Task"], y=bva2["Cost_AED"], marker_color="#58a6ff"))
        fig_bva2.add_trace(go.Bar(name="Actual", x=bva2["Task"], y=bva2["Actual_Cost_AED"], marker_color="#3fb950"))
        fig_bva2.update_layout(**DARK, barmode="group", height=300,
                                margin=dict(l=10,r=10,t=20,b=10),
                                xaxis=dict(gridcolor="#21262d", tickangle=-35),
                                yaxis=dict(gridcolor="#21262d"),
                                legend=dict(orientation="h"))

        # Risk register HTML table
        rr = risk_reg[["Phase","Task","Duration","Risk_Score","Impact","Critical","P95 Finish","Stakeholders"]]
        rr_html = rr.sort_values("Risk_Score", ascending=False).to_html(index=False, border=0)

        # Cost tracker HTML table
        ct_html = cost_track.to_html(index=False, border=0)

        # Phase detail charts HTML
        phase_charts_html = ""
        for phase in phases_ordered:
            pt2 = gantt_df[gantt_df["Phase"] == phase].copy()
            if pt2.empty:
                continue
            pt2["Bar_Color"] = pt2["Critical"].apply(
                lambda c: "#f85149" if c else PHASE_COLORS.get(phase, "#58a6ff")
            )
            pt2["Label"] = pt2.apply(
                lambda r: f"🔴 {r['Task']}" if r["Critical"] else r["Task"], axis=1
            )
            f = px.timeline(pt2, x_start="Start_Date", x_end="End_Date",
                            y="Label", color="Bar_Color", color_discrete_map="identity")
            f.update_yaxes(autorange="reversed")
            f.update_layout(**DARK, height=max(180, len(pt2)*35+60),
                            margin=dict(l=10,r=10,t=20,b=20), showlegend=False,
                            xaxis=dict(gridcolor="#21262d", tickformat="%d %b"),
                            yaxis=dict(gridcolor="#21262d"))
            add_reference_lines(f)
            phase_charts_html += f"""
            <h3 style="color:#c9d1d9;margin-top:2rem">{phase}</h3>
            {fig_html(f, max(180, len(pt2)*35+60))}
            """

        html_report = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Fitout Project Report — {pd.to_datetime(opening_date).strftime('%d %b %Y')}</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;700&family=DM+Mono&display=swap');
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0d1117; color: #c9d1d9; font-family: 'DM Sans', sans-serif; padding: 2rem; }}
  h1 {{ color: #f0f6fc; font-size: 1.6rem; margin-bottom: 0.3rem; }}
  h2 {{ color: #c9d1d9; font-size: 1.1rem; margin: 1.5rem 0 0.5rem; border-bottom: 1px solid #21262d; padding-bottom: 0.4rem; }}
  h3 {{ color: #c9d1d9; font-size: 0.95rem; }}
  .subtitle {{ color: #8b949e; font-size: 0.85rem; margin-bottom: 1.5rem; }}
  .kpi-row {{ display: grid; grid-template-columns: repeat(4,1fr); gap: 1rem; margin: 1.5rem 0; }}
  .kpi {{ background: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 1rem; }}
  .kpi-label {{ font-size: 0.65rem; text-transform: uppercase; letter-spacing: 1.5px; color: #8b949e; }}
  .kpi-value {{ font-size: 1.4rem; font-weight: 700; color: #f0f6fc; font-family: 'DM Mono', monospace; margin-top: 0.2rem; }}
  .kpi-sub {{ font-size: 0.7rem; color: #8b949e; margin-top: 0.2rem; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.78rem; margin-top: 0.5rem; }}
  th {{ background: #161b22; color: #8b949e; text-align: left; padding: 6px 8px; font-weight: 600; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.5px; }}
  td {{ padding: 6px 8px; border-bottom: 1px solid #21262d; color: #c9d1d9; }}
  tr:hover td {{ background: #161b22; }}
  .page-break {{ page-break-before: always; }}
  @media print {{
    body {{ background: #fff; color: #111; padding: 0.5cm; }}
    h1,h2,h3 {{ color: #111 !important; }}
    .kpi {{ background: #f5f5f5; border-color: #ddd; }}
    .kpi-label,.kpi-sub {{ color: #666 !important; }}
    .kpi-value {{ color: #111 !important; }}
    table {{ font-size: 0.7rem; }}
    th {{ background: #eee; color: #444; }}
    td {{ color: #111; border-color: #ddd; }}
    @page {{ margin: 1cm; size: A4 landscape; }}
  }}
</style>
</head>
<body>
<h1>🏗️ Fitout Project Report</h1>
<div class="subtitle">Generated {pd.Timestamp.today().strftime('%d %b %Y %H:%M')} &nbsp;·&nbsp;
Target Opening: <strong>{pd.to_datetime(opening_date).strftime('%d %b %Y')}</strong> &nbsp;·&nbsp;
Implied Start: <strong>{project_start.strftime('%d %b %Y')}</strong></div>

<div class="kpi-row">
  <div class="kpi">
    <div class="kpi-label">Opening Date</div>
    <div class="kpi-value">{pd.to_datetime(opening_date).strftime('%d %b %Y')}</div>
    <div class="kpi-sub">{days_to_opening} days from today</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">On-Time Probability</div>
    <div class="kpi-value">{on_time_pct:.0f}%</div>
    <div class="kpi-sub">P80: {p80:.0f}d · P95: {p95:.0f}d</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Base Duration</div>
    <div class="kpi-value">{int(base_duration)}d</div>
    <div class="kpi-sub">+ {int(buffer_days)}d buffer</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Budget w/ Contingency</div>
    <div class="kpi-value">AED {total_needed/1e6:.2f}M</div>
    <div class="kpi-sub">Spent: AED {total_actual/1e6:.2f}M</div>
  </div>
</div>

<h2>Master Gantt — Phase Overview</h2>
{fig_html(fig_m2, 320)}

<div class="page-break"></div>
<h2>Phase Breakdown — Task Detail</h2>
{phase_charts_html}

<div class="page-break"></div>
<h2>Risk Register</h2>
{rr_html}

<h2 style="margin-top:2rem">Cost Tracker</h2>
{ct_html}

<div class="page-break"></div>
<h2>Budget vs Actual</h2>
{fig_html(fig_bva2, 300)}

<div style="margin-top:3rem;color:#8b949e;font-size:0.72rem">
Critical path: {' → '.join(cp_names)}
</div>
</body>
</html>"""

        st.download_button(
            label="⬇️ Download HTML Report",
            data=html_report,
            file_name=f"fitout_report_{pd.to_datetime(opening_date).strftime('%Y%m%d')}.html",
            mime="text/html",
            type="primary",
        )
        st.success("Report ready! Download and open in Chrome/Edge, then Ctrl+P to print.")
