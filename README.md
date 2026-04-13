# 🏗️ Fitout Optimization Engine

> An AI-powered retail fitout project management tool built for the GCC market. Plan, schedule, and de-risk fitout projects using reverse scheduling, Monte Carlo simulation, critical path analysis, and Claude AI-generated recommendations — all in one dashboard.

---

## 🚀 Live App

👉 **[https://fitout-optimizer-mavnycpwxmbopbyqun4hug.streamlit.app/](https://fitout-optimizer-mavnycpwxmbopbyqun4hug.streamlit.app/)**

---

## ✨ What It Does

### 🔄 Reverse Scheduling
Set your **target opening date** and the engine works backwards — automatically calculating the implied project start date, phase windows, and task dates. Add a buffer of days between the last task and opening for contingency. Every chart, KPI, and simulation is anchored to your opening date, not a fixed start.

### 📊 Master Gantt — Phase Overview
A clean, high-level Gantt showing one bar per phase across the full project timeline. Includes:
- Today reference line
- Opening date marker
- Phase summary table (start, end, days, task count, budget)
- Critical path displayed as a dependency flow

### 🔍 Phase Breakdown — Task Detail
Expand any phase to see a **zoomed task-level Gantt** for that phase only. Each chart is scoped to the phase's own date window so even 1-day tasks are clearly visible — no more hairline bars. Shows task name, duration, dates, stakeholders, and critical path flag per task.

### ⚠️ Risk Analysis
- **S-Curve** — cumulative probability of finishing before your opening date, with Mean, P80, P95 markers
- **Tornado Chart** — ranks tasks by Pearson correlation with total duration. Shows which tasks are actually driving your schedule risk, not just which ones are long
- **Risk Register** — full table with risk score, impact rating (High/Medium/Low), critical path flag, and P95 finish date per task

### 💰 Cost Tracking
- **Budget vs Actual** grouped bar chart — enter actual costs as you go, see over/under runs instantly
- **Cost Tracker table** — every task with Budget, Actual, Variance, % Spent, and status (On track / Over budget / Not started)
- **Phase donut chart** — budget distribution by phase
- Contingency modelling and variance against approved budget

### 👥 Stakeholder Engagement
Replace headcount tracking with stakeholder visibility:
- **Engagement Timeline** — Gantt showing when each stakeholder is active across the project
- **Stakeholder × Phase Matrix** — heatmap of who is involved in what
- **Phase-level stakeholder tags** — quick reference per phase

### 🤖 AI Analysis (Claude-Powered)
- One-click **AI strategic analysis** using Claude Sonnet — tailored to your live project data
- Covers: executive summary, top 3 risks to opening date, critical path strategy, budget risk, and immediate actions
- Always-visible automated insights: variability, sensitivity, budget position, start date status

### ⏱️ Duration Sliders
Every task has a duration slider in the sidebar. Drag any slider and every chart, KPI, simulation, and date recalculates instantly — no manual updates needed.

### 💾 Save & Load
- **Save JSON** — export your full task register (durations, costs, actuals, stakeholders) to a file
- **Import JSON** — reload a saved project in one click. All sliders, costs, and stakeholder data restore automatically

### 🖨️ Management Report
Generate a professional, print-ready HTML report with:
- Blue branded cover with opening date badge
- RAG status banner (On Track / At Risk / Critical) in plain English
- 6 executive KPI cards (on-time probability, opening date, start status, budget, spend, critical path)
- Auto-written executive summary paragraph
- Master Gantt (light theme, print-friendly)
- Per-phase CSS Gantt charts with minimum bar widths — every task clearly visible even at 1 day duration
- Top risk table (colour-coded rows)
- Recommended actions (Immediate / This Week / Monitor)
- Budget vs Actual chart
- Full cost tracker table with totals row
- Stakeholder engagement table
- Open in Chrome → Ctrl+P → Save as PDF for the cleanest output

---

## 🖥️ Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/your-username/fitout-optimizer.git
cd fitout-optimizer
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add your Anthropic API key**

Create `.streamlit/secrets.toml`:
```toml
ANTHROPIC_API_KEY = "sk-ant-xxxxxxxxxxxxxxxx"
```
Get your key at [console.anthropic.com](https://console.anthropic.com).

**4. Run**
```bash
streamlit run HUK.py
```

---

## ☁️ Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → connect GitHub
3. Select repo, set main file to `HUK.py`
4. Under **Advanced settings → Secrets** paste:
```toml
ANTHROPIC_API_KEY = "sk-ant-xxxxxxxxxxxxxxxx"
```
5. Click **Deploy**

> ⚠️ Make sure `requirements.txt` is named in **all lowercase** — Streamlit Cloud is case-sensitive.

---

## 📁 Project Structure

```
fitout-optimizer/
├── HUK.py                # Main Streamlit app
├── requirements.txt      # Python dependencies (lowercase!)
├── README.md             # This file
└── LICENSE               # Custom non-commercial license
```

---

## 📦 Requirements

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.26.0
plotly>=5.18.0
anthropic>=0.25.0
python-dotenv>=1.0.0
```

---

## 🔑 Environment Variables

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Anthropic API key for AI analysis tab |

---

## 🛠️ How It Works

1. **Set your opening date** in the sidebar — all schedules compute backwards from here
2. **Adjust durations** with sliders — every chart updates instantly
3. **Sort tasks** by ID or logical dependency flow (topological sort)
4. **Edit the task register** — fine-tune costs, uncertainty, stakeholders, dependencies
5. **Enter actual costs** as work progresses — cost tracker updates automatically
6. **Run Monte Carlo** — each task duration sampled from a normal distribution, producing a probability curve for your opening date
7. **Generate AI analysis** — Claude reviews your live project data and returns strategic recommendations
8. **Export the report** — download a management-ready HTML file and print to PDF

---

## 🧠 Technical Notes

- **Reverse scheduling**: computed as `project_start = opening_date − base_duration − buffer_days`
- **Critical path**: forward/backward pass algorithm (CPM) with float calculation
- **Monte Carlo**: up to 2,000 simulations using NumPy normal distribution per task
- **Sensitivity (Tornado)**: Pearson correlation between each task's end distribution and total project duration
- **Phase Gantt bars**: minimum 3% display width enforced in CSS so 1-day tasks are always visible
- **Save/Load**: full session state via JSON — includes all edits, actuals, and stakeholder data

---

## ⚠️ Important

- Never commit your API key to GitHub. Always use `.streamlit/secrets.toml`
- The AI analysis button requires an active Anthropic API key with available credits
- For best print output: Chrome or Edge → Ctrl+P → Landscape → No margins → Save as PDF

---

## 📄 License

This project is licensed under a **Custom Non-Commercial License** — see [LICENSE](LICENSE) for full terms.

Free for personal, educational, and internal use. **Commercial use requires written permission from the author.**
Contact via [the live app](https://fitout-optimizer-mavnycpwxmbopbyqun4hug.streamlit.app/) or GitHub.

---

## 🙏 Built With

- [Streamlit](https://streamlit.io) — app framework
- [Plotly](https://plotly.com) — interactive charts
- [Anthropic Claude](https://anthropic.com) — AI analysis
- [Pandas](https://pandas.pydata.org) / [NumPy](https://numpy.org) — data engine
