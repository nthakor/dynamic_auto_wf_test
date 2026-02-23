import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import BytesIO

st.set_page_config(page_title="Rule Waterfall Analyzer", layout="wide")

# ================================================================
# Helper Functions
# ================================================================

@st.cache_data
def load_parquet(file_bytes):
    return pd.read_parquet(BytesIO(file_bytes))


@st.cache_data
def generate_dummy_data(n_records=100_000, n_rules=20):
    np.random.seed(42)
    dates = pd.date_range("2022-01-01", "2024-12-31", periods=n_records)
    data = {
        "application_date": np.sort(np.random.choice(dates, n_records, replace=True)),
        "region": np.random.choice(["East", "West", "North", "South"], n_records),
        "product": np.random.choice(
            ["Personal Loan", "Auto Loan", "Mortgage", "Credit Card"], n_records
        ),
        "channel": np.random.choice(["Online", "Branch", "Partner"], n_records),
    }
    base_risk = np.random.beta(2, 5, n_records)
    for i in range(1, n_rules + 1):
        threshold = np.random.uniform(0.3, 0.8)
        noise = np.random.normal(0, 0.15, n_records)
        data[f"rule_{i:03d}"] = ((base_risk + noise) > threshold).astype(int)

    rule_sum = sum(data[f"rule_{i:03d}"] for i in range(1, n_rules + 1))
    bad_prob = 1 / (1 + np.exp(-(rule_sum / n_rules * 4 - 2)))
    data["bad_flag"] = (np.random.random(n_records) < bad_prob).astype(int)
    return pd.DataFrame(data)


def detect_column_types(df):
    """Auto-detect date, binary-rule, categorical, and numeric columns."""
    date_cols, rule_cols, cat_cols, num_cols = [], [], [], []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_cols.append(col)
            continue
        vals = set(df[col].dropna().unique())
        if vals and vals.issubset({0, 1, 0.0, 1.0, True, False}):
            rule_cols.append(col)
            continue
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(
            df[col]
        ):
            try:
                sample = df[col].dropna().head(50)
                if len(sample) > 0:
                    pd.to_datetime(sample)
                    date_cols.append(col)
                    continue
            except (ValueError, TypeError, OverflowError):
                pass
            cat_cols.append(col)
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            num_cols.append(col)
            continue
        cat_cols.append(col)
    return date_cols, rule_cols, cat_cols, num_cols


def filter_by_date(df, date_col, years, quarters):
    if not date_col or (not years and not quarters):
        return df
    dt = pd.to_datetime(df[date_col])
    mask = pd.Series(True, index=df.index)
    if years:
        mask &= dt.dt.year.isin(years)
    if quarters:
        mask &= dt.dt.quarter.isin(quarters)
    return df[mask]


def filter_by_categories(df, cat_filters):
    mask = pd.Series(True, index=df.index)
    for col, vals in cat_filters.items():
        if vals:
            mask &= df[col].isin(vals)
    return df[mask]


def compute_waterfall(df, ordered_groups, weight_col=None):
    w = df[weight_col].values.astype(float) if weight_col else np.ones(len(df))
    total = w.sum()
    remaining = np.ones(len(df), dtype=bool)
    results = []
    for name, rules in ordered_groups:
        if not rules:
            continue
        hit = df[rules].fillna(0).values.astype(bool).any(axis=1) & remaining
        cnt = (hit * w).sum()
        results.append(
            {
                "group": name,
                "count": cnt,
                "pct": cnt / total * 100 if total else 0,
            }
        )
        remaining &= ~hit
    approved = (remaining * w).sum()
    results.append(
        {
            "group": "Approved",
            "count": approved,
            "pct": approved / total * 100 if total else 0,
        }
    )
    return results, total


def compute_bad_rates(df, ordered_groups, bad_col, weight_col=None):
    w = df[weight_col].values.astype(float) if weight_col else np.ones(len(df))
    b = df[bad_col].fillna(0).values.astype(float)
    remaining = np.ones(len(df), dtype=bool)
    results = []
    for name, rules in ordered_groups:
        if not rules:
            continue
        hit = df[rules].fillna(0).values.astype(bool).any(axis=1) & remaining
        seg_total = (hit * w).sum()
        seg_bads = (hit * b).sum()
        results.append(
            {
                "group": name,
                "total": seg_total,
                "bads": seg_bads,
                "bad_rate": seg_bads / seg_total * 100 if seg_total else 0,
            }
        )
        remaining &= ~hit
    seg_total = (remaining * w).sum()
    seg_bads = (remaining * b).sum()
    results.append(
        {
            "group": "Approved",
            "total": seg_total,
            "bads": seg_bads,
            "bad_rate": seg_bads / seg_total * 100 if seg_total else 0,
        }
    )
    return results


def plot_waterfall(results, total_pop, mode="absolute"):
    n_decline = len(results) - 1  # everything except Approved
    labels = (
        ["Total Population"]
        + [r["group"] for r in results[:n_decline]]
        + ["Approved"]
    )
    measures = ["absolute"] + ["relative"] * n_decline + ["total"]

    if mode == "absolute":
        values = [total_pop] + [-r["count"] for r in results[:n_decline]] + [0]
        text = (
            [f"{total_pop:,.0f}"]
            + [f"{r['count']:,.0f}" for r in results[:n_decline]]
            + [f"{results[-1]['count']:,.0f}"]
        )
        y_title = "Applications"
    else:
        values = [100.0] + [-r["pct"] for r in results[:n_decline]] + [0]
        text = (
            ["100%"]
            + [f"{r['pct']:.1f}%" for r in results[:n_decline]]
            + [f"{results[-1]['pct']:.1f}%"]
        )
        y_title = "Percentage (%)"

    fig = go.Figure(
        go.Waterfall(
            measure=measures,
            x=labels,
            y=values,
            text=text,
            textposition="outside",
            connector={"line": {"color": "rgb(63,63,63)"}},
            decreasing={"marker": {"color": "#EF553B"}},
            increasing={"marker": {"color": "#00CC96"}},
            totals={"marker": {"color": "#636EFA"}},
        )
    )
    fig.update_layout(
        title=f"Decline Waterfall ({mode.title()})",
        yaxis_title=y_title,
        showlegend=False,
        height=500,
    )
    return fig


def plot_bad_rates(results):
    df_br = pd.DataFrame(results)
    colors = ["#EF553B"] * (len(df_br) - 1) + ["#00CC96"]
    fig = go.Figure(
        go.Bar(
            x=df_br["group"],
            y=df_br["bad_rate"],
            text=[f"{r:.2f}%" for r in df_br["bad_rate"]],
            textposition="outside",
            marker_color=colors,
        )
    )
    fig.update_layout(
        title="Bad Rate by Waterfall Segment",
        yaxis_title="Bad Rate (%)",
        height=500,
    )
    return fig


# ================================================================
# Session State
# ================================================================
if "groups" not in st.session_state:
    st.session_state.groups = []

# ================================================================
# Sidebar
# ================================================================
with st.sidebar:
    st.header("Data Source")
    data_source = st.radio("Choose data source:", ["Upload Parquet", "Use Dummy Data"])

    if data_source == "Upload Parquet":
        uploaded = st.file_uploader("Upload a Parquet file", type=["parquet"])
        if uploaded:
            st.session_state["df"] = load_parquet(uploaded.getvalue())
            st.success(
                f"Loaded {len(st.session_state['df']):,} rows, "
                f"{len(st.session_state['df'].columns)} columns"
            )
    else:
        c1, c2 = st.columns(2)
        n_records = c1.number_input("Records", 1_000, 3_000_000, 100_000, step=10_000)
        n_rules = c2.number_input("Rules", 5, 200, 20)
        if st.button("Generate Dummy Data", use_container_width=True):
            st.session_state["df"] = generate_dummy_data(int(n_records), int(n_rules))
            st.session_state.groups = []
            st.success(f"Generated {int(n_records):,} rows, {int(n_rules)} rules")

df = st.session_state.get("df")

# Column configuration & filters (only when data is loaded)
date_cols = []
rule_cols = []
cat_cols = []
num_cols = []
weight_col = None
bad_col = None
cat_filters = {}

if df is not None:
    auto_date, auto_rule, auto_cat, auto_num = detect_column_types(df)
    all_cols = list(df.columns)

    with st.sidebar:
        st.divider()
        st.header("Column Configuration")

        with st.expander("Date Columns", expanded=False):
            date_cols = st.multiselect(
                "Select date columns", all_cols, default=auto_date, key="cfg_date"
            )
        with st.expander("Rule Columns (binary 0/1)", expanded=False):
            rule_cols = st.multiselect(
                "Select rule columns", all_cols, default=auto_rule, key="cfg_rule"
            )
        with st.expander("Categorical Columns", expanded=False):
            cat_cols = st.multiselect(
                "Select categorical columns",
                all_cols,
                default=auto_cat,
                key="cfg_cat",
            )
        with st.expander("Aggregation & Target", expanded=True):
            weight_options = ["(None)"] + auto_num
            wc = st.selectbox(
                "Weight/Count column (aggregated data)", weight_options, key="cfg_wt"
            )
            weight_col = None if wc == "(None)" else wc

            bad_options = ["(None)"] + [
                c for c in auto_rule + auto_num if c in all_cols
            ]
            bc = st.selectbox("Bad/Target column", bad_options, key="cfg_bad")
            bad_col = None if bc == "(None)" else bc

        st.divider()
        st.header("Categorical Filters")
        if cat_cols:
            for col in cat_cols:
                unique_vals = sorted(df[col].dropna().unique().tolist())
                sel = st.multiselect(f"{col}", unique_vals, key=f"filt_{col}")
                if sel:
                    cat_filters[col] = sel
        else:
            st.caption("No categorical columns configured.")

# ================================================================
# Main Area
# ================================================================
st.title("Rule Waterfall Analyzer")

if df is None:
    st.info("Upload a Parquet file or generate dummy data from the sidebar to begin.")
    st.stop()

# Data preview
with st.expander("Data Preview", expanded=False):
    st.dataframe(df.head(100), use_container_width=True, height=300)
    st.caption(f"{len(df):,} rows x {len(df.columns)} columns")

tab_group, tab_wf, tab_br = st.tabs(
    ["Rule Grouping", "Waterfall Analysis", "Bad Rate Analysis"]
)

# ----------------------------------------------------------------
# Tab 1: Rule Grouping
# ----------------------------------------------------------------
with tab_group:
    st.subheader("Configure Rule Groups")
    st.caption(
        "Create groups, assign rules, and order them. Applications are evaluated "
        "top-to-bottom; once declined by a group, they are excluded from later groups."
    )

    # Compute assigned / unassigned
    assigned = set()
    for g in st.session_state.groups:
        assigned.update(g["rules"])
    unassigned = [r for r in rule_cols if r not in assigned]

    st.info(
        f"**{len(unassigned)}** unassigned / **{len(rule_cols)}** total rule columns"
    )

    # Action buttons
    bcol1, bcol2, bcol3 = st.columns(3)
    with bcol1:
        if st.button("Add Group", use_container_width=True):
            n = len(st.session_state.groups) + 1
            st.session_state.groups.append({"name": f"Group {n}", "rules": []})
            st.rerun()
    with bcol2:
        if st.button("Auto-group (1 rule each)", use_container_width=True):
            st.session_state.groups = [
                {"name": r, "rules": [r]} for r in rule_cols
            ]
            st.rerun()
    with bcol3:
        if st.button("Clear All Groups", use_container_width=True):
            st.session_state.groups = []
            st.rerun()

    # Render groups
    to_delete = []
    for i, group in enumerate(st.session_state.groups):
        with st.container(border=True):
            cols = st.columns([4, 1, 1, 1, 1])
            with cols[0]:
                new_name = st.text_input(
                    "Name",
                    group["name"],
                    key=f"gname_{i}",
                    label_visibility="collapsed",
                )
                if new_name != group["name"]:
                    st.session_state.groups[i]["name"] = new_name
            with cols[1]:
                st.caption(f"#{i+1}")
            with cols[2]:
                if st.button("Up", key=f"up_{i}", disabled=(i == 0)):
                    gs = st.session_state.groups
                    gs[i], gs[i - 1] = gs[i - 1], gs[i]
                    st.rerun()
            with cols[3]:
                if st.button("Dn", key=f"dn_{i}", disabled=(i == len(st.session_state.groups) - 1)):
                    gs = st.session_state.groups
                    gs[i], gs[i + 1] = gs[i + 1], gs[i]
                    st.rerun()
            with cols[4]:
                if st.button("Del", key=f"del_{i}"):
                    to_delete.append(i)

            available = sorted(set(unassigned + group["rules"]))
            selected = st.multiselect(
                "Assign rules",
                available,
                default=sorted(group["rules"]),
                key=f"grules_{i}",
                label_visibility="collapsed",
            )
            if set(selected) != set(group["rules"]):
                st.session_state.groups[i]["rules"] = selected
                st.rerun()

    if to_delete:
        for idx in sorted(to_delete, reverse=True):
            st.session_state.groups.pop(idx)
        st.rerun()


# Build ordered groups list
ordered_groups = [
    (g["name"], g["rules"]) for g in st.session_state.groups if g["rules"]
]

# ----------------------------------------------------------------
# Tab 2: Waterfall
# ----------------------------------------------------------------
with tab_wf:
    if not ordered_groups:
        st.warning("Configure at least one rule group with rules assigned.")
        st.stop()

    st.subheader("Waterfall Date Filter")
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        wf_date_col_options = ["(None)"] + date_cols
        wf_dc = st.selectbox("Date column", wf_date_col_options, key="wf_datecol")
        wf_date_col = None if wf_dc == "(None)" else wf_dc
    wf_years, wf_quarters = [], []
    if wf_date_col:
        dt_s = pd.to_datetime(df[wf_date_col])
        with fc2:
            wf_years = st.multiselect(
                "Year(s)",
                sorted(dt_s.dt.year.dropna().unique()),
                key="wf_yr",
            )
        with fc3:
            wf_quarters = st.multiselect("Quarter(s)", [1, 2, 3, 4], key="wf_qt")

    st.divider()

    # Filter data
    df_wf = filter_by_date(df, wf_date_col, wf_years, wf_quarters)
    df_wf = filter_by_categories(df_wf, cat_filters)

    if len(df_wf) == 0:
        st.warning("No data matches the current filters.")
    else:
        results_wf, total_pop = compute_waterfall(df_wf, ordered_groups, weight_col)
        approved = results_wf[-1]
        total_declined = total_pop - approved["count"]

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Population", f"{total_pop:,.0f}")
        m2.metric("Total Declined", f"{total_declined:,.0f}")
        m3.metric("Approval Rate", f"{approved['pct']:.1f}%")

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                plot_waterfall(results_wf, total_pop, "absolute"),
                use_container_width=True,
            )
        with c2:
            st.plotly_chart(
                plot_waterfall(results_wf, total_pop, "percentage"),
                use_container_width=True,
            )

        st.subheader("Detail Table")
        detail = pd.DataFrame(results_wf)
        detail.columns = ["Group", "Declined Count", "Declined %"]
        detail["Declined Count"] = detail["Declined Count"].map(lambda x: f"{x:,.0f}")
        detail["Declined %"] = detail["Declined %"].map(lambda x: f"{x:.2f}%")
        st.dataframe(detail, use_container_width=True, hide_index=True)

# ----------------------------------------------------------------
# Tab 3: Bad Rate
# ----------------------------------------------------------------
with tab_br:
    if not ordered_groups:
        st.warning("Configure at least one rule group with rules assigned.")
        st.stop()
    if not bad_col:
        st.warning("Select a Bad/Target column in the sidebar Column Configuration.")
        st.stop()

    st.subheader("Bad Rate Date Filter")
    bc1, bc2, bc3 = st.columns(3)
    with bc1:
        br_date_col_options = ["(None)"] + date_cols
        br_dc = st.selectbox("Date column", br_date_col_options, key="br_datecol")
        br_date_col = None if br_dc == "(None)" else br_dc
    br_years, br_quarters = [], []
    if br_date_col:
        dt_s = pd.to_datetime(df[br_date_col])
        with bc2:
            br_years = st.multiselect(
                "Year(s)",
                sorted(dt_s.dt.year.dropna().unique()),
                key="br_yr",
            )
        with bc3:
            br_quarters = st.multiselect("Quarter(s)", [1, 2, 3, 4], key="br_qt")

    st.divider()

    df_br = filter_by_date(df, br_date_col, br_years, br_quarters)
    df_br = filter_by_categories(df_br, cat_filters)

    if len(df_br) == 0:
        st.warning("No data matches the current filters.")
    else:
        results_br = compute_bad_rates(df_br, ordered_groups, bad_col, weight_col)

        overall_total = sum(r["total"] for r in results_br)
        overall_bads = sum(r["bads"] for r in results_br)
        overall_rate = overall_bads / overall_total * 100 if overall_total else 0

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Records", f"{overall_total:,.0f}")
        m2.metric("Total Bads", f"{overall_bads:,.0f}")
        m3.metric("Overall Bad Rate", f"{overall_rate:.2f}%")

        st.plotly_chart(plot_bad_rates(results_br), use_container_width=True)

        st.subheader("Detail Table")
        br_detail = pd.DataFrame(results_br)
        br_detail.columns = ["Group", "Total", "Bads", "Bad Rate (%)"]
        br_detail["Total"] = br_detail["Total"].map(lambda x: f"{x:,.0f}")
        br_detail["Bads"] = br_detail["Bads"].map(lambda x: f"{x:,.0f}")
        br_detail["Bad Rate (%)"] = br_detail["Bad Rate (%)"].map(
            lambda x: f"{x:.2f}%"
        )
        st.dataframe(br_detail, use_container_width=True, hide_index=True)
