"""PAMAP2 Human Activity Recognition - interactive demo dashboard.

    python -m streamlit run app/streamlit_app.py

A read-only demo over the existing project. It reads result files, the frozen split and
(optionally) exported model artifacts; it never trains a model, never runs CV and never writes
to results/. Every number comes from the repository's own files.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:          # `streamlit run app/...` puts only app/ on the path
    sys.path.insert(0, str(REPO_ROOT))

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from app import data_layer as dl
from har.config import ACTIVITY_NAMES, FEATURE_SETS, LABEL_ORDER

st.set_page_config(page_title="HAR · PAMAP2", page_icon="🧭", layout="wide",
                   initial_sidebar_state="expanded")

# ============================================================================ constants

PAGES = ["🏠 Overview", "📊 Models", "🎯 Confusion Matrix", "🤖 Predict", "📚 About"]

INK, INK_2, GRID, AXIS, ACCENT = "#e9eaec", "#c3c2b7", "#262a31", "#383835", "#3987e5"

# Status -> (badge colour, chart colour, short meaning). Only statuses with real results appear.
STATUS = {
    "FULL": ("blue", "#3987e5", "Standard project experiment (all training rows, full test split)."),
    "SUBSAMPLE": ("orange", "#d95926", "Uses a subset of the available data. Not directly comparable to FULL."),
    "QUICK": ("violet", "#9a7fd1", "Resource-constrained run (reduced compute, no tuning). Not comparable to FULL."),
    "UNVERIFIED": ("yellow", "#c98500", "Row counts could not be checked against the frozen split."),
}

MODEL_INFO = {
    "logreg": ("Logistic Regression", "Linear classification baseline."),
    "svm_rbf": ("SVM (RBF)", "Finds a separating decision boundary using an RBF kernel."),
    "decision_tree": ("Decision Tree", "Recursively splits the feature space into decision regions."),
    "random_forest": ("Random Forest", "Combines multiple decision trees to improve robustness."),
    "adaboost": ("AdaBoost", "Sequentially combines weak learners, each focusing on earlier mistakes."),
    "mlp": ("MLP", "Neural network that learns nonlinear relationships."),
}

FS_LABEL = {"reduced": "Reduced", "full": "Full"}
FS_DESC = {"reduced": "11 features: heart rate + hand IMU",
           "full": "31 features: heart rate + hand, chest and ankle IMUs"}

CSS = """
<style>
.block-container {padding-top: 2.2rem; padding-bottom: 1.5rem; max-width: 1400px;}
h1 {font-size: 1.85rem !important; padding: 0 0 .1rem 0 !important; letter-spacing: .01em;}
h3 {font-size: 1.1rem !important; padding: .2rem 0 .3rem 0 !important;}
[data-testid="stMetricValue"] {font-size: 1.55rem;}
[data-testid="stSidebarContent"] h3 {padding-top: 0 !important;}
</style>
"""

# ============================================================================ helpers


def pct(v, digits: int = 2) -> str:
    return "—" if v is None or pd.isna(v) else f"{100 * float(v):.{digits}f}%"


def secs(v) -> str:
    if v is None or pd.isna(v):
        return "—"
    v = float(v)
    return f"{v:,.1f} s" if v < 120 else f"{v / 60:,.1f} min"


def count(v) -> str:
    return "—" if v is None or pd.isna(v) else f"{int(v):,}"


def act_name(activity_id: int) -> str:
    n = ACTIVITY_NAMES.get(int(activity_id), str(activity_id))
    return n[:1].upper() + n[1:]


def badge(status: str) -> str:
    return f":{STATUS.get(status, ('gray',))[0]}-badge[{status}]"


def model_name(key: str, fallback: str = "") -> str:
    return MODEL_INFO.get(key, (fallback or key, ""))[0]


def config_label(r) -> str:
    return f"{model_name(r['model'], r['model_name'])} · {FS_LABEL.get(r['feature_set'], r['feature_set'])}"


def status_legend():
    st.caption("  \n".join(f"{badge(s)} {STATUS[s][2]}" for s in ("FULL", "SUBSAMPLE", "QUICK")))


def show_chart(chart: alt.Chart, alt_text: str):
    chart = (chart.configure(background="transparent", font="system-ui, -apple-system, Segoe UI, sans-serif")
             .configure_view(stroke=None)
             .configure_axis(labelColor=INK_2, titleColor=INK_2, gridColor=GRID, domainColor=AXIS,
                             tickColor=AXIS, labelFontSize=12, titleFontSize=12, titleFontWeight="normal")
             .configure_legend(labelColor=INK_2, titleColor=INK_2, labelFontSize=12, orient="top"))
    st.altair_chart(chart, width="stretch", theme=None, alt=alt_text)


# ============================================================================ cached loaders

@st.cache_data(show_spinner=False)
def get_results():
    """(completed results table, error). NOT RUN / MALFORMED rows are dropped here, once."""
    try:
        table, _ = dl.load_comparison()
        return dl.completed_results(table), None
    except Exception as e:                       # never crash the whole app on bad results
        return None, f"{type(e).__name__}: {e}"


@st.cache_data(show_spinner=False)
def get_stats():
    return dl.dataset_stats()


@st.cache_data(show_spinner=False)
def get_confusion(stem: str):
    return dl.load_confusion(stem)


@st.cache_data(show_spinner=False)
def get_confusion_stems():
    return [s for s in dl.list_confusion_stems() if dl.load_confusion(s) is not None]


@st.cache_resource(show_spinner=False)
def get_test_samples():
    return dl.load_test_samples()


@st.cache_resource(show_spinner=False, max_entries=4)
def get_model(path: str, feature_set: str, model_key: str):
    return dl.load_demo_model(path, feature_set, model_key)


@st.cache_data(show_spinner=False, ttl=60)
def get_demo_models():
    return dl.find_demo_models()


def require_results() -> pd.DataFrame:
    table, err = get_results()
    if err:
        st.error(f"Could not read the result files: {err}")
        st.stop()
    if table.empty:
        st.info("No completed experiment results found in `results/metrics/`.")
        st.stop()
    return table


# ============================================================================ pages

def page_overview():
    s = get_stats()
    table = require_results()
    st.title("HUMAN ACTIVITY RECOGNITION")
    st.markdown("##### PAMAP2 Sensor-Based Activity Classification")

    tiles = [("Subjects", count(s["n_subjects"]), None), ("Activities", count(s["n_activities"]), None),
             ("Training samples", count(s["n_train"]), None), ("Test samples", count(s["n_test"]), None),
             ("Feature sets", str(len(FEATURE_SETS)), " · ".join(f"{FS_LABEL[k]}: {v}" for k, v in FS_DESC.items())),
             ("Models", str(table["model"].nunique()), "Model families with completed results")]
    for col, (label, value, hlp) in zip(st.columns(6), tiles):
        col.metric(label, value, help=hlp, border=True)

    left, right = st.columns([1.7, 1], gap="medium")
    with left:
        st.markdown("### Completed results · test accuracy")
        accuracy_chart(table)
    with right:
        full = table[table["status"] == "FULL"]
        if not full.empty:
            best = full.loc[full["test_accuracy"].idxmax()]
            st.markdown("### Best FULL result")
            with st.container(border=True):
                st.markdown(f"**{config_label(best)} features** {badge('FULL')}")
                a, b = st.columns(2)
                a.metric("Accuracy", pct(best["test_accuracy"]))
                b.metric("Macro F1", pct(best["test_macro_f1"]))
        st.markdown("### About")
        st.markdown("Wearable IMUs on the hand, chest and ankle plus a heart-rate monitor record 12 everyday "
                    "activities. Each model classifies a single sensor reading into one activity.")

    st.markdown("**Pipeline** &nbsp; " + " → ".join(
        f":blue-badge[{x}]" for x in ("DATA", "CLEAN", "FEATURES", "TRAIN", "EVALUATE", "PREDICT")))


def accuracy_chart(table: pd.DataFrame):
    d = table.sort_values("test_accuracy", ascending=False).copy()
    d["Config"] = [config_label(r) + ("" if r["status"] == "FULL" else f" ({r['status']})")
                   for _, r in d.iterrows()]
    d["Label"] = d["test_accuracy"].map(pct)
    present = [s for s in STATUS if s in set(d["status"])]
    y = alt.Y("Config:N", sort=list(d["Config"]), title=None, axis=alt.Axis(labelLimit=260))
    bars = alt.Chart(d).mark_bar(cornerRadiusEnd=3, height=16).encode(
        y=y, x=alt.X("test_accuracy:Q", scale=alt.Scale(domain=[0, 1.12]), title=None,
                     axis=alt.Axis(format="%", values=[0, .25, .5, .75, 1])),
        color=alt.Color("status:N", title=None,
                        scale=alt.Scale(domain=present, range=[STATUS[p][1] for p in present])),
        tooltip=["Config", alt.Tooltip("status:N", title="Status"), alt.Tooltip("Label:N", title="Accuracy")])
    text = alt.Chart(d).mark_text(align="left", dx=4, color=INK_2, fontSize=12).encode(
        y=y, x="test_accuracy:Q", text="Label:N")
    show_chart((bars + text).properties(height=26 * len(d) + 10), "Test accuracy per completed configuration")


def page_models():
    table = require_results()
    st.title("📊 Models")
    st.caption("Completed experiments only, read from `results/metrics/`. Configurations without results are omitted.")

    left, right = st.columns([1.75, 1], gap="medium")
    with left:
        d = pd.DataFrame({
            "Model": [model_name(m, n) for m, n in zip(table["model"], table["model_name"])],
            "Feature Set": table["feature_set"].map(FS_LABEL),
            "Accuracy": table["test_accuracy"] * 100, "Macro F1": table["test_macro_f1"] * 100,
            "Weighted F1": table["test_weighted_f1"] * 100, "Status": table["status"]})
        d = d.sort_values("Accuracy", ascending=False)
        sty = (d.style.format({c: "{:.2f}%" for c in ("Accuracy", "Macro F1", "Weighted F1")}, na_rep="—")
               .map(lambda v: f"color: {STATUS.get(v, ('', INK))[1]}; font-weight: 700", subset=["Status"]))
        st.dataframe(sty, hide_index=True, width="stretch", height=35 * (len(d) + 1) + 3,
                     alt="Comparison of completed experiments",
                     column_config={"Model": st.column_config.TextColumn(width=150),
                                    "Feature Set": st.column_config.TextColumn(width=80),
                                    **{c: st.column_config.NumberColumn(width=82)
                                       for c in ("Accuracy", "Macro F1", "Weighted F1")},
                                    "Status": st.column_config.TextColumn(width=95)})
        status_legend()

    with right:
        opts = list(d.index)
        rf = table.index[(table["model"] == "random_forest") & (table["feature_set"] == "full")]
        idx = st.selectbox("Selected model", opts, index=opts.index(rf[0]) if len(rf) else 0,
                           format_func=lambda i: f"{config_label(table.loc[i])} — {table.loc[i, 'status']}",
                           key="models_sel")
        r = table.loc[idx]
        with st.container(border=True):
            st.markdown(f"**{model_name(r['model'], r['model_name'])}** {badge(r['status'])}")
            st.caption(MODEL_INFO.get(r["model"], ("", ""))[1])
            a, b, c = st.columns(3)
            a.metric("Accuracy", pct(r["test_accuracy"]))
            b.metric("Macro F1", pct(r["test_macro_f1"]))
            c.metric("Weighted F1", pct(r["test_weighted_f1"]))
            a, b = st.columns(2)
            a.metric("Training time", secs(r["fit_seconds"]))
            b.metric("Feature set", FS_LABEL.get(r["feature_set"], r["feature_set"]), help=FS_DESC.get(r["feature_set"]))
            st.caption(STATUS.get(r["status"], ("", "", ""))[2]
                       + (f" Trained on {count(r['n_train'])} rows, tested on {count(r['n_test'])}."
                          if r["status"] == "SUBSAMPLE" else ""))


def page_confusion():
    table = require_results()
    st.title("🎯 Confusion Matrix")
    stems = get_confusion_stems()
    by_stem = {r["stem"]: r for _, r in table.iterrows()}
    stems = [s for s in stems if s in by_stem]
    if not stems:
        st.info("No completed configuration has a saved confusion matrix in `results/confusion/`.")
        return

    c1, c2, c3 = st.columns([1.4, 1, 1.6], vertical_alignment="bottom")
    stem = c1.selectbox("Model / configuration", stems, format_func=lambda s: config_label(by_stem[s]),
                        key="cm_stem")
    mode = c2.segmented_control("Values", ["Raw Counts", "Normalized"], default="Normalized",
                                key="cm_mode", required=True)
    r, cm = by_stem[stem], get_confusion(stem)
    c3.markdown(f"{badge(r['status'])} &nbsp; {FS_LABEL[r['feature_set']]} features · accuracy "
                f"**{pct(r['test_accuracy'])}** · {int(cm.to_numpy().sum()):,} test rows")
    st.caption("Diagonal cells represent correct predictions; off-diagonal cells represent misclassifications."
               + (" Normalized = share of each true activity." if mode == "Normalized" else ""))
    confusion_heatmap(cm, mode == "Normalized")
    missing = table[~table["stem"].isin(stems)]
    if not missing.empty:
        st.caption("These configurations do not have a saved confusion matrix: "
                   + ", ".join(config_label(x) for _, x in missing.iterrows()) + ".")


def confusion_heatmap(cm: pd.DataFrame, normalised: bool):
    disp = [n[:1].upper() + n[1:] for n in cm.index]
    cm = pd.DataFrame(cm.to_numpy(), index=disp, columns=disp)
    norm = dl.row_normalise(cm)
    show = norm if normalised else cm
    long = show.reset_index(names="True").melt(id_vars="True", var_name="Predicted", value_name="Value")
    long["Count"] = cm.to_numpy().ravel()
    long["Share"] = norm.to_numpy().ravel()
    palette = ["#161a21", "#184f95", "#3987e5", "#cde2fb"]
    if normalised:
        long["Text"] = long["Value"].map(lambda v: f"{v:.2f}" if v >= 0.005 else "")
        cscale, thresh = alt.Scale(domain=[0, 1], range=palette), 0.6
    else:
        long["Text"] = long["Value"].map(lambda v: f"{int(v):,}" if v > 0 else "")
        vmax = float(long["Value"].max()) or 1.0
        cscale, thresh = alt.Scale(type="sqrt", domain=[0, vmax], range=palette), 0.36 * vmax
    base = alt.Chart(long).encode(
        x=alt.X("Predicted:N", sort=disp, title="Predicted activity", axis=alt.Axis(labelAngle=-40, labelAlign="right", labelBaseline="top")),
        y=alt.Y("True:N", sort=disp, title="True activity"))
    rect = base.mark_rect(stroke="#0d0f13", strokeWidth=1).encode(
        color=alt.Color("Value:Q", scale=cscale, legend=None),
        tooltip=["True", "Predicted", alt.Tooltip("Count:Q", format=","),
                 alt.Tooltip("Share:Q", title="share of true class", format=".2%")])
    text = base.mark_text(fontSize=10 if normalised else 9).encode(
        text="Text:N", color=alt.condition(alt.datum.Value > thresh, alt.value("#0b0b0b"), alt.value(INK)))
    show_chart((rect + text).properties(height=500), "Confusion matrix heatmap")


# ----------------------------------------------------------------------------- prediction demo

def pick_sample(test: pd.DataFrame, activity: str):
    pool = test.index if activity == "Any activity" else \
        test.index[test["activity_id"] == {act_name(k): k for k in ACTIVITY_NAMES}[activity]]
    st.session_state["demo_pos"] = int(np.random.default_rng().choice(pool.to_numpy()))


def model_option_label(m: dict) -> str:
    return f"{model_name(m['model'])} — {FS_LABEL.get(m['feature_set'], m['feature_set'])}"


def page_predict():
    st.title("🤖 Prediction Demo")
    st.markdown("##### Interactive PAMAP2 Test-Sample Prediction")

    try:
        with st.spinner("Loading held-out test samples (first time only)…"):
            test, check = get_test_samples()
    except FileNotFoundError as e:
        st.error(f"Processed data not found ({e}). Run `python -m scripts.build_dataset` first.")
        return
    if not check["ok"]:
        st.error(f"The split on disk ({check['fingerprint']}) does not match results/split_meta.json "
                 f"({check['expected']}); samples may not be the reported test rows.")

    # Deployable = artifact + metadata exist and the model loads and fits its feature set.
    deployable, load_errors = [], []
    for m in get_demo_models():
        try:
            get_model(m["artifact_path"], m["feature_set"], m["model"])
            deployable.append(m)
        except Exception as e:
            load_errors.append(f"{m.get('stem', '?')}: {e}")

    names = ["Any activity"] + [act_name(c) for c in LABEL_ORDER]
    if "demo_pos" not in st.session_state:
        pick_sample(test, "Any activity")

    c_sample, c_model, c_result = st.columns([1, 1, 1.15], gap="medium")

    with c_sample:
        st.markdown("### 1 · Test sample")
        act = st.selectbox("Activity", names, key="demo_act",
                           on_change=lambda: pick_sample(test, st.session_state["demo_act"]))
        st.button("🎲 Random test sample", width="stretch", key="demo_random",
                  on_click=lambda: pick_sample(test, act))
        sample = test.loc[st.session_state["demo_pos"]]
        true_id = int(sample["activity_id"])
        with st.container(border=True):
            st.caption("CURRENT TEST SAMPLE")
            st.markdown(f"**Sample ID** `{int(sample['row_id']):,}` · Subject {int(sample['subject_id'])}  \n"
                        f"**Actual activity:** {act_name(true_id)}")

    with c_model:
        st.markdown("### 2 · Model")
        meta = st.selectbox("Model", deployable, format_func=model_option_label, key="demo_model",
                            disabled=not deployable, placeholder="No deployment model available",
                            index=0 if deployable else None)
        if len(deployable) == 1:
            st.caption(f"Available deployment model: {model_option_label(deployable[0])}")
        go = st.button("PREDICT", type="primary", width="stretch", disabled=meta is None, key="demo_predict")
        for err in load_errors:
            st.error(f"Model artifact could not be loaded safely ({err}).")

    preds = st.session_state.setdefault("demo_preds", {})
    if go and meta is not None:
        model = get_model(meta["artifact_path"], meta["feature_set"], meta["model"])
        x = sample[FEATURE_SETS[meta["feature_set"]]].to_numpy(dtype=np.float32)
        preds[(st.session_state["demo_pos"], meta["stem"])] = dl.predict_one(model, x)

    with c_result:
        st.markdown("### 3 · Result")
        with st.container(border=True):
            res = preds.get((st.session_state["demo_pos"], meta["stem"])) if meta else None
            if res is None:
                if meta:
                    st.caption("Press **PREDICT** to classify the current test sample.")
                else:
                    st.warning("No saved deployment artifact. The experiments have results but no saved "
                               "trained model, so none can be used in the prediction demo. The dashboard "
                               "does not train models.", icon=":material/info:")
            else:
                pred_id, proba = res
                st.markdown(f"**Model used:** {model_option_label(meta)}  \n"
                            f"**Actual:** {act_name(true_id).upper()}  \n"
                            f"**Predicted:** {act_name(pred_id).upper()}")
                if pred_id == true_id:
                    st.success("✓ CORRECT PREDICTION")
                else:
                    st.error("✗ INCORRECT PREDICTION")
                if proba is None:
                    st.caption("This model has no predict_proba(), so no confidence is shown.")
                else:
                    top = proba.sort_values(ascending=False).head(3)
                    st.dataframe(pd.DataFrame({"Top predictions": [n[:1].upper() + n[1:] for n in top.index],
                                               "Probability": top.to_numpy()}),
                                 hide_index=True, width="stretch",
                                 column_config={"Probability": st.column_config.ProgressColumn(
                                     format="percent", min_value=0.0, max_value=1.0)})

    st.markdown("### Sample features")
    used = set(FEATURE_SETS[meta["feature_set"]]) if meta else set(FEATURE_SETS["full"])
    rows = []
    for loc in ("hand", "chest", "ankle"):
        row = {"Sensor": f"{loc.title()} IMU", "Temp °C": sample[f"{loc}_temp"]}
        for k, lab in (("acc16", "Acc"), ("gyro", "Gyro"), ("mag", "Mag")):
            for a in "xyz":
                row[f"{lab} {a}"] = sample[f"{loc}_{k}_{a}"]
        row["Used by model"] = f"{loc}_temp" in used
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.2f") for c in rows[0]
                                if c not in ("Sensor", "Used by model")})
    st.caption(f"Heart rate {float(sample['heart_rate']):.0f} bpm (used by every model). One 100 Hz reading "
               f"from the frozen held-out test split ({len(test):,} rows). Acc in m/s², gyro in rad/s, mag in µT.")


def page_about():
    s = get_stats()
    st.title("📚 About")
    a, b = st.columns([1, 1], gap="large")
    with a:
        st.table(pd.DataFrame({"": ["Dataset", "Activities", "Feature sets", "Models", "Metrics"],
                               "Details": [f"PAMAP2 (Protocol, {count(s['n_subjects'])} subjects, 100 Hz)",
                                           str(s["n_activities"]),
                                           f"Reduced ({s['n_reduced']}) / Full ({s['n_full']})",
                                           ", ".join(v[0] for v in MODEL_INFO.values()),
                                           "Accuracy, Macro F1, Weighted F1"]}).set_index(""))
    with b:
        st.markdown("**Activities**")
        st.markdown(" ".join(f":gray-badge[{act_name(c)}]" for c in LABEL_ORDER))
        st.markdown("**Limitation**")
        st.warning("The project uses a frozen random row-level train/test split. Because nearby sensor "
                   "observations can be correlated, this may produce optimistic estimates of generalization "
                   "to completely unseen subjects or future time periods.", icon=":material/warning:")


# ============================================================================ main

def main():
    st.markdown(CSS, unsafe_allow_html=True)
    if "page" not in st.session_state:
        st.session_state["page"] = PAGES[0]
    with st.sidebar:
        st.markdown("### Activity Recognition")
        st.caption("PAMAP2 · interactive demo")
        page = st.radio("Navigate", PAGES, key="page", label_visibility="collapsed")
        st.divider()
        st.caption("Reads saved results only. Never trains a model.")
    {PAGES[0]: page_overview, PAGES[1]: page_models, PAGES[2]: page_confusion,
     PAGES[3]: page_predict, PAGES[4]: page_about}[page]()


main()
