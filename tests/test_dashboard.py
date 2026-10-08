"""Tests for the Streamlit dashboard (app/). Dashboard owner.

Unit tests use hand-made fixtures in temp directories (numbers are arbitrary test inputs, not
results). The AppTest checks render every page against the real repository files and confirm
that rendering leaves results/ byte-for-byte unchanged. No model from the project is trained;
the prediction-flow test fits a tiny throwaway tree on random numbers in a temp dir.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app import data_layer as dl
from har.config import ACTIVITY_NAMES, FEATURE_SETS, LABEL_ORDER, PROCESSED_DIR, RESULTS_DIR
from har.metrics import save_result

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
PAGES = ["🏠 Overview", "📊 Models", "🎯 Confusion Matrix", "🤖 Predict", "📚 About"]
PREDICT = PAGES[3]
HAVE_DATA = (PROCESSED_DIR / "pamap2_protocol_clean.parquet").is_file()


def _cm_csv(path, a):
    pd.DataFrame(a, index=[f"true_{c}" for c in LABEL_ORDER],
                 columns=[f"pred_{c}" for c in LABEL_ORDER]).to_csv(path)


# ----------------------------------------------------------------------------- data layer

def test_confusion_helpers_on_a_known_matrix(tmp_path):
    (tmp_path / "confusion").mkdir()
    a = np.diag(np.arange(1, 13) * 10)
    a[0, 1] = 5                                     # lying -> sitting, 5 rows
    _cm_csv(tmp_path / "confusion" / "m_full.csv", a)
    cm = dl.load_confusion("m_full", tmp_path)
    assert list(cm.index) == [ACTIVITY_NAMES[c] for c in LABEL_ORDER]
    norm = dl.row_normalise(cm)
    assert np.allclose(norm.sum(axis=1), 1.0)
    assert dl.per_class_recall(cm)["lying"] == pytest.approx(10 / 15)
    top = dl.top_confusions(cm, 3)
    assert len(top) == 1 and top.iloc[0]["True activity"] == "lying" and top.iloc[0]["Count"] == 5
    assert dl.list_confusion_stems(tmp_path) == ["m_full"]


def test_missing_or_malformed_files_return_none(tmp_path):
    (tmp_path / "confusion").mkdir()
    (tmp_path / "metrics").mkdir()
    pd.DataFrame([[1, 2], [3, 4]]).to_csv(tmp_path / "confusion" / "bad.csv")   # wrong shape
    (tmp_path / "metrics" / "broken.json").write_text("{not json")
    assert dl.load_confusion("bad", tmp_path) is None
    assert dl.load_confusion("absent", tmp_path) is None
    assert dl.load_result_json("broken", tmp_path) is None
    assert dl.load_result_json("absent", tmp_path) is None
    s = dl.dataset_stats(tmp_path)                  # no split_meta / cleaning_log
    assert s["n_train"] is None and s["n_subjects"] is None and s["class_counts"] == {}
    assert s["n_reduced"] == 11 and s["n_full"] == 31


def test_comparison_uses_project_rules_and_keeps_missing_as_missing(tmp_path):
    (tmp_path / "metrics").mkdir()
    (tmp_path / "split_meta.json").write_text(json.dumps({"n_train": 100, "n_test": 20}))
    rec = {"model": "logreg", "owner": "P1", "feature_set": "full", "protocol": "random_85_15", "seed": 229,
           "n_train": 100, "n_test": 20, "train_accuracy": 0.9, "test_accuracy": 0.8, "test_macro_f1": 0.7,
           "test_weighted_f1": 0.75, "params": {}, "fit_seconds": 1.0, "predict_seconds": 0.1,
           "paper_test_accuracy": 0.815, "notes": ""}
    save_result(rec, results_dir=tmp_path)
    (tmp_path / "metrics" / "mlp_full.json").write_text("{broken")
    table, problems = dl.load_comparison(tmp_path)
    prim = table[table["primary"].astype(bool)]
    assert len(prim) == 12
    st = dict(zip(prim["stem"], prim["status"]))
    assert st["logreg_full"] == "FULL" and st["mlp_full"] == "MALFORMED" and st["svm_rbf_full"] == "NOT RUN"
    assert prim.loc[prim["stem"] == "svm_rbf_full", "test_accuracy"].isna().all()   # never 0
    assert len(problems) == 1
    assert not (tmp_path / "comparison.csv").exists()                              # read-only
    done = dl.completed_results(table)
    assert list(done["stem"]) == ["logreg_full"]                                   # NOT RUN / MALFORMED dropped


def test_demo_model_discovery_validation_and_prediction(tmp_path):
    import joblib
    from sklearn.linear_model import Perceptron
    from sklearn.tree import DecisionTreeClassifier
    rng = np.random.default_rng(0)
    X = rng.normal(size=(240, 31)).astype(np.float32)
    y = np.repeat(LABEL_ORDER, 20)
    tree = DecisionTreeClassifier(random_state=0).fit(X, y)
    joblib.dump(tree, tmp_path / "t_full.joblib")
    (tmp_path / "t_full_metadata.json").write_text(json.dumps({"stem": "t_full", "model": "decision_tree",
                                                      "feature_set": "full"}))
    (tmp_path / "orphan_metadata.json").write_text(json.dumps({"feature_set": "full"}))     # no .joblib
    found = dl.find_demo_models(tmp_path)
    assert [m["stem"] for m in found] == ["t_full"]

    model = dl.load_demo_model(found[0]["artifact_path"], "full")
    with pytest.raises(ValueError):
        dl.load_demo_model(found[0]["artifact_path"], "reduced")                    # 31 vs 11 features
    pred, proba = dl.predict_one(model, X[0])
    assert pred == y[0] and proba is not None and proba.sum() == pytest.approx(1.0)
    assert set(proba.index) == set(ACTIVITY_NAMES.values())

    _, no_proba = dl.predict_one(Perceptron(random_state=0).fit(X, y), X[0])
    assert no_proba is None                                                         # never fabricated

    joblib.dump(Perceptron(random_state=0).fit(X, y), tmp_path / "bare.joblib")
    with pytest.raises(ValueError, match="scaler"):                                 # unscaled logreg-type model
        dl.load_demo_model(str(tmp_path / "bare.joblib"), "full", "logreg")


def test_dashboard_code_never_fits_a_model():
    for f in ("streamlit_app.py", "data_layer.py"):
        src = (APP.parent / f).read_text(encoding="utf-8")
        assert ".fit(" not in src and "run_experiment" not in src
        assert "import export_demo_model" not in src and "from app.export_demo_model" not in src


# ----------------------------------------------------------------------------- deployment artifact

ARTIFACT = dl.MODELS_DIR / "decision_tree_full.joblib"
ARTIFACT_META = dl.MODELS_DIR / "decision_tree_full_metadata.json"
needs_artifact = pytest.mark.skipif(not ARTIFACT.is_file(),
                                    reason="run python -m app.export_demo_model first")


@needs_artifact
def test_decision_tree_artifact_exists_with_metadata():
    meta = json.loads(ARTIFACT_META.read_text())
    assert meta["model"] == "decision_tree" and meta["feature_set"] == "full"
    assert meta["feature_count"] == len(FEATURE_SETS["full"]) == 31
    assert meta["seed"] == 229 and meta["split_fingerprint"] == "204cf31f6f415437"
    assert sorted(int(k) for k in meta["class_labels"]) == LABEL_ORDER
    assert [m["stem"] for m in dl.find_demo_models()] == ["decision_tree_full"]


@needs_artifact
def test_decision_tree_artifact_loads_fitted():
    from sklearn.tree import DecisionTreeClassifier
    model = dl.load_demo_model(str(ARTIFACT), "full", "decision_tree")
    assert isinstance(model, DecisionTreeClassifier)
    assert hasattr(model, "tree_") and model.n_features_in_ == 31            # fitted, not a bare estimator
    assert [int(c) for c in model.classes_] == LABEL_ORDER


@needs_artifact
@pytest.mark.skipif(not HAVE_DATA, reason="processed PAMAP2 data not available")
def test_decision_tree_artifact_predicts_a_held_out_sample():
    model = dl.load_demo_model(str(ARTIFACT), "full", "decision_tree")
    test, check = dl.load_test_samples()
    assert check["ok"]
    x = test.loc[0, FEATURE_SETS["full"]].to_numpy(dtype=np.float32)
    pred, proba = dl.predict_one(model, x)
    assert pred in ACTIVITY_NAMES and len(ACTIVITY_NAMES) == 12
    assert proba is not None and proba.sum() == pytest.approx(1.0)


# ----------------------------------------------------------------------------- app rendering

def _hash_tree(root: Path) -> dict:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.fixture
def app_test():
    st = pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    st.cache_data.clear()
    st.cache_resource.clear()

    def make(page):
        at = AppTest.from_file(str(APP), default_timeout=180)
        at.session_state["page"] = page
        return at
    return make


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_and_results_are_untouched(app_test, page):
    if page == PREDICT and not HAVE_DATA:
        pytest.skip("processed PAMAP2 data not available")
    before = _hash_tree(RESULTS_DIR)
    at = app_test(page).run()
    assert not at.exception, [e.value for e in at.exception]
    assert _hash_tree(RESULTS_DIR) == before
    shown = " ".join(str(x.value) for x in list(at.markdown) + list(at.caption))
    for df in at.dataframe:
        shown += " " + df.value.to_string()
    assert "NOT RUN" not in shown


def _plug_in_tree(tmp_path, monkeypatch):
    import joblib
    from sklearn.tree import DecisionTreeClassifier
    rng = np.random.default_rng(0)
    X = rng.normal(size=(240, len(FEATURE_SETS["full"]))).astype(np.float32)
    tree = DecisionTreeClassifier(random_state=0).fit(X, np.repeat(LABEL_ORDER, 20))   # throwaway
    joblib.dump(tree, tmp_path / "decision_tree_full.joblib")
    (tmp_path / "decision_tree_full_metadata.json").write_text(json.dumps(
        {"stem": "decision_tree_full", "model": "decision_tree", "feature_set": "full", "verified": {}}))
    monkeypatch.setattr(dl, "MODELS_DIR", tmp_path)
    return tree


@pytest.mark.skipif(not HAVE_DATA, reason="processed PAMAP2 data not available")
def test_prediction_flow_uses_the_selected_held_out_sample(app_test, tmp_path, monkeypatch):
    tree = _plug_in_tree(tmp_path, monkeypatch)
    at = app_test(PREDICT).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.selectbox(key="demo_model").value["stem"] == "decision_tree_full"
    pos = at.session_state["demo_pos"]
    at = at.button(key="demo_predict").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["demo_pos"] == pos                                      # sample unchanged

    test, _ = dl.load_test_samples()
    x = test.loc[pos, FEATURE_SETS["full"]].to_numpy(dtype=np.float32).reshape(1, -1)
    pred_id, proba = at.session_state["demo_preds"][(pos, "decision_tree_full")]
    assert pred_id == int(tree.predict(x)[0])                                        # real test row went in
    text = " ".join(m.value for m in at.markdown)
    assert "Model used" in text and "Actual" in text and "Predicted" in text
    assert at.success or at.error


@pytest.mark.skipif(not HAVE_DATA, reason="processed PAMAP2 data not available")
def test_demo_without_artifact_disables_prediction_and_filters_by_activity(app_test, tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "MODELS_DIR", tmp_path)                                  # empty: no artifact
    at = app_test(PREDICT).run()
    assert at.button(key="demo_predict").disabled
    assert any("no saved deployment artifact" in w.value.lower() for w in at.warning)
    at = at.selectbox(key="demo_act").set_value("Rope jumping").run()
    assert not at.exception
    test, _ = dl.load_test_samples()
    pos = at.session_state["demo_pos"]
    assert ACTIVITY_NAMES[int(test.loc[pos, "activity_id"])] == "rope jumping"
    at = at.button(key="demo_random").click().run()
    assert ACTIVITY_NAMES[int(test.loc[at.session_state["demo_pos"], "activity_id"])] == "rope jumping"


@pytest.mark.skipif(not HAVE_DATA, reason="processed PAMAP2 data not available")
def test_switching_model_keeps_the_same_sample(app_test, tmp_path, monkeypatch):
    import joblib
    from sklearn.dummy import DummyClassifier
    _plug_in_tree(tmp_path, monkeypatch)
    X = np.zeros((12, len(FEATURE_SETS["full"])), dtype=np.float32)
    dummy = DummyClassifier(strategy="constant", constant=24).fit(X, LABEL_ORDER)      # always rope jumping
    joblib.dump(dummy, tmp_path / "random_forest_full.joblib")
    (tmp_path / "random_forest_full_metadata.json").write_text(json.dumps(
        {"stem": "random_forest_full", "model": "random_forest", "feature_set": "full"}))

    at = app_test(PREDICT).run()
    pos = at.session_state["demo_pos"]
    at = at.button(key="demo_predict").click().run()
    other = next(m for m in at.selectbox(key="demo_model").options if "Random Forest" in m)
    at = at.selectbox(key="demo_model").select(next(
        m for m in dl.find_demo_models() if m["stem"] == "random_forest_full")).run()
    assert other and at.session_state["demo_pos"] == pos                            # sample unchanged
    at = at.button(key="demo_predict").click().run()
    assert not at.exception, [e.value for e in at.exception]
    preds = at.session_state["demo_preds"]
    assert (pos, "decision_tree_full") in preds and preds[(pos, "random_forest_full")][0] == 24
    assert at.session_state["demo_pos"] == pos
