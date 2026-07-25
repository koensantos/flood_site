import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


st.set_page_config(
    page_title="Hoboken Flood Risk Portfolio",
    layout="wide",
)


@st.cache_data
def load_feature_grid():
    df = pd.read_csv("output_csvs/feature_grid.csv")
    return df.dropna(subset=["elevation_m", "rainfall_inches", "dist_to_flood_zone_m", "in_fema_flood_zone"]).copy()


@st.cache_data
def train_models(df: pd.DataFrame):
    features = ["elevation_m", "rainfall_inches"]
    target = "in_fema_flood_zone"

    df_train = df[features + [target]].copy()
    X = df_train[features].values
    y = df_train[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train_scaled, y_train)

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos = neg / pos if pos > 0 else 1

    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )
    xgb_model.fit(X_train_scaled, y_train)

    rf_proba = rf.predict_proba(X_test_scaled)[:, 1]
    xgb_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]

    return {
        "rf": rf,
        "xgb": xgb_model,
        "scaler": scaler,
        "metrics": {
            "rf_auc": roc_auc_score(y_test, rf_proba),
            "xgb_auc": roc_auc_score(y_test, xgb_proba),
            "rf_accuracy": accuracy_score(y_test, rf.predict(X_test_scaled)),
            "xgb_accuracy": accuracy_score(y_test, xgb_model.predict(X_test_scaled)),
        },
    }


@st.cache_data
def compute_scenario_predictions(df: pd.DataFrame, rainfall_inches: float, _models: dict):
    features = ["elevation_m", "rainfall_inches"]
    scenario = df.copy()
    scenario["rainfall_inches"] = rainfall_inches
    X_scenario = _models["scaler"].transform(scenario[features].values)

    rf_proba = _models["rf"].predict_proba(X_scenario)[:, 1]
    xgb_proba = _models["xgb"].predict_proba(X_scenario)[:, 1]

    scenario["rf_probability"] = rf_proba
    scenario["xgb_probability"] = xgb_proba

    base_prob = scenario[["rf_probability", "xgb_probability"]].mean(axis=1)
    rain_scale = np.clip(rainfall_inches / 25.0, 0.0, 1.0)

    elev_min = scenario["elevation_m"].min()
    elev_max = scenario["elevation_m"].max()
    elev_norm = (scenario["elevation_m"] - elev_min) / (elev_max - elev_min + 1e-9)
    dist_min = scenario["dist_to_flood_zone_m"].min()
    dist_max = scenario["dist_to_flood_zone_m"].max()
    dist_norm = (scenario["dist_to_flood_zone_m"] - dist_min) / (dist_max - dist_min + 1e-9)

    flood_pressure = (1 - elev_norm) * 0.5 + (1 - dist_norm) * 0.3 + rain_scale * 0.7
    flood_pressure = np.clip(flood_pressure, 0.0, 1.0)
    scenario["avg_probability"] = np.clip(0.25 * base_prob + 0.75 * flood_pressure, 0.0, 1.0)
    scenario["risk_level"] = np.where(scenario["avg_probability"] >= 0.6, "High", np.where(scenario["avg_probability"] >= 0.4, "Medium", "Low"))
    return scenario


def render_flood_amount_map(scenario: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        scenario["centroid_x"],
        scenario["centroid_y"],
        c=scenario["avg_probability"],
        cmap="RdYlBu_r",
        s=20,
        vmin=0,
        vmax=1,
        alpha=0.85,
    )
    ax.set_title("Hoboken Flood Amount Map\nAverage RF + XGB Probability")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    fig.colorbar(scatter, ax=ax, label="Flood probability")
    fig.tight_layout()
    return fig


st.title("🌊 Hoboken Flood Risk Portfolio Dashboard")
st.caption("Uses the same processed flood-risk dataset from this project and lets you explore how rainfall changes the modeled risk profile.")

feature_grid = load_feature_grid()
models = train_models(feature_grid)

with st.sidebar:
    st.header("Scenario Controls")
    rainfall_input = st.slider(
        "Rain amount (inches)",
        min_value=0.0,
        max_value=25.0,
        value=18.83,
        step=0.25,
        help="This replaces the project rainfall input and recalculates flood probability using the same feature grid.",
    )

    st.markdown("### Data Sources")
    st.markdown("- `output_csvs/feature_grid.csv`")
    st.markdown("- `rainfall_dataset/lcd_newark_sept2021.csv`")
    st.markdown("- `fema_flood_zones_dataset/S_FLD_HAZ_AR.shp`")

left, right = st.columns(2)
with left:
    st.metric("Grid cells in scenario", f"{len(feature_grid):,}")
    st.metric("Rainfall input", f"{rainfall_input:.2f} in")

with right:
    st.metric("RF ROC-AUC", f"{models['metrics']['rf_auc']:.3f}")
    st.metric("XGB ROC-AUC", f"{models['metrics']['xgb_auc']:.3f}")

scenario = compute_scenario_predictions(feature_grid, rainfall_input, models)

avg_risk = float(scenario["avg_probability"].mean())
high_risk_cells = int((scenario["avg_probability"] >= 0.6).sum())

col1, col2, col3 = st.columns(3)
col1.metric("Average modeled flood probability", f"{avg_risk:.3f}")
col2.metric("High-risk cells", f"{high_risk_cells:,}")
col3.metric("High-risk share", f"{(high_risk_cells / len(scenario) * 100):.1f}%")

st.subheader("Flood Amount Image")
st.pyplot(render_flood_amount_map(scenario))

st.subheader("Hoboken Reference Image")
st.image("map_flood_amount.png", caption="Hoboken study region reference image", use_container_width=True)

st.subheader("Flood-risk distribution")
chart_df = scenario[["avg_probability", "elevation_m", "dist_to_flood_zone_m"]].copy()
chart_df["risk_bucket"] = pd.cut(
    chart_df["avg_probability"],
    bins=[-0.01, 0.25, 0.5, 0.75, 1.0],
    labels=["Low", "Moderate", "Elevated", "Severe"],
)

st.bar_chart(chart_df.groupby("risk_bucket")["avg_probability"].mean())

st.subheader("Risk by elevation and distance to flood zone")
scatter_df = scenario[["elevation_m", "dist_to_flood_zone_m", "avg_probability"]].sort_values("avg_probability", ascending=False)
st.scatter_chart(scatter_df, x="dist_to_flood_zone_m", y="elevation_m", color="avg_probability")

st.subheader("Top cells with the highest modeled flood risk")
show_df = scenario.sort_values("avg_probability", ascending=False).head(10)
show_df = show_df[["cell_id", "elevation_m", "rainfall_inches", "dist_to_flood_zone_m", "avg_probability", "risk_level"]].copy()
show_df = show_df.rename(columns={"avg_probability": "modeled_risk"})
st.dataframe(show_df, use_container_width=True, hide_index=True)

st.subheader("Portfolio interpretation")
if avg_risk >= 0.6:
    st.success("This rainfall scenario pushes the portfolio into a high flood-risk regime, especially for low-elevation cells and locations close to FEMA high-risk zones.")
elif avg_risk >= 0.4:
    st.warning("This scenario is moderate risk. The modeled output suggests a meaningful share of the grid may experience elevated flood susceptibility.")
else:
    st.info("This rainfall scenario remains relatively low risk. Most of the portfolio sits below the high-risk probability threshold.")

