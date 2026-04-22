import json
import gradio as gr
import pandas as pd
import joblib
import lasio
import tempfile
import matplotlib.pyplot as plt

# ── Load artifacts ─────────────────────────────────────────────

model  = joblib.load("data/lithology_model.pkl")
scaler = joblib.load("data/lithology_scaler.pkl")

with open("data/selected_features.json") as f:
    FEATURES = json.load(f)

LITHO_MAP = {
    30000: "Sandstone",    65030: "Sandstone/Shale", 65000: "Shale",
    80000: "Marl",         74000: "Dolomite",         70000: "Limestone",
    70032: "Chalk",        88000: "Halite",            86000: "Anhydrite",
    99000: "Tuff",         90000: "Coal",              93000: "Basement",
}

COLOR_MAP = {
    "Sandstone": "#FFFF00", "Sandstone/Shale": "#D2B48C", "Shale": "#808080",
    "Marl": "#7FFFD4",      "Dolomite": "#800080",         "Limestone": "#0000FF",
    "Chalk": "#F0FFF0",     "Halite": "#778899",           "Anhydrite": "#FFD700",
    "Tuff": "#FF4500",      "Coal": "#000000",             "Basement": "#FF1493",
}

# ── Plot ───────────────────────────────────────────────────────

def plot_lithology(df):
    df = df.sort_values("DEPT")
    fig, ax = plt.subplots(figsize=(3, 10))

    for litho, color in COLOR_MAP.items():
        subset = df[df["LITHOLOGY"] == litho]
        if not subset.empty:
            ax.scatter([1] * len(subset), subset["DEPT"],
                       color=color, label=litho, marker="s", s=50)

    ax.invert_yaxis()
    ax.set_ylabel("Depth (m)")
    ax.set_title("Lithology Column")
    ax.set_xticks([])
    ax.set_xlim(0.5, 1.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
    plt.tight_layout()

    path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    plt.savefig(path, dpi=100)
    plt.close()
    return path

# ── Prediction ─────────────────────────────────────────────────

def predict_lithology(file):
    if file is None:
        return None, None, None
    try:
        las = lasio.read(file.name)
        df = las.df().reset_index()
        df.columns = [c.upper() for c in df.columns]

        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            return f"Missing columns: {missing}", None, None

        df_clean = df[FEATURES].dropna().copy()
        df_clean["PREDICTED_ID"] = model.predict(scaler.transform(df_clean))
        df_clean["LITHOLOGY"]    = df_clean["PREDICTED_ID"].map(LITHO_MAP)

        plot_path = plot_lithology(df_clean)
        csv_path  = tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name
        df_clean.to_csv(csv_path, index=False)

        return df_clean.head(15), csv_path, plot_path

    except Exception as e:
        return str(e), None, None

# ── Gradio UI ──────────────────────────────────────────────────

with gr.Blocks() as demo:
    gr.Markdown("# 🛢️ Well Lithology Predictor")
    gr.Markdown(f"**Model trained on:** {', '.join(FEATURES)}")
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload LAS File")
            run_btn    = gr.Button("Generate Lithology Log", variant="primary")
            file_out   = gr.File(label="Download CSV")
        with gr.Column(scale=1):
            plot_out  = gr.Image(label="Lithology Track")
        with gr.Column(scale=2):
            table_out = gr.Dataframe(label="Data Preview")

    run_btn.click(predict_lithology, inputs=file_input, outputs=[table_out, file_out, plot_out])

demo.launch()
