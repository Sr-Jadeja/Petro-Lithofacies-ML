import gradio as gr
import pandas as pd
import joblib
import lasio
import tempfile
import matplotlib.pyplot as plt

# Load artifacts
model = joblib.load('data/lithology_model.pkl')
scaler = joblib.load('data/lithology_scaler.pkl')

def create_lithology_plot(df):
    df = df.sort_values('DEPT')
    
    # Standard Geological Colors
    color_map = {
        'Sandstone': '#FFFF00', 'Sandstone/Shale': '#D2B48C', 'Shale': '#808080',
        'Marl': '#7FFFD4', 'Dolomite': '#800080', 'Limestone': '#0000FF',
        'Chalk': '#F0FFF0', 'Halite': '#778899', 'Anhydrite': '#FFD700',
        'Tuff': '#FF4500', 'Coal': '#000000', 'Basement': '#FF1493'
    }

    fig, ax = plt.subplots(figsize=(3, 10))
    
    # We plot the lithology as a colored vertical bar
    for litho, color in color_map.items():
        subset = df[df['LITHOLOGY'] == litho]
        if not subset.empty:
            ax.scatter([1] * len(subset), subset['DEPT'], 
                        color=color, label=litho, marker='s', s=50)

    ax.invert_yaxis() # Depth 0 at top
    ax.set_ylabel('Depth (m)')
    ax.set_title('Lithology Column')
    ax.set_xticks([])
    ax.set_xlim(0.5, 1.5)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()

    plot_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    plt.savefig(plot_path, dpi=100)
    plt.close()
    return plot_path

def predict_lithology(file):
    if file is None: return None, None, None
    try:
        las = lasio.read(file.name)
        df = las.df().reset_index()
        df.columns = [col.upper() for col in df.columns]
        
        training_features = ['DEPT', 'CALI', 'RDEP', 'RMED', 'RSHA']
        
        if not all(col in df.columns for col in training_features):
            return "Missing Columns", None, None

        df_clean = df[training_features].dropna().copy()
        X_scaled = scaler.transform(df_clean[training_features])
        df_clean['PREDICTED_ID'] = model.predict(X_scaled)

        litho_map = {30000: 'Sandstone', 65030: 'Sandstone/Shale', 65000: 'Shale',
                     80000: 'Marl', 74000: 'Dolomite', 70000: 'Limestone',
                     70032: 'Chalk', 88000: 'Halite', 86000: 'Anhydrite',
                     99000: 'Tuff', 90000: 'Coal', 93000: 'Basement'}
        df_clean['LITHOLOGY'] = df_clean['PREDICTED_ID'].map(litho_map)

        plot_img = create_lithology_plot(df_clean)
        
        temp_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        df_clean.to_csv(temp_csv.name, index=False)
        
        return df_clean.head(15), temp_csv.name, plot_img
    except Exception as e:
        return str(e), None, None

# NEW UI LAYOUT
with gr.Blocks() as demo:
    gr.Markdown("# 🛢️ Well Lithology Predictor")
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload LAS")
            run_btn = gr.Button("Generate Lithology Log", variant="primary")
            file_out = gr.File(label="Download CSV")
        with gr.Column(scale=1):
            plot_out = gr.Image(label="Vertical Lithology Track") # Label changed here
        with gr.Column(scale=2):
            table_out = gr.Dataframe(label="Data Preview")
            
    run_btn.click(predict_lithology, inputs=file_input, outputs=[table_out, file_out, plot_out])

demo.launch()