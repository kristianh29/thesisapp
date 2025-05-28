import dash
from dash import html, dcc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import base64
import plotly.colors as pc
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# === Load and prepare data ===
data = pd.read_csv("C:/Users/hussa/OneDrive/Desktop/thesisapp/thesisapp/standardized_test_etios+figo.csv", low_memory=False)
data_trip8 = data[data["trip"] == 8].copy()
data_trip8["fourth_root_NOx"] = data_trip8["NOx_mass_cor"] ** 0.25
data_trip8["fourth_root_CO"] = data_trip8["CO_mass"] ** 0.25
data_trip8["CO2_mass"] = data_trip8["CO2_mass"]

# Observation IDs
obs_ids_dict = {
    "NOx": [651, 1750, 3332, 5138],
    "CO": [3594, 4093, 4504],
    "CO2": [405, 925, 2502, 3718, 4889]
}
default_pollutant = "NOx"
obs_ids = obs_ids_dict[default_pollutant]


image_paths = {
    "NOx": {
        "LightGBM": {
            651: "C:/Users/hussa/OneDrive/Desktop/thesisapp/thesisapp/graphics/ShapleyLGBM/NOx_11827.png",
            1750: "C:/Users/hussa/OneDrive/Desktop/thesisapp/thesisapp/graphics/ShapleyLGBM/NOx_12926.png",
            3332: "C:/Users/hussa/OneDrive/Desktop/thesisapp/thesisapp/graphics/ShapleyLGBM/NOx_14508.png",
            5138: "C:/Users/hussa/OneDrive/Desktop/thesisapp/thesisapp/graphics/ShapleyLGBM/NOx_16314.png"
        },
        "MLP": {
            651: "C:/Users/hussa/OneDrive/Desktop/thesisapp/thesisapp/graphics/ShapleyANN/NOx/shap_waterfall_11827.png",
            1750: "C:/Users/hussa/OneDrive/Desktop/thesisapp/thesisapp/graphics/ShapleyANN/NOx/shap_waterfall_12926.png",
            3332: "C:/Users/hussa/OneDrive/Desktop/thesisapp/thesisapp/graphics/ShapleyANN/NOx/shap_waterfall_14508.png",
            5138: "C:/Users/hussa/OneDrive/Desktop/thesisapp/thesisapp/graphics/ShapleyANN/NOx/shap_waterfall_16314.png"
        },
        "LSTM": {
            651: "C:/Users/hussa/OneDrive/Desktop/thesisapp/thesisapp/graphics/LIMEplots/LSTM/LSTMnox11827.png",
            1750: "C:/Users/hussa/OneDrive/Desktop/thesisapp/thesisapp/graphics/LIMEplots/LSTM/LSTMnox12926.png",
            3332: "C:/Users/hussa/OneDrive/Desktop/thesisapp/thesisapp/graphics/LIMEplots/LSTM/LSTMnox14508.png",
            5138: "C:/Users/hussa/OneDrive/Desktop/thesisapp/thesisapp/graphics/LIMEplots/LSTM/LSTMnox16314.png"
        },
        "GRU": {
            651: "C:/Users/hussa/OneDrive/Desktop/thesisapp/thesisapp/graphics/LIMEplots/GRU/GRUnox11827.png",
            1750: "C:/Users/hussa/OneDrive/Desktop/thesisapp/thesisapp/graphics/LIMEplots/GRU/GRUnox12926.png",
            3332: "C:/Users/hussa/OneDrive/Desktop/thesisapp/thesisapp/graphics/LIMEplots/GRU/GRUnox14508.png",
            5138: "C:/Users/hussa/OneDrive/Desktop/thesisapp/thesisapp/graphics/LIMEplots/GRU/GRUnox16314.png"
        }
    },
    "CO": {
        "LightGBM": {
            3594: "C:/Users/hussa/OneDrive/Desktop/thesisapp/thesisapp/graphics/ShapleyLGBM/CO_14770.png",
            4093: "C:/Users/hussa/OneDrive/Desktop/thesisapp/thesisapp/graphics/ShapleyLGBM/CO_15269.png",
            4504: "C:/Users/hussa/OneDrive/Desktop/thesisapp/thesisapp/graphics/ShapleyLGBM/CO_15680.png",     
        },
        "MLP": {
            3594: "C:/Users/hussa/OneDrive/Desktop/thesisapp/thesisapp/graphics/ShapleyANN/CO/shap_waterfall_14770.png",
            4093: "C:/Users/hussa/OneDrive/Desktop/thesisapp/thesisapp/graphics/ShapleyANN/CO/shap_waterfall_15269.png",
            4504: "C:/Users/hussa/OneDrive/Desktop/thesisapp/thesisapp/graphics/ShapleyANN/CO/shap_waterfall_15680.png",
        },
        "LSTM": {
            3594: "C:/Users/hussa/OneDrive/Desktop/thesisapp/thesisapp/graphics/LIMEplots/LSTM/LSTMco14770.png",
            4093: "C:/Users/hussa/OneDrive/Desktop/thesisapp/thesisapp/graphics/LIMEplots/LSTM/LSTMco15269.png",
            4504: "C:/Users/hussa/OneDrive/Desktop/thesisapp/thesisapp/graphics/LIMEplots/LSTM/LSTMco15680.png"
        },
        "GRU": {
            3594: "C:/Users/hussa/OneDrive/Desktop/thesisapp/thesisapp/graphics/LIMEplots/GRU/GRUco14770.png",
            4093: "C:/Users/hussa/OneDrive/Desktop/thesisapp/thesisapp/graphics/LIMEplots/GRU/GRUco15269.png",
            4504: "C:/Users/hussa/OneDrive/Desktop/thesisapp/thesisapp/graphics/LIMEplots/GRU/GRUco15680.png"
        }
    },
    "CO2":{
        "LightGBM": {
            405: "C:/Users/hussa/OneDrive/Desktop/thesisapp/thesisapp/graphics/ShapleyLGBM/CO2_11581.png",
            925: "C:/Users/hussa/OneDrive/Desktop/thesisapp/thesisapp/graphics/ShapleyLGBM/CO2_12101.png",
            2502: "C:/Users/hussa/OneDrive/Desktop/thesisapp/thesisapp/graphics/ShapleyLGBM/CO2_13678.png",
            3718: "C:/Users/hussa/OneDrive/Desktop/thesisapp/thesisapp/graphics/ShapleyLGBM/CO2_14894.png",
            4889: "C:/Users/hussa/OneDrive/Desktop/thesisapp/thesisapp/graphics/ShapleyLGBM/CO2_16065.png",                
        },
        "SVR" : {
            405: "C:/Users/hussa/OneDrive/Desktop/thesisapp/thesisapp/graphics/ShapleySVR/shap_waterfall_11581.png",
            925: "C:/Users/hussa/OneDrive/Desktop/thesisapp/thesisapp/graphics/ShapleySVR/shap_waterfall_12101.png",
            2502: "C:/Users/hussa/OneDrive/Desktop/thesisapp/thesisapp/graphics/ShapleySVR/shap_waterfall_13678.png",
            3718: "C:/Users/hussa/OneDrive/Desktop/thesisapp/thesisapp/graphics/ShapleySVR/shap_waterfall_14894.png",
            4889: "C:/Users/hussa/OneDrive/Desktop/thesisapp/thesisapp/graphics/ShapleySVR/shap_waterfall_16065.png",
        },
        "MLP": {
            405: "C:/Users/hussa/OneDrive/Desktop/thesisapp/thesisapp/graphics/ShapleyANN/CO2/shap_waterfall_11581.png",
            925: "C:/Users/hussa/OneDrive/Desktop/thesisapp/thesisapp/graphics/ShapleyANN/CO2/shap_waterfall_12101.png",
            2502: "C:/Users/hussa/OneDrive/Desktop/thesisapp/thesisapp/graphics/ShapleyANN/CO2/shap_waterfall_13678.png",
            3718: "C:/Users/hussa/OneDrive/Desktop/thesisapp/thesisapp/graphics/ShapleyANN/CO2/shap_waterfall_14894.png",
            4889: "C:/Users/hussa/OneDrive/Desktop/thesisapp/thesisapp/graphics/ShapleyANN/CO2/shap_waterfall_16065.png"
        },
        "LSTM": {
            405: "C:/Users/hussa/OneDrive/Desktop/thesisapp/thesisapp/graphics/LIMEplots/LSTM/LSTMco2_11581.png",
            925: "C:/Users/hussa/OneDrive/Desktop/thesisapp/thesisapp/graphics/LIMEplots/LSTM/LSTMco2_12101.png",
            2502: "C:/Users/hussa/OneDrive/Desktop/thesisapp/thesisapp/graphics/LIMEplots/LSTM/LSTMco2_13678.png",
            3718: "C:/Users/hussa/OneDrive/Desktop/thesisapp/thesisapp/graphics/LIMEplots/LSTM/LSTMco2_14894.png",
            4889: "C:/Users/hussa/OneDrive/Desktop/thesisapp/thesisapp/graphics/LIMEplots/LSTM/LSTMco2_16065.png"
        },
        "GRU": {
            405: "C:/Users/hussa/OneDrive/Desktop/thesisapp/thesisapp/graphics/LIMEplots/GRU/GRUco2_11581.png", 
            925: "C:/Users/hussa/OneDrive/Desktop/thesisapp/thesisapp/graphics/LIMEplots/GRU/GRUco2_12101.png",
            2502: "C:/Users/hussa/OneDrive/Desktop/thesisapp/thesisapp/graphics/LIMEplots/GRU/GRUco2_13678.png",
            3718: "C:/Users/hussa/OneDrive/Desktop/thesisapp/thesisapp/graphics/LIMEplots/GRU/GRUco2_14894.png",
            4889: "C:/Users/hussa/OneDrive/Desktop/thesisapp/thesisapp/graphics/LIMEplots/GRU/GRUco2_16065.png"                              
        }
    }
}
encoded_images = {}
for pol, model_paths_dict in image_paths.items():
    encoded_images[pol] = {}
    for model, obs_paths in model_paths_dict.items():
        encoded_images[pol][model] = {
            obs_id: base64.b64encode(open(path, 'rb').read()).decode()
            for obs_id, path in obs_paths.items()
        }

# === Create Dash App ===
app = dash.Dash(__name__)
server = app.server
# Initial point
initial_id = obs_ids[0]
initial_point = data_trip8.iloc[initial_id]

start_point = data_trip8.iloc[0]


fig = px.scatter_mapbox(
    data_trip8,
    lat="gps_lat", lon="gps_lon",
    color="fourth_root_NOx",
    color_continuous_scale="RdYlGn_r",
    zoom=11,
    height=600,
    labels={"fourth_root_NOx": "NOx (4th root)"}
)

fig.update_layout(mapbox_style="carto-positron")

# Add dynamic star marker (initially on first point)
fig.add_trace(go.Scattermapbox(
    lat=[initial_point["gps_lat"]],
    lon=[initial_point["gps_lon"]],
    mode='markers',
    marker=dict(size=15, symbol='square', color='blue'),
    name="Selected Point",
    hoverinfo='skip'
))
fig.add_trace(go.Scattermapbox(
    lat=[start_point["gps_lat"]],
    lon=[start_point["gps_lon"]],
    mode='markers',
    marker=dict(size=18, symbol='circle', color='red'),
    name="Start of Trip",
    hoverinfo='text',
    text=["Trip Start"],
    showlegend=True
))

# === App layout ===
app.layout = html.Div([
    dcc.Store(id='obs-id-store'),

    # Left Panel: Controls + Map
    html.Div([
        html.Div([
            html.Div([
                html.Label("Emission Output", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='pollutant-selector',
                    options=[
                        {'label': 'NOx', 'value': 'NOx'},
                        {'label': 'CO', 'value': 'CO'},
                        {'label': 'CO2', 'value': 'CO2'}
                    ],
                    value='NOx',
                    clearable=False,
                    style={'width': '200px', 'margin-top': '8px'}
                ),
            ], style={'display': 'flex', 'flexDirection': 'column', 'margin-right': '40px'}),

            html.Div([
                html.Label("Model", style={'fontWeight': 'bold'}),
                dcc.RadioItems(
                    id='model-selector',
                    options=[],  # Empty initially, will be filled by callback
                    value='MLP',
                    labelStyle={'display': 'inline-block', 'margin-right': '10px'},
                    style={'margin-top': '8px'}
                ),
            ], id='model-selector-container', style={'display': 'none'})
        ], style={
            'display': 'flex',
            'flexDirection': 'row',
            'alignItems': 'flex-start',
            'margin-bottom': '10px'
        }),

        dcc.Graph(id='map', figure=fig, style={'height': '800px', 'width': '100%'})
    ], style={'flex': '70%', 'padding': '20px'}),

    # Right Panel: SHAP image + slider
    html.Div([
        html.Img(id='shap-image',
                 src='data:image/png;base64,{}'.format(encoded_images["NOx"]["MLP"][obs_ids_dict["NOx"][0]]),
                 style={'width': '100%', 'height': 'auto'}),
        dcc.Slider(id='obs-slider', min=0, max=0, step=1, marks={}, value=0)
    ], style={'flex': '30%', 'padding': '20px'})
], style={'display': 'flex', 'flex-direction': 'row'})

# === Callbacks ===
@app.callback(
    [dash.Output('map', 'figure'),
     dash.Output('shap-image', 'src')],
    [dash.Input('obs-slider', 'value'),
     dash.Input('pollutant-selector', 'value'),
     dash.Input('model-selector', 'value')],
    dash.State('obs-id-store', 'data')
)
def update_output(selected_idx, pollutant, model, obs_ids):
    obs_id = obs_ids[selected_idx]
    selected_point = data_trip8.iloc[obs_id]

    # Choose appropriate variable
    color_column = {
        "NOx": "fourth_root_NOx",
        "CO": "fourth_root_CO",
        "CO2": "CO2_mass"  
    }[pollutant]
    shap_prefix = "NOx" if pollutant == "NOx" else "CO"  # if SHAP images are in separate folders or named accordingly

    # Create map
    label_title = {
        "fourth_root_NOx": "NOx (4th root)",
        "fourth_root_CO": "CO (4th root)",
        "CO2_mass": "CO₂ mass"
    }[color_column]

    new_fig = px.scatter_mapbox(
        data_trip8,
        lat="gps_lat", lon="gps_lon",
        color=color_column,
        color_continuous_scale="RdYlGn_r",
        zoom=11,
        height=600,
        labels={color_column: label_title}
    )
    new_fig.update_layout(mapbox_style="carto-positron")



    # Move the trace legend to bottom left or any non-overlapping spot
    new_fig.update_layout(
        legend=dict(
            title="Legend",
            x=0.01, y=0.99,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1,
            font=dict(size=12)
        )
    )

    # Get the color value for the selected observation
    cmap = cm.get_cmap('RdYlGn_r')
    norm = mcolors.Normalize(
        vmin=data_trip8[color_column].min(),
        vmax=data_trip8[color_column].max()
    )
    rgba = cmap(norm(selected_point[color_column]))
    r, g, b, a = [int(255 * rgba[i]) if i < 3 else rgba[i] for i in range(4)]
    legend_color = f'rgba({r}, {g}, {b}, {a:.2f})'

    # Add star marker (still black or fixed)
    value_text = f"{selected_point[color_column]:.2f}"
    new_fig.add_trace(go.Scattermapbox(
        lat=[selected_point["gps_lat"]],
        lon=[selected_point["gps_lon"]],
        mode='markers+text',
        marker=dict(size=15, symbol='circle', color='black'),
        text=[f"Value ≈ {value_text}"],
        textposition="top right",  # Or try "top center", "bottom left", etc.
        name="Selected Point",
        hoverinfo='skip',
        showlegend=False  # Don't show this in the legend
    ))

    start_point = data_trip8.iloc[0]
   
    new_fig.add_trace(go.Scattermapbox(
        lat=[start_point["gps_lat"]],
        lon=[start_point["gps_lon"]],
        mode='markers',
        marker=dict(size=18, symbol='circle', color='red'),
        name="Start of Trip",
        hoverinfo='text',
        text=["Trip Start"],
        showlegend=True
))

    # Add dummy marker to simulate a legend entry for selected observation color
    new_fig.add_trace(go.Scattermapbox(
        lat=[None], lon=[None],
        mode='markers',
        marker=dict(size=12, color=legend_color),
        name=f"Obs {obs_id} value ≈ {value_text}",
        showlegend=True,
        hoverinfo='skip'
    ))

    # SHAP image
    image_base64 = encoded_images.get(pollutant, {}).get(model, {}).get(obs_id, "")


    image_src = f'data:image/png;base64,{image_base64}' if image_base64 else ""

    return new_fig, image_src

@app.callback(
    dash.Output('obs-id-store', 'data'),
    dash.Input('pollutant-selector', 'value')
)
def update_obs_ids(pollutant):
    return obs_ids_dict[pollutant]

@app.callback(
    [dash.Output('obs-slider', 'min'),
     dash.Output('obs-slider', 'max'),
     dash.Output('obs-slider', 'marks'),
     dash.Output('obs-slider', 'value')],
    dash.Input('obs-id-store', 'data')
)
def update_slider(obs_ids):
    marks = {i: str(i + 1) for i in range(len(obs_ids))}
    return 0, len(obs_ids) - 1, marks, 0  # Reset slider to 0 on update

@app.callback(
    dash.Output('model-selector-container', 'style'),
    dash.Input('pollutant-selector', 'value')
)
def toggle_model_selector(pollutant):
    return {'display': 'block'}

@app.callback(
    [dash.Output('model-selector', 'options'),
     dash.Output('model-selector', 'value')],
    dash.Input('pollutant-selector', 'value')
)
def update_model_options(pollutant):
    # Always available models
    base_models = [
        {'label': 'LightGBM', 'value': 'LightGBM'},
        {'label': 'MLP', 'value': 'MLP'},
        {'label': 'LSTM', 'value': 'LSTM'},
        {'label': 'GRU', 'value': 'GRU'}
    ]

    # Only add SVR for CO2 between LightGBM and MLP
    if pollutant == 'CO2':
        base_models.insert(1, {'label': 'SVR', 'value': 'SVR'})

    return base_models, 'SVR' if pollutant == 'CO2' else 'MLP'


# === Run the app ===
if __name__ == '__main__':
    app.run(debug=True)





