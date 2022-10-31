import plotly.graph_objects as go
import pandas as pd

# Read data from a csv
z_data = pd.read_csv('mt_bruno_elevation.csv')

fig = go.Figure(data=go.Surface(z=z_data, showscale=False))
fig.update_layout(
    title='Mt Bruno Elevation',
    width=400, height=400,
    margin=dict(t=30, r=0, l=20, b=10)
)

name = 'eye = (x:2, y:2, z:1)'
camera = dict(
    eye=dict(x=0, y=0, z=1)
)

fig.update_layout(scene_camera=camera, title=name)
fig.show()