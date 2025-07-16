import plotly.graph_objects as go
import plotly.io as pio

videos = [
    {"name": "video1", "frames": 100},
    {"name": "video2", "frames": 150},
    {"name": "video3", "frames": 80},
]

fig = go.Figure()

for v in videos:
    fig.add_trace(go.Bar(
        x=[v["frames"]],
        y=[v["name"]],
        orientation='h',
        hovertemplate=f'{v["name"]}<br>Frames: {v["frames"]}<extra></extra>',
        marker=dict(color='lightblue'),
    ))

fig.update_layout(
    title="Videos Frame Count",
    xaxis_title="Number of Frames",
    yaxis_title="Video Name",
    yaxis=dict(autorange="reversed"),
    height=300,
    margin=dict(l=100, r=20, t=40, b=20),
)

# Save as HTML and open automatically
pio.write_html(fig, "output.html", auto_open=True)

input("Press Enter to exit...")
