"""
SwarmScale dashboard.

    python3 dashboard.py        then open http://localhost:8050

Swarm tab:       compile + run any version, watch it live, change the rules mid-run.
Performance tab: charts from results/bench.csv (run bench.py first).
"""
import os
import shutil
import subprocess
import threading

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, State, ctx, no_update

ROOT = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.join(ROOT, "runs", "live")
BIN_DIR = os.path.join(ROOT, "runs", "bin")
BENCH = os.path.join(ROOT, "results", "bench.csv")
SAVE_EVERY = 5
DENSITY = 0.1

BLUE, RED, GREY = "#38bdf8", "#fb7185", "#94a3b8"     # prey / predators / neutral
BG, PANEL, BORDER, TEXT, MUTED = "#0b1120", "#111a2e", "#22304a", "#e2e8f0", "#94a3b8"
ACCENT = "#818cf8"
HEAT = [[0, "rgba(11,17,32,0)"], [0.25, "#3b0f4f"], [0.55, "#9d174d"], [0.8, "#f97316"], [1, "#fde68a"]]
VERSIONS = {
    "omp": "OpenMP (CPU)",
    "cuda": "CUDA + OpenMP",
    "mpi": "MPI + OpenMP",
    "hybrid": "Hybrid (MPI + CUDA + OpenMP)",
}
PRESETS = {
    "flock":  dict(align=0.10, coh=0.02, sep=0.5),
    "avoid":  dict(align=0.0,  coh=0.0,  sep=2.0),
    "loose":  dict(align=0.03, coh=0.005, sep=1.0),
    "tight":  dict(align=0.25, coh=0.08, sep=0.3),
}


# ------------------------------------------------------------------ simulation process

class Sim:
    """The running C program plus everything read back from its output files."""

    def __init__(self):
        self.proc = None
        self.status = "idle"
        self.message = "Choose settings and press Run."
        self.lock = threading.Lock()
        self.reset_data(1000)

    def reset_data(self, n):
        self.n = n
        self.world = float(np.sqrt(n / DENSITY))
        self.frames, self.dens = [], []          # lists of numpy arrays, one per saved step
        self.steps = []
        self.metrics = pd.DataFrame()
        self.offsets = {"frames.csv": 0, "density.csv": 0}
        self.partial = {"frames.csv": "", "density.csv": ""}
        self.pending = {}                         # frame rows not yet complete

    # ---- building and starting

    def build(self, version, n, pred, steps):
        os.makedirs(BIN_DIR, exist_ok=True)
        exe = os.path.join(BIN_DIR, f"{version}_{n}_{pred}_{steps}")
        if os.path.exists(exe):
            return exe, None
        flags = f"-DN={n} -DSTEPS={steps} -DSAVE_EVERY={SAVE_EVERY} -DPRED_PCT={pred}"
        src = {"omp": "swarm.c", "cuda": "swarm_cuda.cu", "mpi": "swarm_mpi.c", "hybrid": "swarm_hybrid.cu"}[version]
        src = os.path.join(ROOT, src)
        cmd = {
            "omp":    f"gcc -O2 -fopenmp {flags} '{src}' -o '{exe}' -lm",
            "cuda":   f"nvcc -O2 --fmad=false -arch=sm_89 -Xcompiler -fopenmp {flags} '{src}' -o '{exe}' -lgomp",
            "mpi":    f"mpicc -O2 -fopenmp {flags} '{src}' -o '{exe}' -lm",
            "hybrid": f"nvcc -O2 --fmad=false -arch=sm_89 -ccbin mpicxx -Xcompiler -fopenmp {flags} '{src}' -o '{exe}' -lgomp",
        }[version]
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if r.returncode != 0:
            return None, (r.stderr or r.stdout)[-600:]
        return exe, None

    def start(self, version, n, pred, steps, threads, procs, weights):
        self.stop()
        with self.lock:
            if os.path.isdir(RUN_DIR):
                shutil.rmtree(RUN_DIR)
            self.reset_data(n)
            self.status, self.message = "compiling", f"Compiling {VERSIONS[version]} for N = {n}..."
        threading.Thread(target=self._start, args=(version, n, pred, steps, threads, procs, weights),
                         daemon=True).start()

    def _start(self, version, n, pred, steps, threads, procs, weights):
        exe, err = self.build(version, n, pred, steps)
        if err:
            with self.lock:
                self.status, self.message = "error", "Compile failed: " + err
            return
        if os.path.isdir(RUN_DIR):
            shutil.rmtree(RUN_DIR)
        os.makedirs(RUN_DIR)
        write_controls(weights)
        env = dict(os.environ, OMP_NUM_THREADS=str(threads), OMP_SCHEDULE="dynamic")
        cmd = [exe] if version in ("omp", "cuda") else ["mpirun", "--oversubscribe", "-np", str(procs), exe]
        self.log = open(os.path.join(RUN_DIR, "output.txt"), "w")
        with self.lock:
            self.proc = subprocess.Popen(cmd, cwd=RUN_DIR, env=env, stdout=self.log, stderr=subprocess.STDOUT)
            self.status = "running"
            how = f"{threads} threads" if version in ("omp", "cuda") else f"{procs} processes x {threads} threads"
            self.message = f"Running {VERSIONS[version]}, N = {n}, {how}"

    def stop(self):
        with self.lock:
            if self.proc and self.proc.poll() is None:
                self.proc.terminate()
                self.proc.wait()
                self.status, self.message = "stopped", "Stopped."

    # ---- reading output as it appears

    def _new_lines(self, name):
        path = os.path.join(RUN_DIR, name)
        if not os.path.exists(path):
            return []
        with open(path) as f:
            f.seek(self.offsets[name])
            chunk = f.read()
            self.offsets[name] = f.tell()
        text = self.partial[name] + chunk
        lines = text.split("\n")
        self.partial[name] = lines.pop()          # last piece may be an unfinished line
        return [l for l in lines if l and not l.startswith("step")]

    def poll(self):
        with self.lock:
            for line in self._new_lines("frames.csv"):
                v = line.split(",")
                self.pending.setdefault(int(v[0]), []).append([float(a) for a in v[1:]])
            for step in sorted(self.pending):
                if len(self.pending[step]) >= self.n:          # a frame is complete
                    self.frames.append(np.array(self.pending.pop(step)))
                    self.steps.append(step)
            for line in self._new_lines("density.csv"):
                v = np.array(line.split(",")[1:], dtype=float)
                g = int(np.sqrt(len(v)))
                self.dens.append(v.reshape(g, g))
            mpath = os.path.join(RUN_DIR, "metrics.csv")
            if os.path.exists(mpath):
                try:
                    self.metrics = pd.read_csv(mpath)
                except (pd.errors.EmptyDataError, pd.errors.ParserError):
                    pass
            if self.status == "running" and self.proc and self.proc.poll() is not None:
                ok = self.proc.returncode == 0
                self.status = "finished" if ok else "error"
                if ok:
                    self.message = f"Finished: {len(self.frames)} frames. Drag the slider to replay."
                else:
                    out = open(os.path.join(RUN_DIR, "output.txt")).read()
                    self.message = "The program stopped with an error: " + out.strip().splitlines()[0] if out.strip() else "The program stopped with an error."


SIM = Sim()


def write_controls(w):
    os.makedirs(RUN_DIR, exist_ok=True)
    tmp = os.path.join(RUN_DIR, "controls.tmp")
    with open(tmp, "w") as f:
        f.write(f"{w['align']} {w['coh']} {w['sep']}\n")
    os.replace(tmp, os.path.join(RUN_DIR, "controls.txt"))   # atomic: C never sees half a file


# ------------------------------------------------------------------ figures

def swarm_figure(k):
    fig = go.Figure()
    world = SIM.world
    if SIM.dens and k < len(SIM.dens):
        fig.add_trace(go.Heatmap(z=SIM.dens[k], x0=world / 128, dx=world / 64, y0=world / 128, dy=world / 64,
                                 colorscale=HEAT, showscale=False, zmin=0,
                                 zmax=max(1e-6, float(np.percentile(SIM.dens[k], 99.5))),
                                 hoverinfo="skip", zsmooth="best"))
    if SIM.frames and k < len(SIM.frames):
        f = SIM.frames[k]
        prey, pred = f[f[:, 4] == 0], f[f[:, 4] == 1]
        fig.add_trace(go.Scattergl(x=prey[:, 0], y=prey[:, 1], mode="markers", name="prey",
                                   marker=dict(size=4 if SIM.n <= 4000 else 2.5, color=BLUE), hoverinfo="skip"))
        fig.add_trace(go.Scattergl(x=pred[:, 0], y=pred[:, 1], mode="markers", name="predators",
                                   marker=dict(size=9, color=RED, symbol="triangle-up",
                                               line=dict(width=1, color="#fecdd3")), hoverinfo="skip"))
    fig.update_xaxes(range=[0, world], visible=False)
    fig.update_layout(template="plotly_dark")
    fig.update_yaxes(range=[0, world], visible=False, scaleanchor="x")
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), showlegend=False, plot_bgcolor=BG, paper_bgcolor=PANEL,
                      uirevision="keep", height=560)
    return fig


def metrics_figure(k):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Polarization (alignment, 0 to 1)", "Prey fleeing", "Catches (total)"))
    m = SIM.metrics
    if len(m):
        for row, col, color in ((1, "polarization", BLUE), (2, "fleeing", RED), (3, "catches", GREY)):
            fig.add_trace(go.Scatter(x=m["step"], y=m[col], mode="lines", line=dict(color=color, width=2),
                                     showlegend=False), row=row, col=1)
        if k < len(SIM.steps):
            for row in (1, 2, 3):
                fig.add_vline(x=SIM.steps[k], line=dict(color=TEXT, width=1, dash="dot"), row=row, col=1)
    fig.update_yaxes(range=[0, 1], row=1, col=1)
    fig.update_layout(height=560, margin=dict(l=40, r=10, t=30, b=30), **DARK)
    fig.update_annotations(font_size=13)
    return fig


DARK = dict(template="plotly_dark", paper_bgcolor=PANEL, plot_bgcolor=PANEL,
            font=dict(color=TEXT, family="Inter, system-ui, sans-serif"))


def card(title, ident):
    return html.Div([html.Div(title, style={"fontSize": 12, "color": MUTED}),
                     html.Div("-", id=ident, style={"fontSize": 22, "fontWeight": 600, "color": TEXT})],
                    style={"padding": "8px 14px", "background": PANEL, "border": f"1px solid {BORDER}",
                           "borderRadius": 10, "minWidth": 110})


def perf_figures():
    if not os.path.exists(BENCH):
        return [html.P("No results/bench.csv yet - run python3 bench.py first.")]
    b = pd.read_csv(BENCH)
    figs = []

    def style(fig, title, h=360):
        fig.update_layout(title=title, height=h, **DARK,
                          margin=dict(l=50, r=20, t=50, b=40), legend=dict(orientation="h", y=-0.2))
        fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER)
        fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER)
        return html.Div(dcc.Graph(figure=fig, config={"displaylogo": False}),
                        style={"background": PANEL, "border": f"1px solid {BORDER}", "borderRadius": 10,
                               "padding": 6, "margin": 6})

    s = b[b.experiment == "serial_scaling"]
    if len(s):
        fig = go.Figure(go.Scatter(x=s.n, y=s.ms_per_step, mode="lines+markers", name="measured", line=dict(color=BLUE)))
        ref = s.ms_per_step.iloc[0] * (s.n / s.n.iloc[0]) ** 2
        fig.add_trace(go.Scatter(x=s.n, y=ref, mode="lines", name="pure N²", line=dict(color=GREY, dash="dash")))
        fig.update_xaxes(title="agents (N)")
        fig.update_yaxes(title="ms per step")
        figs.append(style(fig, "1. Serial cost grows with N²"))

    t = b[b.experiment == "omp_threads"]
    if len(t):
        sp = t.ms_per_step.iloc[0] / t.ms_per_step
        fig = go.Figure(go.Scatter(x=t.threads, y=sp, mode="lines+markers", name="measured", line=dict(color=BLUE)))
        fig.add_trace(go.Scatter(x=t.threads, y=t.threads, mode="lines", name="ideal", line=dict(color=GREY, dash="dash")))
        fig.update_xaxes(title="OpenMP threads")
        fig.update_yaxes(title="speedup (×)")
        figs.append(style(fig, "2. OpenMP speedup, N = 4000"))

    sc = b[b.experiment == "schedules"]
    if len(sc):
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=sc.schedule, y=sc.ms_per_step, name="ms per step", marker_color=BLUE), secondary_y=False)
        fig.add_trace(go.Scatter(x=sc.schedule, y=sc.imbalance, name="load imbalance", mode="markers+lines",
                                 marker=dict(color=RED, size=10)), secondary_y=True)
        fig.update_yaxes(title="ms per step", secondary_y=False)
        fig.update_yaxes(title="imbalance (1 = perfect)", secondary_y=True)
        figs.append(style(fig, "3. Scheduling strategies, 12 threads"))

    g = b[b.experiment == "cpu_vs_gpu"]
    if len(g):
        fig = go.Figure()
        for ver, name, color in (("omp", "CPU, 12 threads", GREY), ("cuda", "GPU + CPU", BLUE)):
            d = g[g.version == ver]
            fig.add_trace(go.Bar(x=d.n.astype(str), y=d.ms_per_step, name=name, marker_color=color,
                                 text=d.ms_per_step.round(1), textposition="outside"))
        fig.update_xaxes(title="agents (N)")
        fig.update_yaxes(title="ms per step (log scale)", type="log", dtick=1)
        figs.append(style(fig, "4. CPU vs GPU (log scale)"))

    mp = b[b.experiment == "mpi_procs"]
    if len(mp):
        comm = mp.comm_s.fillna(0)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=mp.procs.astype(str), y=(mp.total_s - comm) * 1000 / 100, name="computation", marker_color=BLUE))
        fig.add_trace(go.Bar(x=mp.procs.astype(str), y=comm * 1000 / 100, name="communication + waiting", marker_color=RED))
        fig.update_layout(barmode="stack")
        fig.update_xaxes(title="MPI processes (1 thread each)")
        fig.update_yaxes(title="ms per step")
        figs.append(style(fig, "5. MPI: computation vs communication, N = 4000"))

    h = b[b.experiment == "hybrid"]
    if len(h):
        h = h.copy()
        h["label"] = h.apply(lambda r: ("CPU only " if r.version == "mpi" else "hybrid ") + f"{r.procs}×{r.threads}", axis=1)
        fig = go.Figure()
        for n, color in ((4000, "#1e6fa3"), (16000, BLUE)):
            d = h[h.n == n]
            fig.add_trace(go.Bar(x=d.label, y=d.ms_per_step, name=f"N = {n}", marker_color=color,
                                 text=d.ms_per_step.round(1), textposition="outside"))
        fig.update_yaxes(title="ms per step")
        figs.append(style(fig, "6. Hybrid configurations (processes × threads)"))

    grid = [html.Div(f, style={"width": "50%", "display": "inline-block", "verticalAlign": "top"}) for f in figs]
    return grid


# ------------------------------------------------------------------ layout

app = Dash(__name__, title="SwarmScale")
app.index_string = """<!DOCTYPE html>
<html>
<head>
{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>
:root {
  --Dash-Fill-Interactive-Strong: """ + ACCENT + """;
  --Dash-Fill-Interactive-Weak: rgba(129, 140, 248, 0.12);
  --Dash-Fill-Inverse-Strong: """ + PANEL + """;
  --Dash-Fill-Primary-Hover: rgba(148, 163, 184, 0.10);
  --Dash-Fill-Primary-Active: rgba(148, 163, 184, 0.18);
  --Dash-Fill-Disabled: rgba(148, 163, 184, 0.15);
  --Dash-Stroke-Strong: #3b4a66;
  --Dash-Stroke-Weak: """ + BORDER + """;
  --Dash-Text-Primary: """ + TEXT + """;
  --Dash-Text-Strong: #f8fafc;
  --Dash-Text-Weak: """ + MUTED + """;
  --Dash-Text-Disabled: #64748b;
  --Dash-Tooltip-Background-Color: #1e293b;
  --Dash-Tooltip-Border-Color: """ + BORDER + """;
}
body { background: """ + BG + """; color: """ + TEXT + """; margin: 0; }
input { background: """ + PANEL + """; color: """ + TEXT + """; border: 1px solid """ + BORDER + """; }
hr { border: 0; border-top: 1px solid """ + BORDER + """; }
button { cursor: pointer; }
input[type=radio], input[type=checkbox] { accent-color: """ + ACCENT + """; }
.preset { background: #1e293b; color: """ + TEXT + """; border: 1px solid """ + BORDER + """; border-radius: 6px; padding: 4px 8px; }
.preset:hover { border-color: """ + ACCENT + """; }
::-webkit-scrollbar { width: 10px; } ::-webkit-scrollbar-thumb { background: #22304a; border-radius: 5px; }
</style>
</head>
<body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body>
</html>"""
label = {"fontSize": 13, "fontWeight": 600, "marginTop": 14, "marginBottom": 4, "color": TEXT}

controls = html.Div([
    html.Div("Version", style=label),
    dcc.RadioItems(id="version", options=[{"label": v, "value": k} for k, v in VERSIONS.items()],
                   value="omp", labelStyle={"display": "block", "fontSize": 13}),
    html.Div("Agents (N)", style=label),
    dcc.Dropdown(id="n", options=[1000, 2000, 4000, 8000, 16000], value=2000, clearable=False),
    html.Div("Predators (% of agents)", style=label),
    dcc.Slider(id="pred", min=0, max=10, step=1, value=2, marks={0: "0", 2: "2", 5: "5", 10: "10"}),
    html.Div("Threads per process", style=label),
    dcc.Slider(id="threads", min=1, max=12, step=1, value=12, marks={1: "1", 4: "4", 8: "8", 12: "12"}),
    html.Div("MPI processes", style=label),
    dcc.Dropdown(id="procs", options=[1, 2, 4], value=2, clearable=False),
    html.Div("Steps", style=label),
    dcc.Dropdown(id="steps", options=[500, 1000, 2000, 5000], value=2000, clearable=False),
    html.Hr(),
    html.Div("Rules (change them while it runs)", style=label),
    html.Div("Alignment", style={"fontSize": 12}),
    dcc.Slider(id="align", min=0, max=0.5, step=0.01, value=0.10, marks={0: "0", 0.1: "0.1", 0.5: "0.5"},
               tooltip={"placement": "bottom"}),
    html.Div("Cohesion", style={"fontSize": 12}),
    dcc.Slider(id="coh", min=0, max=0.1, step=0.005, value=0.02, marks={0: "0", 0.02: "0.02", 0.1: "0.1"},
               tooltip={"placement": "bottom"}),
    html.Div("Separation", style={"fontSize": 12}),
    dcc.Slider(id="sep", min=0, max=2, step=0.05, value=0.5, marks={0: "0", 0.5: "0.5", 2: "2"},
               tooltip={"placement": "bottom"}),
    html.Div([html.Button(t, id=f"preset-{k}", n_clicks=0, className="preset", style={"margin": "3px", "fontSize": 12})
              for k, t in (("flock", "Flock"), ("avoid", "Avoid everyone"), ("loose", "Loose swarm"),
                           ("tight", "Tight flock"))]),
    html.Hr(),
    html.Button("▶ Run", id="run", n_clicks=0,
                style={"width": "48%", "padding": 8, "background": ACCENT, "color": "#0b1120", "fontWeight": 600,
                       "border": 0, "borderRadius": 6}),
    html.Button("■ Stop", id="stop", n_clicks=0,
                style={"width": "48%", "padding": 8, "marginLeft": "4%", "background": "#1e293b", "color": TEXT,
                       "border": f"1px solid {BORDER}", "borderRadius": 6}),
], style={"width": 270, "padding": 16, "background": PANEL, "borderRight": f"1px solid {BORDER}", "flexShrink": 0,
          "minHeight": "calc(100vh - 110px)"})

TAB = {"background": BG, "color": MUTED, "border": "none", "borderBottom": f"1px solid {BORDER}", "padding": 10}
TAB_ON = {"background": BG, "color": TEXT, "border": "none", "borderBottom": f"2px solid {ACCENT}",
          "fontWeight": 600, "padding": 10}

swarm_tab = html.Div([
    controls,
    html.Div([
        html.Div(id="status", style={"fontSize": 13, "color": MUTED, "marginBottom": 8}),
        html.Div([card("Step", "c-step"), card("Polarization", "c-pol"), card("Fleeing prey", "c-flee"),
                  card("Catches", "c-catch"), card("ms per step", "c-ms")],
                 style={"display": "flex", "gap": 10, "marginBottom": 10, "flexWrap": "wrap"}),
        html.Div([
            html.Div(dcc.Graph(id="swarm", config={"displaylogo": False}),
                     style={"flex": "1 1 560px", "minWidth": 400, "border": f"1px solid {BORDER}", "borderRadius": 10,
                            "overflow": "hidden"}),
            html.Div(dcc.Graph(id="metrics", config={"displaylogo": False}),
                     style={"flex": "1 1 380px", "minWidth": 320, "border": f"1px solid {BORDER}", "borderRadius": 10,
                            "overflow": "hidden"}),
        ], style={"display": "flex", "gap": 10, "flexWrap": "wrap"}),
        dcc.Slider(id="frame", min=0, max=0, step=1, value=0, marks=None, updatemode="drag"),
        dcc.Checklist(id="follow", options=[{"label": " follow the latest frame", "value": "on"}], value=["on"],
                      style={"fontSize": 13, "color": MUTED}),
    ], style={"flex": 1, "padding": 16, "minWidth": 0}),
], style={"display": "flex"})

app.layout = html.Div([
    html.Div([html.Span("Swarm", style={"fontWeight": 700, "fontSize": 20, "color": TEXT}),
              html.Span("Scale", style={"fontWeight": 700, "fontSize": 20, "color": ACCENT}),
              html.Span("   hybrid MPI + CUDA + OpenMP swarm simulator", style={"color": MUTED, "fontSize": 14})],
             style={"padding": "12px 16px", "background": PANEL, "borderBottom": f"1px solid {BORDER}"}),
    dcc.Tabs([
        dcc.Tab(label="Swarm", children=swarm_tab, style=TAB, selected_style=TAB_ON),
        dcc.Tab(label="Performance", children=html.Div(id="perf", style={"padding": 10}),
                style=TAB, selected_style=TAB_ON),
    ], id="tabs"),
    dcc.Interval(id="tick", interval=400),
], style={"fontFamily": "Inter, system-ui, sans-serif", "background": BG, "minHeight": "100vh"})


# ------------------------------------------------------------------ callbacks

@app.callback(Output("align", "value"), Output("coh", "value"), Output("sep", "value"),
              Input("preset-flock", "n_clicks"), Input("preset-avoid", "n_clicks"),
              Input("preset-loose", "n_clicks"), Input("preset-tight", "n_clicks"), prevent_initial_call=True)
def preset(*_):
    p = PRESETS[ctx.triggered_id.split("-")[1]]
    return p["align"], p["coh"], p["sep"]


@app.callback(Output("status", "children", allow_duplicate=True),
              Input("align", "value"), Input("coh", "value"), Input("sep", "value"), prevent_initial_call=True)
def live_rules(a, c, s):
    if SIM.status == "running":
        write_controls(dict(align=a, coh=c, sep=s))
    return no_update


@app.callback(Output("status", "children", allow_duplicate=True),
              Input("run", "n_clicks"), Input("stop", "n_clicks"),
              State("version", "value"), State("n", "value"), State("pred", "value"), State("steps", "value"),
              State("threads", "value"), State("procs", "value"),
              State("align", "value"), State("coh", "value"), State("sep", "value"), prevent_initial_call=True)
def run_stop(_r, _s, version, n, pred, steps, threads, procs, a, c, s):
    if ctx.triggered_id == "stop":
        SIM.stop()
    else:
        SIM.start(version, n, pred, steps, threads, procs, dict(align=a, coh=c, sep=s))
    return SIM.message


@app.callback(Output("frame", "max"), Output("frame", "value"), Output("status", "children"),
              Output("c-step", "children"), Output("c-pol", "children"), Output("c-flee", "children"),
              Output("c-catch", "children"), Output("c-ms", "children"),
              Input("tick", "n_intervals"), State("follow", "value"), State("frame", "value"))
def tick(_, follow, k):
    SIM.poll()
    last = max(0, len(SIM.frames) - 1)
    k = last if follow else min(k, last)
    m = SIM.metrics
    if len(m):
        row = m.iloc[-1]
        cards = (int(row.step), f"{row.polarization:.2f}", int(row.fleeing), int(row.catches), f"{row.ms_per_step:.1f}")
    else:
        cards = ("-",) * 5
    return (last, k, SIM.message) + cards


@app.callback(Output("swarm", "figure"), Output("metrics", "figure"), Input("frame", "value"))
def draw(k):
    return swarm_figure(k or 0), metrics_figure(k or 0)


@app.callback(Output("perf", "children"), Input("tabs", "value"))
def perf(_):
    return perf_figures()


if __name__ == "__main__":
    print("SwarmScale dashboard: open http://localhost:8050")
    app.run(debug=False, host="127.0.0.1", port=8050)
