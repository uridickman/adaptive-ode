# adaptive-ode

*Description*: A solver for non-stiff Ordinary Differential Equations of the form

$$
\mathbf{y}' = \mathbf{f}(t,\mathbf{y}), \qquad \mathbf{y}(t_0)=\mathbf{y}_0
$$

Uses a second order Predictor-Corrector method (PECE) with adaptive step sizes for error control, with Adams-Bashforth as the predictor and Adams-Moulton as the corrector.

---

Install the required packages in a virtual environment with:

```bash
uv venv /path/to/venv --python 3.14 && source /path/to/venv/bin/activate
uv pip install -r requirements.txt
```

Run the examples with:

```bash
python main.py
```