# adaptive-ode

*Description*: A solver for non-stiff Ordinary Differential Equations of the form

```math
\begin{align*}
y' &=f(t,y(t)) \\
y(0) &= y_0
\end{align*}
```

Uses a second order Predictor-Corrector method (PECE) with adaptive step sizes for error control, with Adams-Bashforth as the predictor and Adams-Moulton as the corrector.

---

Run the examples with:

```python
python main.py
```