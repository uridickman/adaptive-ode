from examples import *
from pathlib import Path

Path("./figs").mkdir(parents=True, exist_ok=True)

solve_constant_h()

solve_PredatorPrey()

solve_VanDerPol()

solve_MoL()