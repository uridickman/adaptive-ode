from examples import *
from pathlib import Path
import os

if __name__ == "__main__":

    Path("./figs").mkdir(parents=True, exist_ok=True)

    print("Solving example with constant timestep...")
    solve_constant_h()

    print("Solving Predator-Prey problem...")
    solve_PredatorPrey()

    print("Solving Van Der Pol Oscillator...")
    solve_VanDerPol()

    print("Solving Method of Lines upwind advection with timestep interpolation...")
    solve_MoL()

    print(f"Saved figures to {os.path.join(os.getcwd(),"figs")}")