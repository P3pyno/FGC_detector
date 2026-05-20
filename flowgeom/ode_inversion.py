from __future__ import annotations
import numpy as np
import torch


def invert_trajectory(z1: torch.Tensor, velocity_model, num_steps: int = 30, solver: str = "euler", t_start: float = 1.0, t_end: float = 0.0):
    ts = np.linspace(t_start, t_end, num_steps + 1)
    z = z1.clone()
    latents = [z.clone()]
    velocities = []
    for i in range(num_steps):
        t, t_next = float(ts[i]), float(ts[i + 1])
        dt = t_next - t
        v = velocity_model.velocity(z, t)
        if solver == "heun":
            z_pred = z + dt * v
            v2 = velocity_model.velocity(z_pred, t_next)
            z = z + 0.5 * dt * (v + v2)
        else:
            z = z + dt * v
        latents.append(z.clone())
        velocities.append(v.clone())
    return {"latents": latents, "timesteps": ts.tolist(), "velocities": velocities}
