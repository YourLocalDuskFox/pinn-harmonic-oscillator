# Physics-Informed Neural Network — Simple Harmonic Oscillator

A PyTorch network that solves d²x/dt² + ω²x = 0 with just the equation as the loss. Built for INFO 4160 Industrial Internet of Things.

## What it does

The network takes time t and outputs displacement x(t). Loss has a physics residual that penalizes violations of the ODE and an initial condition loss anchoring x(0) = 0, dx/dt(0) = 1. Autograd finds the second derivative of the output to make the residual term possible. After 10,000 epochs the prediction matches the analytical solution x(t) = v₀/ω · sin(ωt).

## Stack

- `torch` for the network and autograd-based derivatives
- `numpy` and `matplotlib` for the analytical comparison and plotting
- Architecture: 1 → 128 → 128 → 1, tanh activations
- Trained for 10,000 epochs

## Running it

```bash
pip install -r requirements.txt
jupyter notebook MP2PINN.ipynb
```

Run all cells. The last two plots show the PINN prediction alone, then overlaid against the analytical solution.

## What I struggled with

I started with ReLU and it wasn't able to learn the oscillation. When I switched to tanh and it worked. This is because ReLU's second derivative is zero so the physics loss has no gradient to push against. Tanh actually carries signal back to the weights.

## Files

```
MP2PINN.ipynb     full notebook — network definition, loss, training, plots
```

