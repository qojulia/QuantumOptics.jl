# Time-evolution


**QuantumOptics.jl** implements various solvers to simulate the dynamics of closed as well as open quantum systems. The interfaces are designed to be as consistent as possible which makes it easy to switch between different methods. Internally, all of them rely on [**DifferentialEquations.jl**](https://github.com/SciML/), which is a very rich numerical library for the solution of differential equations. It offers quite a few options for the user to tailor the solver to their specific needs. The defaults are chosen for to suit commonly encountered problems and should work fine for most use cases. If you require more specialized methods, such as the choice of algorithm, please refer to the documentation of **DifferentialEquations.jl**, which can be found [here](http://docs.juliadiffeq.org/stable/).

```@example
using QuantumOptics # hide
b = FockBasis(2) # hide
H = number(b) # hide
J = [destroy(b)] # hide
T = [0,0.1] # hide
psi0 = fockstate(b, 0) # hide
tout, psi_t = timeevolution.schroedinger(T, psi0, H)
tout, rho_t = timeevolution.master(T, psi0, H, J)
tout, psi_t = timeevolution.mcwf(T, psi0, H, J)
nothing # hide
```
