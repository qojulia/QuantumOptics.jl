<img src="https://github.com/qojulia/QuantumOptics.jl-website/blob/master/src/images/logo.png" alt="QuantumOptics.jl" width="400">

#

![Build_state](https://github.com/qojulia/QuantumOptics.jl/workflows/CI/badge.svg) [![Stable documentation][docs-img]][docs-url]

**QuantumOptics.jl** is a numerical framework written in [Julia] that makes it easy to simulate various kinds of quantum systems. It is inspired by the [Quantum Optics Toolbox] for MATLAB and the Python framework [QuTiP].

More information, documentation and examples can be found on our website http://qojulia.org.


**Latest release:**
  * Version: [![Latest version tag][version-img]][version-url]
  * Test coverage:
        [![Test coverage status on codecov][codecov-img]][codecov-url]


### Project structure

The source content associated with **QuantumOptics.jl** is distributed over several repositories under the [qojulia] organization on github:

* The main code: https://github.com/qojulia/QuantumOptics.jl
* Documentation: https://github.com/qojulia/QuantumOptics.jl-documentation
* Examples: https://github.com/qojulia/QuantumOptics.jl-examples
* Benchmarks: https://github.com/qojulia/QuantumOptics.jl-benchmarks
* Website: https://github.com/qojulia/QuantumOptics.jl-website


### Discussion and Help

For questions, support, and discussions about QuantumOptics.jl, please use the [GitHub issue tracker](https://github.com/qojulia/QuantumOptics.jl/issues) for bug reports and feature requests. For general discussions and help, visit the [Julia Discourse forum under the quantum topic](https://discourse.julialang.org/c/domain/quantum/52). You can also join the `#quantum-optics` channel on the [Julia Slack](https://julialang.org/slack/) for real-time discussions.

Contributions of any kind are always welcome! Be it as ideas for new features, bug reports or, our favorite case, sending pull requests.

### Citing

If you like **QuantumOptics.jl**, we would appreciate it if you starred the repository in order to help us increase its visibility. Furthermore, if you find the framework useful in your research, we would be grateful if you could cite our [publication](https://www.sciencedirect.com/science/article/pii/S0010465518300328) using the following bibtex entry:

```bib
@article{kramer2018quantumoptics,
  title={QuantumOptics. jl: A Julia framework for simulating open quantum systems},
  author={Kr{\"a}mer, Sebastian and Plankensteiner, David and Ostermann, Laurin and Ritsch, Helmut},
  journal={Computer Physics Communications},
  volume={227},
  pages={109--116},
  year={2018},
  publisher={Elsevier}
}
```

[Julia]: http://julialang.org
[qojulia]: https://github.com/qojulia
[Quantum Optics Toolbox]: http://qo.phy.auckland.ac.nz/toolbox
[QuTiP]: http://qutip.org

[codecov-url]: https://codecov.io/gh/qojulia/QuantumOptics.jl
[codecov-img]: https://codecov.io/gh/qojulia/QuantumOptics.jl/branch/master/graph/badge.svg


[docs-url]: https://docs.qojulia.org/
[docs-img]: https://img.shields.io/badge/docs-stable-blue.svg

[version-url]: https://github.com/qojulia/QuantumOptics.jl/releases
[version-img]: https://img.shields.io/github/release/qojulia/QuantumOptics.jl.svg
