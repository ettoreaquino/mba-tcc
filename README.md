# Content

## Mathematics

### Unit Root Test

> Tests whether a time series is not stationary.

The presence of a unit root in time series defines the null hypothesis, with the alternative hypothesis defining the time series as **stationary**

$$y_t = D_t + z_t + \varepsilon_t$$

Where,

- $D_t$: Deterministic component
- $z_t$: Stochastic component
- $\varepsilon_t$: Stationary error process

### Dickey-Fuller

A simple Auto-regressive model can be represented as:

$y_t = \rho y_{t-1} + \varepsilon_t$

Where,

- $y_t$: Variable of interest at time $t$
- $\rho$: Coefficient that defines the unit root
- $\varepsilon_t$: Noise (or error) term.

Observe that, if $\rho = 1$ then:

$$y_t = y_{t-1} + \varepsilon_t$$
$$y_t - y_{t-1} = \varepsilon_t$$

Which means that the difference between to periods of the series is only accounted as an noise element.

In t he most intuitive sense, stationarity means that the statistical properties of a process generating a time series do not change over time . It does not mean that the series does not change over time, just that the way it changes does not itself change over time
