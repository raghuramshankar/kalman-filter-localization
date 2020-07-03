[![Total alerts](https://img.shields.io/lgtm/alerts/g/raghuramshankar/kalman-filter-localization.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/raghuramshankar/kalman-filter-localization/alerts/)

# Kalman Filter

Implementation of sensor fusion using Kalman Filters for Localization of Autonomous Vehicles.

## Kalman Filter:

The Kalman Filter is widely regarded as the best solution to the optimal filtering problem. When working with linear and gaussian variables, the Kalman Filter is able to provide the best possible estimate of the concerned state variables while accounting for uncertainties, noise and delays in the system. This is also possible when the state variables are internal and not directly measurable, but the physical relationships of the system are known through the motion model and the measurement model. 

The Kalman Filter algorithm uses the following steps:

1. Prediction: 
   Assuming that the plant is a Markov process, ie, the future is only dependant on the present and not the past, we can predict the values of the state variables in the next time step using information from the current time step based on the known motion model.

2. Update: 
    Assuming Gaussian variables, we can incorporate the new information from the current measurement and compute the Kalman gain, Innovation and Innovation Covariance, which along with knowledge of the measurement model, allows us to correct our prediction of the state variables. The Kalman gain provides us a way to quantify how much we trust our prediction or measurement after tuning the process noise accordingly.
    
## Linear Kalman Filter:

![Open Notebook](https://github.com/raghuramshankar/kalman-filter-localization/blob/master/jupyter/KF-CA-Linear.ipynb)

## Extended Kalman Filter:

to be updated

## Cubature Kalman Filter:

to be updated
