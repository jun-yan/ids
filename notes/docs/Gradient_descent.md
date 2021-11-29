---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.0
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Gradient descent
- Batch GD & Mini-batch GD
- GD with momentum
- Adam

## Batch GD & Mini-batch GD
The vanilla GD updates the parameters for entire dataset:

$$
\theta = \theta - \eta\nabla_\theta J(\theta)
$$

SGD updates the parameters for each training examples.

$$
\theta = \theta - \eta\nabla_\theta J(\theta;x^{(i)};y^{(i)})
$$

Mini-batch GD performs updates for every mini-batch.

$$
\theta = \theta - \eta\nabla_\theta J(\theta;x^{(i:i+n)};y^{(i:i+n)})
$$

The Batch GD converges to a local minimum for non-convex problems and slow when the size of data is large.\
The SGD updates frequently with a high variance, causing the lost function fluctuate heavily.\
Mini-batch GD takes the advantage of both. Note that in training neural network models, term SGD is usually employed.\
**Challenges**:  Saddle points are usually surrounded by a plateau of the same error, which makes it notoriously hard for SGD to escape, as the gradient is close to zero in all dimensions.

## GD with momentum
The momentum term increase for dimensions whose gradients point in the same directions and reduces updates for dimensions whose gradients change directions.\
"Ball runing down a surface"\
We we gain faster convergence.
$$
v_t = \gamma v_{t-1}+ \eta\nabla_\theta J(\theta) \\
\theta = \theta - v_t
$$

## Adam
- Adagrad: adapts the learning rate
- Adadelta: extension of Adagrad, more conservative 
- Adam: Adaptive Moment Estimation
Previously, we updates for $\theta$ using the same learning rate $\eta$.\
The adaptive learning rate is changing with the second-order moment of gradient.\
"Ball with friction"
$$
m_t = \beta_1 m_{t-1}+(1-\beta_1)g_t\\
v_t = \beta_2{t-1}+(1-\beta_2)g_t^2
$$

The $m_t,v_t$ serve as estimates for the gradient $g_t$. With some bias correction, the updates is performed as follow:

$$
\theta = \theta - \frac{\eta}{\sqrt{\hat{v}_t}+\epsilon}\hat{m}_t
$$


