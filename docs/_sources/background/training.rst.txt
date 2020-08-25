:todo:

Training Strategies
###################


Variance Loss
*************

As defined in the :doc:`normalizing flows <nis>` section, our model consists of

1. a PDF over the latent space
2. a trainable bijection from the latent space to the target space.

Together they allow us to sample points :math:`x` from the model distribution :math:`q(x)` which is
also known for every sampled point.

Our goal is to maximize the integration speed of our integral estimator, i.e. to find the :math:`q` that minimizes

.. math::

    \underset{x\sim q(x)}{\sigma} \left[\frac{f(x)}{q(x)}\right] =\int dx q(x) \left( \left(\frac{f(x)}{q(x)} \right)^2 - I^2\right),

Where :math:`I` is our desired integral. Note that, because :math:`q` is a normalized PDF,
the second term in the integral is independent of it and we can limit ourselves to minimizing the first term only:

.. math::

    {\cal L} = \int dx q(x) \left(\frac{f(x)}{q(x)}\right)^2.

As an integral this is not a tractable loss function defined on a sample of points, we must build an estimator
for it, and the multiple possibilities yield different ways of training the model

Forward Training
================

The most straightforward way to formulate an estimator for the loss :math:`{\cal L}` is to take it at face value
as an expectation value over :math:`q`:

.. math::

    {\cal L} = \underset{x \sim q(x)}{\mathbb{E}} \left[\left(\frac{f(x)}{q(x)}\right)^2\right]

We can therefore sample a collection of points :math:`\left\{x_i\right\}_{i=1\dots N}` from our model,
which will be distributed according to :math:`q` and build the estimator

.. math::

    \hat{\cal L}_\text{forward} = \frac{1}{N} \sum_{i=0}^N \left(\frac{f(x_i)}{q(x_i)}\right)^2

of which we can compute the gradient with respects to the parameters of :math:`q` and use a standard optimization
technique to attempt reaching a minimum. Note that there are actually two sources of dependence on :math:`q`:
the first is the explicit PDF in the denominator, and the second is in each actual point :math:`x_i`,
which is obtained by sampling in latent space and mapping them with our model.

A more explicit way of formulating this training strategy is therefore that we sample points
:math:`\left\{y_i\right\}_{i=1\dots N}` in latent space from the latent space PDF :math:`q_y` and map them to a set
:math:`\left\{x_i\right\}_{i=1\dots N}` of points in latent space using our transformation :math:`Q` and evaluate

.. math::

    \hat{\cal L}_\text{forward} = \frac{1}{N} \sum_{i=0}^N \left(\frac{f\left(Q\left(y_i\right)\right)}{q(Q(y_i))}\right)^2

While this method is the most straightforward, it carries several downsides

1. It is susceptible to the initialization of the model. If :math:`q` is poorly sampled, it could avoid exploring relevant regions.
2. It requires resampling new points and re-evaluate the function at each gradient step.

On the other hand, once a decent model has been learned, this approach ensures that most point being sampled
are in the relevant regions where the function is enhanced, thus ensuring good end-time performance.

Backward Training
=================

As a solution to the drawbacks of the forward training method, we formulate an alternative approach in which we reinterpret the loss integral. Let us consider a different PDF :math:`p` over the target space, then

.. math::

    {\cal L} = \int dx q(x) \left(\frac{f(x)}{q(x)}\right)^2 = \int dx p(x) \frac{f(x)^2}{p(x)q(x)},

which we now interpret as a different expectation value:

.. math::

    {\cal L} = \underset{x \sim p(x)}{\mathbb{E}} \left[\frac{f(x)^2}{p(x)q(x)}\right]

For which an estimator is constructed by sampling a collection of points :math:`\left\{x_i\right\}_{i=1\dots N}` from :math:`p` and evaluating

.. math::

    \hat{\cal L}_\text{backward} = \frac{1}{N} \sum_{i=0}^N \frac{f(x_i)^2}{p(x_i)q(x_i)}

Now the sample of points is independent from :math:`q` and we can therefore

1. ensure both that our distribution :math:`p` has a good coverage over the whole space
2. run multiple gradient descent steps using the same batch of points

Note that another practical advantage of this approach is that it yields a simpler computational graph
for the loss function, leading to a reduced memory usage at training time.

From which distribution should we :math:`p` sample? In practice, we use two standard choices:

1. a uniform distribution, which ensures that all corners of the integration domain are covered
2. a frozen copy of the normalizing flow.

The second option can be thought of a similar to the two version of the state-action value model
used in deep-Q learning. When sampling, we freeze the weights of the model and think of it a just any other
PDF on target space :math:`p(x)` and draw a collection of points from it. We then keep training for a while on this sample,
meaning that the sample becomes progressively less representative of the distribution defined by the model.
Nevertheless, as long as this distribution does not veer too far off the evolving model, it is likely to provide
a good estimate of the ideal loss integral.

Adaptive Backward Training
==========================

The description of the two possible PDFs used for sampling point datasets for backward training should make it clear
that there is a "best of both worlds" options: use uniform sampling at the beginning of training, where the model
is random and possibly poorly conditioned to evaluate the integal, and later switch to sampling from the frozen model
after it has sufficiently improved.

The strategy that we use to time the switch between the two sampling PDFs is to compare the current loss to the loss
that we would obtain replacing our model by a uniform model:

.. math::
    x_i &\sim \text{Uniform}(x)\\
    \hat{\cal L}_\text{backward}^\text{flat model} &= \frac{1}{N} \sum_{i=0}^N f(x_i)^2

If the actual loss is smaller than this quantity, then our model does a better job than the flat distribution
at estimating the integral and we therefore switch sampling mode.


Kullback-Leibler Distance Loss
******************************

A commonly used loss for normalizing flows is the `Kullback-Leiber divergence`_ (:math:`D_\text{KL}`), which is an
information-theoretic distance measure between probability distribution. For two PDFs :math:`p` and :math:`q`,
the :math:`D_\text{KL}` is defined as

.. math::
    D_\text{KL}(p|q) = \int dx p(x) \log \frac{p(x)}{q(x)},

which has a minimum when :math:`p=q` as can be easily shown.

In our case, we do not actually have the target PDF, but we the target function :math:`f`, which is un-normalized. The target
PDF is actually :math:`p(x) = f(x)/I`, where :math:`I` is our desired integral. We do, however not need to know the value of :math:`I` to optimize
our model for this loss:

.. math::
    D_\text{KL}(p|q) &= \int dx \frac{f(x)}{I} \log \frac{p(x)}{I} - \frac{f(x)}{I} \log q(x)\\
    &\propto   - \int dx f(x) \log q(x) + \text{terms independent on }q

While the true minimum of the :math:`D_\text{KL}` loss is the same as the variance loss, they do yield
different practical results. It should be clear that the variance should be the standard choice for
the typical user: it optimizes directly the metric that controls the convergence speed of the integrand
estimator. If one compares the variance loss and the :math:`D_\text{KL}` loss, it appears that the variance loss
gives relatively more weight to points where $f$ is very large - which is sensible due to how these affect the
integral estimates. This means that, for practical applications, it is more likely for models trained using the
:math:`D_\text{KL}` loss to correctly approximate the desired PDF in regions where the function is smaller.
This is less-than-optimal for direct integral estimation, but can have useful applications, especially if one wants to
re-use models trained on the full domain to compute integrals on limited sub-regions,
as can be the case in High-Energy Physics when one considers loose- and tight-cut observables.

The same discussion as for the variance loss can be had for converting the integral loss to an estimator defined
on an estimator defined on a sample of point: we can define forward training by sampling points from the model itself
or backward training by sampling in target space using an arbitrary PDF. Adaptive backward training can of course
also be realized, all the more easier since the switching condition corresponds to testing the sign of the loss:
if the model were a flat distribution, it would have unit PDF and therefore 0 loss.

.. _Kullback-Leiber divergence: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence