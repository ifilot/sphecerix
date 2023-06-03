.. _background:
.. index:: Background

Background
==========

Wigner-D matrices
-----------------

The canonical spherical harmonics :math:`Y_{lm}` subject to a rotation in 
:math:`\mathbb{R}^{3}` will always mix among each other. As such, we can
represent the act of rotating the spherical harmonics by the following
matrix-vector equation

.. math::
  \vec{Y}_{l}\prime = \mathbf{D}\vec{Y}_{l}

wherein :math:`\mathbf{D}` is the `Wigner D-matrix <https://en.wikipedia.org/wiki/Wigner_D-matrix>`_
and :math:`\vec{Y}_{l}` a vector composed of the canonical spherical harmonics
of order :math:`l`. The vector :math:`\vec{Y}_{l}\prime` is the linear combination that
represents the result of the rotation upon the (linear combination) of
spherical harmonics prior to the rotation.

Tesseral transformation
-----------------------

The canonical spherical harmonics are complex-valued by nature, yet a transformation
among them exists which casts the canonical spherical harmonics into real-valued
so-called tesseral spherical harmonics. A similar transformation can be applied
to the Wigner-D matrix such that it represents the effect of a rotation among
tesseral spherical harmonics.

Given the tesseral transformation that converts canonical spherical harmonics into
tesseral ones as given by :math:`\mathbf{T}` and the Wigner-D matrix :math:`\mathbf{D}` built for canonical spherical harmonics, we can readily compute the Wigner-D matrix
:math:`\mathbf{D}\prime` for tesseral spherical harmonics by means of a basis transformation.

.. math::
  \mathbf{D}\prime = \mathbf{T} \mathbf{D} \mathbf{T}^{\dagger}.

.. note::
  In :program:`Sphecerix` one can choose to construct the Wigner-D matrix for either
  canonical or tesseral spherical harmonics.