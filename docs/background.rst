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

Mirror operations among tesseral spherical harmonics
----------------------------------------------------

To construct the transformation of a mirror operation :math:`\hat{M}`, we can note that any
mirror operation can be decomposed into a rotation and an inversion. Given a mirror
operation through a mirror plane as represented by its normal vector :math:`\vec{n}`.

The matrix representation of this mirror operation is given by

.. math::

  \mathbf{M} = \mathbf{I}_{3} - 2 \vec{n} \cdot \vec{n}^{\dagger}

This matrix has a negative determinant and thus contains a reflection. We can
get rid of this reflection by multiplying this matrix by -1 which is equivalent
to an extraction of an inversion operation. In other words, we can decompose
the matrix :math:`\mathbf{M}` such that

.. math::
  \mathbf{M} = (-\mathbf{I}_{3}) \mathbf{R}.

The reflection operation in the space spanned by the tesseral spherical
harmonics of order :math:`l` is known and corresponds to the character for the
inversion operation which is equal to :math:`(-1)^{l}`. As such, the
transformation matrix corresponding to a mirror operation among tesseral
spherical harmonics of order :math:`l` is given by

.. math::
  \mathbf{T} = (-1)^{l} \cdot \mathbf{D}\prime(\hat{R})

wherein :math:`\mathbf{D}\prime` is the tesseral Wigner-D matrix for the
rotation :math:`\hat{R}` as extracted from the mirror operation :math:`\hat{M}`.


Labels tesseral spherical harmonics
-----------------------------------

For the real-valued spherical harmonics, i.e. the spherical harmonics after a
tesseral transformation, their canonical names are derived from the
mathematical equation that describes the angular part of the hydrogen-like
wave function. In the table below, an overview is given how the labels for
the tesseral spherical harmonics are associated to the values for :math:`m`.

.. tip::
  For a detailed description how these labels are constructed, have a look 
  at `this publication of Ashkenazi <https://pubs.acs.org/doi/abs/10.1021/ed082p323>`_.

.. list-table:: Labeling of the tesseral spherical harmonics
   :header-rows: 1

   * - :math:`-4`
     - :math:`-3`
     - :math:`-2`
     - :math:`-1`
     - :math:`-0`
     - :math:`-1`
     - :math:`-2`
     - :math:`-3`
     - :math:`-4`
   * - :math:`-`
     - :math:`-`
     - :math:`-`
     - :math:`-`
     - :math:`s`
     - :math:`-`
     - :math:`-`
     - :math:`-`
     - :math:`-`
   * - :math:`-`
     - :math:`-`
     - :math:`-`
     - :math:`p_{y}`
     - :math:`p_{z}`
     - :math:`p_{x}`
     - :math:`-`
     - :math:`-`
     - :math:`-`
   * - :math:`-`
     - :math:`-`
     - :math:`d_{xy}`
     - :math:`d_{yz}`
     - :math:`d_{z^{2}}`
     - :math:`d_{xz}`
     - :math:`d_{x^2-y^2}`
     - :math:`-`
     - :math:`-`
   * - :math:`-`
     - :math:`f_{y(3x^2-y^2)}`
     - :math:`f_{xyz}`
     - :math:`f_{yz^2}`
     - :math:`f_{z^3}`
     - :math:`f_{xz^2}`
     - :math:`f_{z(x^2-y^2)}`
     - :math:`f_{x(x^2-3y^2)}`
     - :math:`-`
   * - :math:`g_{xy(x^2-y^2)}`
     - :math:`g_{zy^3}`
     - :math:`g_{xyz^2}`
     - :math:`g_{yz^3}`
     - :math:`g_{z^4}`
     - :math:`g_{xz^3}`
     - :math:`g_{z^2(x^2-y^2)}`
     - :math:`g_{zx^3}`
     - :math:`g_{x^4+y^4}`