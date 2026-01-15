# ProjectedSphericalHarmonics

This library provides tools for working with the projected spherical harmonics on the unit disk. It includes functions for evaluating and inverting a variety of integral operators commonly encountered in boundary integral equations for open surfaces in 3D. 

## Mathematical Background

The projected spherical harmonics are defined as

$$ y_\ell^m(r, \theta) = N_\ell^m P_\ell^m(\sqrt{1 - r^2}) e^{im\theta}, $$

where $P_\ell^m$ are the associated Legendre polynomials and

$$N^m_\ell = \sqrt{ \frac{2\ell+1}{2\pi} \frac{(\ell - m)!}{(\ell + m)!} }$$

is a normalization constant. Functions of the same parity (even or odd $\ell + m$) are orthogonal with respect to the weighted inner product

$$ \langle u, v \rangle = \int_D \frac{u(\mathbf{x}) v^*(\mathbf{x}) }{\sqrt{1 - |\mathbf{x}|^2}} {\rm d}\mathbf{x}. $$
