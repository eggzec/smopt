# References

The solvers implemented here follow the constraint dissolving and
penalty-free line of work on Stiefel manifold optimization.

- Xiao, N., Liu, X., and Yuan, Y. (2022). *A class of smooth exact
  penalty function methods for optimization problems with orthogonality
  constraints.* Optimization Methods and Software, 37(4), 1205-1241.

- Xiao, N., Liu, X., and Yuan, Y. (2022). *Exact penalty function for
  $\ell_{2,1}$ norm minimization over the Stiefel manifold.* SIAM
  Journal on Optimization, 31(4), 3097-3126.

- Xiao, N., Liu, X., and Toh, K.-C. (2023). *Dissolving constraints for
  Riemannian optimization.* Mathematics of Operations Research.

- Barzilai, J. and Borwein, J. M. (1988). *Two-point step size gradient
  methods.* IMA Journal of Numerical Analysis, 8(1), 141-148.

- Edelman, A., Arias, T. A., and Smith, S. T. (1998). *The geometry of
  algorithms with orthogonality constraints.* SIAM Journal on Matrix
  Analysis and Applications, 20(2), 303-353.

- Absil, P.-A., Mahony, R., and Sepulchre, R. (2008). *Optimization
  Algorithms on Matrix Manifolds.* Princeton University Press.

- Golub, G. H. and Van Loan, C. F. (2013). *Matrix Computations*, 4th
  edition. Johns Hopkins University Press. Chapter 8 covers the cyclic
  Jacobi eigenvalue algorithm used for the polar factor.

## Provenance

The algorithm ported here originates in the **STOP** toolbox by Nachuan
Xiao, Lei Wang, Bin Gao, Xin Liu and Ya-xiang Yuan, distributed at
<https://stmopt.gitee.io/>. `smopt` re-implements its numerics in
Fortran 77 behind the same solver interface.
