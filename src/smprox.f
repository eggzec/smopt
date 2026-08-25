c-----------------------------------------------------------------------
c     Proximal operators of the regularizers supported by SMOPT, plus
c     the closed form multiplier attached to the l_{2,1} penalty.
c
c     The prox of a function r at X with step ETA is the minimizer of
c
c         (1/(2*ETA))*||Y - X||_F^2 + r(Y).
c-----------------------------------------------------------------------

      SUBROUTINE SMPL1(N, P, X, ETA, GAM, Y)
c     Y := prox of GAM*||.||_1 at X with step ETA, that is
c     max(X - GAM*ETA, 0) + min(X + GAM*ETA, 0) entrywise.
      INTEGER N, P
      DOUBLE PRECISION X(N,P), Y(N,P), ETA, GAM
      INTEGER I, J
      DOUBLE PRECISION T

      T = GAM*ETA
      DO 20 J = 1, P
         DO 10 I = 1, N
            Y(I,J) = DMAX1(X(I,J) - T, 0.0D0)
     $             + DMIN1(X(I,J) + T, 0.0D0)
   10    CONTINUE
   20 CONTINUE
      RETURN
      END


      SUBROUTINE SMPL21(N, P, X, ETA, GAM, EPS, Y)
c     Y := prox of GAM*||.||_{2,1} at X with step ETA.  The l_{2,1}
c     norm sums the Euclidean norms of the rows of X, so the prox
c     shrinks each row towards the origin.  EPS guards the division by
c     a vanishing row norm.
      INTEGER N, P
      DOUBLE PRECISION X(N,P), Y(N,P), ETA, GAM, EPS
      INTEGER I, J
      DOUBLE PRECISION T, R, SC

      T = GAM*ETA
      DO 30 I = 1, N
         R = 0.0D0
         DO 10 J = 1, P
            R = R + X(I,J)*X(I,J)
   10    CONTINUE
         R = DSQRT(R)
         SC = DMAX1(R - T, 0.0D0)/(R + EPS)
         DO 20 J = 1, P
            Y(I,J) = SC*X(I,J)
   20    CONTINUE
   30 CONTINUE
      RETURN
      END


      SUBROUTINE SMLM21(N, P, X, GAM, LAM)
c     LAM := -GAM*X**T*diag(w)*X with w(i) = 1/(1.0D-14 + ||X(i,.)||),
c     the multiplier that the l_{2,1} regularized solver attaches to
c     the orthogonality constraint.
      INTEGER N, P
      DOUBLE PRECISION X(N,P), LAM(P,P), GAM
      INTEGER I, J, K
      DOUBLE PRECISION R, T

      DO 20 J = 1, P
         DO 10 I = 1, P
            LAM(I,J) = 0.0D0
   10    CONTINUE
   20 CONTINUE

      DO 60 K = 1, N
         R = 0.0D0
         DO 30 J = 1, P
            R = R + X(K,J)*X(K,J)
   30    CONTINUE
         R = 1.0D0/(1.0D-14 + DSQRT(R))
         DO 50 J = 1, P
            T = GAM*R*X(K,J)
            DO 40 I = 1, P
               LAM(I,J) = LAM(I,J) - X(K,I)*T
   40       CONTINUE
   50    CONTINUE
   60 CONTINUE
      RETURN
      END
