c-----------------------------------------------------------------------
c     Geometry of the Stiefel manifold
c
c         S(n,p) = { X in R^(n x p) : X**T*X = I_p }.
c
c     The routines below are the Fortran 77 counterparts of the manifold
c     class exposed by SMOPT: the constraint map C, its Jacobian JC and
c     adjoint, the projection JA onto the tangent-like space, the
c     feasibility restoring map A, and the polar retraction used to
c     round the final iterate back onto the manifold.
c-----------------------------------------------------------------------

      SUBROUTINE SMCMAP(N, P, X, C)
c     C := X**T*X - I, the constraint violation of X.
      INTEGER N, P
      DOUBLE PRECISION X(N,P), C(P,P)
      INTEGER I

      CALL SMMTM(N, P, P, X, X, C)
      DO 10 I = 1, P
         C(I,I) = C(I,I) - 1.0D0
   10 CONTINUE
      RETURN
      END


      DOUBLE PRECISION FUNCTION SMFEAS(N, P, X, WP)
c     Feasibility measure ||X**T*X - I||_F.  WP is P by P workspace.
      INTEGER N, P
      DOUBLE PRECISION X(N,P), WP(P,P)
      DOUBLE PRECISION SMFRO
      EXTERNAL SMFRO

      CALL SMCMAP(N, P, X, WP)
      SMFEAS = SMFRO(P, P, WP)
      RETURN
      END


      SUBROUTINE SMJA(N, P, X, G, R, WP)
c     R := G - X*Phi(X**T*G), the projection that turns a Euclidean
c     gradient G at X into the search direction used by the solvers.
      INTEGER N, P
      DOUBLE PRECISION X(N,P), G(N,P), R(N,P), WP(P,P)
      INTEGER I, J

      CALL SMMTM(N, P, P, X, G, WP)
      CALL SMSYMM(P, WP)
      CALL SMMM(N, P, P, X, WP, R)
      DO 20 J = 1, P
         DO 10 I = 1, N
            R(I,J) = G(I,J) - R(I,J)
   10    CONTINUE
   20 CONTINUE
      RETURN
      END


      SUBROUTINE SMJC(N, P, X, LAM, R, WP)
c     R := X*Phi(LAM), the Jacobian of the constraint map applied to a
c     multiplier LAM.  LAM is left untouched.
      INTEGER N, P
      DOUBLE PRECISION X(N,P), LAM(P,P), R(N,P), WP(P,P)

      CALL SMCOPY(P, P, LAM, WP)
      CALL SMSYMM(P, WP)
      CALL SMMM(N, P, P, X, WP, R)
      RETURN
      END


      SUBROUTINE SMJCT(N, P, X, D, R)
c     R := Phi(X**T*D), the adjoint of SMJC applied to a direction D.
      INTEGER N, P
      DOUBLE PRECISION X(N,P), D(N,P), R(P,P)

      CALL SMMTM(N, P, P, X, D, R)
      CALL SMSYMM(P, R)
      RETURN
      END


      SUBROUTINE SMAMAP(N, P, X, WP1, WP2, ROW)
c     X := A(X), the feasibility restoring map.  Close to the manifold
c     the cheap second order expansion 1.5*X - X*(X**T*X)/2 is used,
c     and the exact map X*((X**T*X + I)/2)**(-1) otherwise.
      INTEGER N, P
      DOUBLE PRECISION X(N,P), WP1(P,P), WP2(P,P), ROW(P)
      INTEGER I, J, INFO
      DOUBLE PRECISION FEAS
      DOUBLE PRECISION SMFRO
      EXTERNAL SMFRO

      CALL SMMTM(N, P, P, X, X, WP1)
      DO 20 J = 1, P
         DO 10 I = 1, P
            WP2(I,J) = WP1(I,J)
   10    CONTINUE
         WP2(J,J) = WP2(J,J) - 1.0D0
   20 CONTINUE
      FEAS = SMFRO(P, P, WP2)

      IF (FEAS .LT. 0.5D0) THEN
         DO 40 J = 1, P
            DO 30 I = 1, P
               WP2(I,J) = -0.5D0*WP1(I,J)
   30       CONTINUE
            WP2(J,J) = WP2(J,J) + 1.5D0
   40    CONTINUE
         CALL SMRMM(N, P, X, WP2, ROW)
      ELSE
         DO 60 J = 1, P
            DO 50 I = 1, P
               WP2(I,J) = 0.5D0*WP1(I,J)
   50       CONTINUE
            WP2(J,J) = WP2(J,J) + 0.5D0
   60    CONTINUE
         CALL SMCHOL(P, WP2, INFO)
         IF (INFO .EQ. 0) CALL SMSOLR(P, WP2, N, X)
      END IF
      RETURN
      END


      SUBROUTINE SMFIX(N, P, X, WP1, WP2, ROW)
c     Feasibility restoration used by PENCF.  Unlike SMAMAP the map is
c     applied only once X has drifted appreciably off the manifold, and
c     the result is capped in Frobenius norm to keep the penalized
c     iteration bounded.
      INTEGER N, P
      DOUBLE PRECISION X(N,P), WP1(P,P), WP2(P,P), ROW(P)
      INTEGER I, J, INFO
      DOUBLE PRECISION FEAS, XN, CAP
      DOUBLE PRECISION SMFRO
      EXTERNAL SMFRO

      CALL SMMTM(N, P, P, X, X, WP1)
      DO 20 J = 1, P
         DO 10 I = 1, P
            WP2(I,J) = WP1(I,J)
   10    CONTINUE
         WP2(J,J) = WP2(J,J) - 1.0D0
   20 CONTINUE
      FEAS = SMFRO(P, P, WP2)

      IF (FEAS .GT. 1.0D-1) THEN
         IF (FEAS .LT. 0.5D0) THEN
            DO 40 J = 1, P
               DO 30 I = 1, P
                  WP2(I,J) = -0.5D0*WP1(I,J)
   30          CONTINUE
               WP2(J,J) = WP2(J,J) + 1.5D0
   40       CONTINUE
            CALL SMRMM(N, P, X, WP2, ROW)
         ELSE
            DO 60 J = 1, P
               DO 50 I = 1, P
                  WP2(I,J) = 0.5D0*WP1(I,J)
   50          CONTINUE
               WP2(J,J) = WP2(J,J) + 0.5D0
   60       CONTINUE
            CALL SMCHOL(P, WP2, INFO)
            IF (INFO .EQ. 0) CALL SMSOLR(P, WP2, N, X)
         END IF
      END IF

      CAP = 1.001D0*DSQRT(DBLE(P))
      XN = SMFRO(N, P, X)
      IF (XN .GT. CAP) CALL SMSCAL(N, P, X, CAP/XN)
      RETURN
      END


      SUBROUTINE SMPOST(N, P, X, WP1, WP2, WP3, W, ROW)
c     X := U*V**T where X = U*S*V**T is a thin singular value
c     decomposition.  When X has full column rank that orthogonal polar
c     factor equals X*(X**T*X)**(-1/2), obtained here from a Jacobi
c     eigendecomposition of the P by P matrix X**T*X rather than from a
c     decomposition of X itself.
c
c     A rank deficient X has no polar factor: the directions belonging
c     to a vanishing singular value are unconstrained.  Those columns
c     are filled with an arbitrary orthonormal completion so that the
c     result still lands on the manifold, which is what a singular value
c     decomposition would hand back as well.  Regularized solvers reach
c     this case whenever they zero out enough rows.
      INTEGER N, P
      DOUBLE PRECISION X(N,P), WP1(P,P), WP2(P,P), WP3(P,P)
      DOUBLE PRECISION W(P), ROW(P)
      INTEGER I, J, K, M
      DOUBLE PRECISION T, WMAX, TOL, THR
      DOUBLE PRECISION EPS
      PARAMETER (EPS = 2.220446049250313D-16)

      CALL SMMTM(N, P, P, X, X, WP1)
      CALL SMSYMM(P, WP1)
      CALL SMJACO(P, WP1, W, WP2)

c     X := X*V, whose K-th column is the K-th singular value times the
c     corresponding left singular vector.
      CALL SMRMM(N, P, X, WP2, ROW)

      WMAX = 0.0D0
      DO 10 K = 1, P
         IF (W(K) .GT. WMAX) WMAX = W(K)
   10 CONTINUE
      TOL = DBLE(P)*EPS*WMAX

c     Normalize the columns that carry a nonzero singular value and
c     blank the rest, so the completion below sees only real vectors.
      DO 40 K = 1, P
         IF (W(K) .GT. TOL) THEN
            T = 1.0D0/DSQRT(W(K))
            DO 20 I = 1, N
               X(I,K) = T*X(I,K)
   20       CONTINUE
         ELSE
            DO 30 I = 1, N
               X(I,K) = 0.0D0
   30       CONTINUE
         END IF
   40 CONTINUE

c     Complete the blanked columns.  Projecting the N coordinate axes
c     off the P-1 columns already in place leaves a total squared norm
c     of N-P+1, so at least one axis clears the threshold below.
      THR = 0.5D0*DSQRT(DBLE(N-P+1)/DBLE(N))
      DO 130 K = 1, P
         IF (W(K) .LE. TOL) THEN
            DO 120 J = 1, N
               DO 50 I = 1, N
                  X(I,K) = 0.0D0
   50          CONTINUE
               X(J,K) = 1.0D0
               DO 80 M = 1, P
                  IF (M .NE. K) THEN
                     T = 0.0D0
                     DO 60 I = 1, N
                        T = T + X(I,M)*X(I,K)
   60                CONTINUE
                     DO 70 I = 1, N
                        X(I,K) = X(I,K) - T*X(I,M)
   70                CONTINUE
                  END IF
   80          CONTINUE
               T = 0.0D0
               DO 90 I = 1, N
                  T = T + X(I,K)*X(I,K)
   90          CONTINUE
               T = DSQRT(T)
               IF (T .GT. THR) THEN
                  DO 100 I = 1, N
                     X(I,K) = X(I,K)/T
  100             CONTINUE
                  GO TO 130
               END IF
  120       CONTINUE
         END IF
  130 CONTINUE

c     X := U*V**T.
      DO 150 J = 1, P
         DO 140 I = 1, P
            WP3(I,J) = WP2(J,I)
  140    CONTINUE
  150 CONTINUE
      CALL SMRMM(N, P, X, WP3, ROW)
      RETURN
      END


      SUBROUTINE SMINIT(N, P, X, WP)
c     Orthonormalize X unless it already sits on the Stiefel manifold.
      INTEGER N, P
      DOUBLE PRECISION X(N,P), WP(P,P)
      DOUBLE PRECISION SMFEAS
      EXTERNAL SMFEAS

      IF (SMFEAS(N, P, X, WP) .GT. 1.0D-6) CALL SMMGS(N, P, X)
      RETURN
      END
