c-----------------------------------------------------------------------
c     Dense linear algebra kernels used throughout SMOPT.
c
c     Everything the solvers need is implemented here, so the extension
c     module builds against nothing but a Fortran compiler and keeps the
c     dependency footprint of the project template.
c
c     All matrices are double precision and stored column major, which
c     is the layout f2py hands over from NumPy.
c-----------------------------------------------------------------------

      DOUBLE PRECISION FUNCTION SMFRO(N, M, A)
c     Frobenius norm of the N by M matrix A.
      INTEGER N, M
      DOUBLE PRECISION A(N,M)
      INTEGER I, J
      DOUBLE PRECISION T

      T = 0.0D0
      DO 20 J = 1, M
         DO 10 I = 1, N
            T = T + A(I,J)*A(I,J)
   10    CONTINUE
   20 CONTINUE
      SMFRO = DSQRT(T)
      RETURN
      END


      DOUBLE PRECISION FUNCTION SMDOT(N, M, A, B)
c     Sum of the elementwise product of two N by M matrices.
      INTEGER N, M
      DOUBLE PRECISION A(N,M), B(N,M)
      INTEGER I, J
      DOUBLE PRECISION T

      T = 0.0D0
      DO 20 J = 1, M
         DO 10 I = 1, N
            T = T + A(I,J)*B(I,J)
   10    CONTINUE
   20 CONTINUE
      SMDOT = T
      RETURN
      END


      SUBROUTINE SMCOPY(N, M, A, B)
c     B := A for N by M matrices.
      INTEGER N, M
      DOUBLE PRECISION A(N,M), B(N,M)
      INTEGER I, J

      DO 20 J = 1, M
         DO 10 I = 1, N
            B(I,J) = A(I,J)
   10    CONTINUE
   20 CONTINUE
      RETURN
      END


      SUBROUTINE SMSCAL(N, M, A, S)
c     A := S*A for the N by M matrix A.
      INTEGER N, M
      DOUBLE PRECISION A(N,M), S
      INTEGER I, J

      DO 20 J = 1, M
         DO 10 I = 1, N
            A(I,J) = S*A(I,J)
   10    CONTINUE
   20 CONTINUE
      RETURN
      END


      SUBROUTINE SMMM(N, K, M, A, B, C)
c     C := A*B with A(N,K), B(K,M) and C(N,M).
      INTEGER N, K, M
      DOUBLE PRECISION A(N,K), B(K,M), C(N,M)
      INTEGER I, J, L
      DOUBLE PRECISION T

      DO 40 J = 1, M
         DO 10 I = 1, N
            C(I,J) = 0.0D0
   10    CONTINUE
         DO 30 L = 1, K
            T = B(L,J)
            IF (T .NE. 0.0D0) THEN
               DO 20 I = 1, N
                  C(I,J) = C(I,J) + A(I,L)*T
   20          CONTINUE
            END IF
   30    CONTINUE
   40 CONTINUE
      RETURN
      END


      SUBROUTINE SMMTM(N, K, M, A, B, C)
c     C := A**T*B with A(N,K), B(N,M) and C(K,M).
      INTEGER N, K, M
      DOUBLE PRECISION A(N,K), B(N,M), C(K,M)
      INTEGER I, J, L
      DOUBLE PRECISION T

      DO 30 J = 1, M
         DO 20 I = 1, K
            T = 0.0D0
            DO 10 L = 1, N
               T = T + A(L,I)*B(L,J)
   10       CONTINUE
            C(I,J) = T
   20    CONTINUE
   30 CONTINUE
      RETURN
      END


      SUBROUTINE SMRMM(N, P, X, W, ROW)
c     X := X*W in place, with X(N,P) and W(P,P).  ROW is a workspace
c     vector of length P holding one transformed row at a time, which
c     keeps the update free of any N by P scratch storage.
      INTEGER N, P
      DOUBLE PRECISION X(N,P), W(P,P), ROW(P)
      INTEGER I, J, K
      DOUBLE PRECISION T

      DO 40 I = 1, N
         DO 20 J = 1, P
            T = 0.0D0
            DO 10 K = 1, P
               T = T + X(I,K)*W(K,J)
   10       CONTINUE
            ROW(J) = T
   20    CONTINUE
         DO 30 J = 1, P
            X(I,J) = ROW(J)
   30    CONTINUE
   40 CONTINUE
      RETURN
      END


      SUBROUTINE SMSYMM(P, A)
c     A := (A + A**T)/2, the symmetrizing operator written Phi in the
c     accompanying papers.
      INTEGER P
      DOUBLE PRECISION A(P,P)
      INTEGER I, J
      DOUBLE PRECISION T

      DO 20 J = 1, P
         DO 10 I = J+1, P
            T = 0.5D0*(A(I,J) + A(J,I))
            A(I,J) = T
            A(J,I) = T
   10    CONTINUE
   20 CONTINUE
      RETURN
      END


      SUBROUTINE SMCHOL(P, A, INFO)
c     Cholesky factorization A = L*L**T of a symmetric positive
c     definite matrix.  L overwrites the lower triangle of A.  INFO is
c     zero on success and the index of the failing column otherwise.
      INTEGER P, INFO
      DOUBLE PRECISION A(P,P)
      INTEGER I, J, K
      DOUBLE PRECISION T

      INFO = 0
      DO 40 J = 1, P
         T = A(J,J)
         DO 10 K = 1, J-1
            T = T - A(J,K)*A(J,K)
   10    CONTINUE
         IF (T .LE. 0.0D0) THEN
            INFO = J
            RETURN
         END IF
         A(J,J) = DSQRT(T)
         DO 30 I = J+1, P
            T = A(I,J)
            DO 20 K = 1, J-1
               T = T - A(I,K)*A(J,K)
   20       CONTINUE
            A(I,J) = T/A(J,J)
   30    CONTINUE
   40 CONTINUE
      RETURN
      END


      SUBROUTINE SMSOLR(P, L, N, X)
c     Solve M*z = x for every row x of X(N,P), where the symmetric
c     positive definite M is supplied through its Cholesky factor L.
c     X is overwritten by the solutions, that is X := X*M**(-1).
      INTEGER P, N
      DOUBLE PRECISION L(P,P), X(N,P)
      INTEGER I, J, K
      DOUBLE PRECISION T

      DO 50 I = 1, N
         DO 20 J = 1, P
            T = X(I,J)
            DO 10 K = 1, J-1
               T = T - L(J,K)*X(I,K)
   10       CONTINUE
            X(I,J) = T/L(J,J)
   20    CONTINUE
         DO 40 J = P, 1, -1
            T = X(I,J)
            DO 30 K = J+1, P
               T = T - L(K,J)*X(I,K)
   30       CONTINUE
            X(I,J) = T/L(J,J)
   40    CONTINUE
   50 CONTINUE
      RETURN
      END


      SUBROUTINE SMJACO(P, A, W, V)
c     Cyclic Jacobi eigenvalue decomposition of the symmetric matrix A,
c     producing A = V*diag(W)*V**T.  A is destroyed.  The matrices met
c     here are of order P, the column count of the iterate, so a few
c     sweeps always suffice.
      INTEGER P
      DOUBLE PRECISION A(P,P), W(P), V(P,P)
      INTEGER I, J, K, ISW
      DOUBLE PRECISION OFF, THETA, T, C, S, H, U1, U2
      DOUBLE PRECISION TOL
      PARAMETER (TOL = 1.0D-30)

      DO 20 J = 1, P
         DO 10 I = 1, P
            V(I,J) = 0.0D0
   10    CONTINUE
         V(J,J) = 1.0D0
   20 CONTINUE

      DO 100 ISW = 1, 100
         OFF = 0.0D0
         DO 40 J = 2, P
            DO 30 I = 1, J-1
               OFF = OFF + A(I,J)*A(I,J)
   30       CONTINUE
   40    CONTINUE
         IF (OFF .LE. TOL) GO TO 110

         DO 90 J = 2, P
            DO 80 I = 1, J-1
               IF (A(I,J) .NE. 0.0D0) THEN
                  H = A(J,J) - A(I,I)
                  IF (DABS(H) + DABS(A(I,J)) .EQ. DABS(H)) THEN
                     T = A(I,J)/H
                  ELSE
                     THETA = 0.5D0*H/A(I,J)
                     T = 1.0D0/(DABS(THETA)
     $                          + DSQRT(1.0D0 + THETA*THETA))
                     IF (THETA .LT. 0.0D0) T = -T
                  END IF
                  C = 1.0D0/DSQRT(1.0D0 + T*T)
                  S = T*C
                  DO 50 K = 1, P
                     U1 = A(I,K)
                     U2 = A(J,K)
                     A(I,K) = C*U1 - S*U2
                     A(J,K) = S*U1 + C*U2
   50             CONTINUE
                  DO 60 K = 1, P
                     U1 = A(K,I)
                     U2 = A(K,J)
                     A(K,I) = C*U1 - S*U2
                     A(K,J) = S*U1 + C*U2
   60             CONTINUE
                  DO 70 K = 1, P
                     U1 = V(K,I)
                     U2 = V(K,J)
                     V(K,I) = C*U1 - S*U2
                     V(K,J) = S*U1 + C*U2
   70             CONTINUE
               END IF
   80       CONTINUE
   90    CONTINUE
  100 CONTINUE

  110 CONTINUE
      DO 120 I = 1, P
         W(I) = A(I,I)
  120 CONTINUE
      RETURN
      END


      SUBROUTINE SMMGS(N, P, X)
c     Modified Gram-Schmidt orthonormalization.  X is overwritten by an
c     N by P matrix with orthonormal columns.
      INTEGER N, P
      DOUBLE PRECISION X(N,P)
      INTEGER I, J, K
      DOUBLE PRECISION T

      DO 60 J = 1, P
         DO 30 K = 1, J-1
            T = 0.0D0
            DO 10 I = 1, N
               T = T + X(I,K)*X(I,J)
   10       CONTINUE
            DO 20 I = 1, N
               X(I,J) = X(I,J) - T*X(I,K)
   20       CONTINUE
   30    CONTINUE
         T = 0.0D0
         DO 40 I = 1, N
            T = T + X(I,J)*X(I,J)
   40    CONTINUE
         T = DSQRT(T)
         IF (T .GT. 0.0D0) THEN
            DO 50 I = 1, N
               X(I,J) = X(I,J)/T
   50       CONTINUE
         END IF
   60 CONTINUE
      RETURN
      END
