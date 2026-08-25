c-----------------------------------------------------------------------
c     SLPG, the penalty-free first-order family of solvers for
c
c         min f(X) + r(X)   subject to   X**T*X = I_p.
c
c     Three drivers are provided: SMSLPS for a smooth f, SMSLPG for a
c     general r reachable through its proximal operator, and SMSL21 for
c     the l_{2,1} regularizer whose prox and multiplier are known in
c     closed form.
c
c     The objective is supplied by the caller through the OBJFUN
c     callback, which receives the flattened iterate and returns the
c     function value together with its Euclidean gradient.  LOGFUN
c     reports progress; the STAGE argument is 0 for an ordinary
c     iteration, 1 for the line printed on convergence and 2 for the
c     line printed after post-processing.
c-----------------------------------------------------------------------

      DOUBLE PRECISION FUNCTION SMSTEP(NUM, DEN, CAP)
c     Barzilai-Borwein trial step |NUM/DEN| truncated at CAP.  A zero
c     denominator leaves the ratio unbounded, so the cap is returned.
      DOUBLE PRECISION NUM, DEN, CAP

      IF (DEN .EQ. 0.0D0) THEN
         SMSTEP = CAP
      ELSE
         SMSTEP = DMIN1(DABS(NUM/DEN), CAP)
      END IF
      RETURN
      END


      SUBROUTINE SMAHUR(N, P, X, G, ETA, TOL, LAM, PROXFN,
     $                  Z, XT, DX, WP)
c     Five steps of the Arrow-Hurwicz iteration that updates the
c     multiplier LAM of the orthogonality constraint at X.  The loop
c     stops early once the multiplier increment drops below TOL.
      INTEGER N, P
      DOUBLE PRECISION X(N,P), G(N,P), LAM(P,P), ETA, TOL
      DOUBLE PRECISION Z(N,P), XT(N,P), DX(N,P), WP(P,P)
      EXTERNAL PROXFN
      INTEGER I, J, JR, NP
      DOUBLE PRECISION SMFRO
      EXTERNAL SMFRO

      NP = N*P
      DO 20 J = 1, P
         DO 10 I = 1, N
            Z(I,J) = X(I,J) - ETA*G(I,J)
   10    CONTINUE
   20 CONTINUE

      DO 100 JR = 1, 5
         CALL SMJC(N, P, X, LAM, DX, WP)
         DO 40 J = 1, P
            DO 30 I = 1, N
               XT(I,J) = Z(I,J) - ETA*DX(I,J)
   30       CONTINUE
   40    CONTINUE
         CALL PROXFN(NP, XT, ETA, DX)
         DO 60 J = 1, P
            DO 50 I = 1, N
               DX(I,J) = (DX(I,J) - X(I,J))/ETA
   50       CONTINUE
   60    CONTINUE
         CALL SMJCT(N, P, X, DX, WP)
         DO 80 J = 1, P
            DO 70 I = 1, P
               LAM(I,J) = LAM(I,J) + WP(I,J)
   70       CONTINUE
   80    CONTINUE
         IF (SMFRO(P, P, WP) .LT. TOL) RETURN
  100 CONTINUE
      RETURN
      END


      SUBROUTINE SMSLPS(N, P, X, MAXIT, GTOL, IPOST, OBJFUN, LOGFUN,
     $                  NIT, FVALS, KKTS, FEASV, FVAL, KKT, FEA,
     $                  GF, GR, GRP, S, Y, XP, WP1, WP2, WP3, WEIG,
     $                  ROW)
c     SLPG for a smooth objective.  A Barzilai-Borwein step is taken
c     along the projected gradient and the iterate is pulled back onto
c     the manifold by the feasibility restoring map A.
      INTEGER N, P, MAXIT, IPOST, NIT
      DOUBLE PRECISION X(N,P), GTOL, FVAL, KKT, FEA
      DOUBLE PRECISION FVALS(MAXIT), KKTS(MAXIT), FEASV(MAXIT)
      DOUBLE PRECISION GF(N,P), GR(N,P), GRP(N,P)
      DOUBLE PRECISION S(N,P), Y(N,P), XP(N,P)
      DOUBLE PRECISION WP1(P,P), WP2(P,P), WP3(P,P)
      DOUBLE PRECISION WEIG(P), ROW(P)
      EXTERNAL OBJFUN, LOGFUN
      INTEGER I, J, JJ, NP
      DOUBLE PRECISION L, STEP
      DOUBLE PRECISION SMFRO, SMDOT, SMFEAS, SMSTEP
      EXTERNAL SMFRO, SMDOT, SMFEAS, SMSTEP

      NP = N*P
      NIT = 0
      KKT = 0.0D0
      FEA = 0.0D0

      CALL OBJFUN(NP, X, FVAL, GF)
      CALL SMJA(N, P, X, GF, GR, WP1)
      L = SMFRO(N, P, GF) + SMFRO(N, P, GR)

      DO 100 JJ = 1, MAXIT
         IF (JJ .LE. 3) THEN
            STEP = 0.01D0/L
         ELSE
            STEP = SMSTEP(SMDOT(N, P, S, Y), SMDOT(N, P, Y, Y),
     $                    1.0D10)
         END IF

         CALL SMCOPY(N, P, X, XP)
         DO 20 J = 1, P
            DO 10 I = 1, N
               X(I,J) = X(I,J) - STEP*GR(I,J)
   10       CONTINUE
   20    CONTINUE
         CALL SMAMAP(N, P, X, WP1, WP2, ROW)

         DO 40 J = 1, P
            DO 30 I = 1, N
               S(I,J) = X(I,J) - XP(I,J)
   30       CONTINUE
   40    CONTINUE

         CALL OBJFUN(NP, X, FVAL, GF)
         CALL SMCOPY(N, P, GR, GRP)
         CALL SMJA(N, P, X, GF, GR, WP1)
         DO 60 J = 1, P
            DO 50 I = 1, N
               Y(I,J) = GR(I,J) - GRP(I,J)
   50       CONTINUE
   60    CONTINUE

         KKT = SMFRO(N, P, GR)
         FEA = SMFEAS(N, P, X, WP1)

         NIT = JJ
         FVALS(JJ) = FVAL
         KKTS(JJ) = KKT
         FEASV(JJ) = FEA
         CALL LOGFUN(JJ-1, FVAL, KKT, FEA, 0)

         IF (KKT .LT. GTOL) THEN
            CALL LOGFUN(JJ-1, FVAL, KKT, FEA, 1)
            GO TO 110
         END IF
  100 CONTINUE

  110 CONTINUE
      IF (IPOST .NE. 0 .AND. NIT .GE. 1) THEN
         CALL SMPOST(N, P, X, WP1, WP2, WP3, WEIG, ROW)
         CALL OBJFUN(NP, X, FVAL, GF)
         CALL SMJA(N, P, X, GF, GR, WP1)
         KKT = SMFRO(N, P, GR)
         FEA = SMFEAS(N, P, X, WP1)
         CALL LOGFUN(NIT-1, FVAL, KKT, FEA, 2)
         FVALS(NIT) = FVAL
         KKTS(NIT) = KKT
         FEASV(NIT) = FEA
      END IF
      RETURN
      END


      SUBROUTINE SMSLPG(N, P, X, MAXIT, GTOL, IPOST, OBJFUN, PROXFN,
     $                  LOGFUN, NIT, FVALS, KKTS, FEASV, FVAL, KKT,
     $                  FEA, GF, GR, GRAD, GRDP, S, Y, XP, Z, XT, DX,
     $                  LAM, WP1, WP2, WP3, WEIG, ROW, STEPS)
c     SLPG for a nonsmooth regularizer reached through its proximal
c     operator PROXFN.  The multiplier of the orthogonality constraint
c     is tracked by an Arrow-Hurwicz inner iteration so that no penalty
c     parameter has to be tuned.
      INTEGER N, P, MAXIT, IPOST, NIT
      DOUBLE PRECISION X(N,P), GTOL, FVAL, KKT, FEA
      DOUBLE PRECISION FVALS(MAXIT), KKTS(MAXIT), FEASV(MAXIT)
      DOUBLE PRECISION GF(N,P), GR(N,P), GRAD(N,P), GRDP(N,P)
      DOUBLE PRECISION S(N,P), Y(N,P), XP(N,P)
      DOUBLE PRECISION Z(N,P), XT(N,P), DX(N,P)
      DOUBLE PRECISION LAM(P,P), WP1(P,P), WP2(P,P), WP3(P,P)
      DOUBLE PRECISION WEIG(P), ROW(P), STEPS(MAXIT)
      EXTERNAL OBJFUN, PROXFN, LOGFUN
      INTEGER I, J, K, JJ, M1, NP
      DOUBLE PRECISION L, STEP, STRY, T
      DOUBLE PRECISION SMFRO, SMDOT, SMFEAS, SMSTEP
      EXTERNAL SMFRO, SMDOT, SMFEAS, SMSTEP

      NP = N*P
      NIT = 0
      KKT = 0.0D0
      FEA = 0.0D0

      CALL OBJFUN(NP, X, FVAL, GF)
      CALL SMJA(N, P, X, GF, GR, WP1)
      L = SMFRO(N, P, GF) + SMFRO(N, P, GR)

      DO 20 J = 1, P
         DO 10 I = 1, P
            LAM(I,J) = 0.0D0
   10    CONTINUE
   20 CONTINUE
      CALL SMAHUR(N, P, X, GR, 0.01D0/L, 0.0D0, LAM, PROXFN,
     $            Z, XT, DX, WP1)
      CALL SMJC(N, P, X, LAM, GRAD, WP1)
      DO 40 J = 1, P
         DO 30 I = 1, N
            GRAD(I,J) = GR(I,J) + GRAD(I,J)
   30    CONTINUE
   40 CONTINUE

      DO 200 JJ = 1, MAXIT
         IF (JJ .LE. 5) THEN
            STEP = 0.01D0/L
         ELSE
            STEP = SMSTEP(SMDOT(N, P, S, S), SMDOT(N, P, S, Y),
     $                    1.0D10)
         END IF
         STEPS(JJ) = STEP

         CALL SMCOPY(N, P, X, XP)
         DO 60 J = 1, P
            DO 50 I = 1, N
               XT(I,J) = X(I,J) - STEP*GRAD(I,J)
   50       CONTINUE
   60    CONTINUE
         CALL PROXFN(NP, XT, STEP, X)
         CALL SMAMAP(N, P, X, WP1, WP2, ROW)

         DO 80 J = 1, P
            DO 70 I = 1, N
               S(I,J) = X(I,J) - XP(I,J)
   70       CONTINUE
   80    CONTINUE

         CALL OBJFUN(NP, X, FVAL, GF)
         CALL SMCOPY(N, P, GRAD, GRDP)
         CALL SMJA(N, P, X, GF, GR, WP1)

         M1 = MAX0(1, JJ-10)
         T = 0.0D0
         DO 90 K = M1, JJ
            T = T + STEPS(K)
   90    CONTINUE
         STRY = T/DBLE(JJ - M1 + 1)
         STRY = DMIN1(DMAX1(STRY, 1.0D-5/L), 1.0D10/L)

         FEA = SMFEAS(N, P, X, WP1)
         CALL SMAHUR(N, P, X, GR, STRY, 1.0D3*FEA, LAM, PROXFN,
     $               Z, XT, DX, WP1)

         CALL SMJC(N, P, X, LAM, GRAD, WP1)
         DO 110 J = 1, P
            DO 100 I = 1, N
               GRAD(I,J) = GR(I,J) + GRAD(I,J)
               Y(I,J) = GRAD(I,J) - GRDP(I,J)
  100       CONTINUE
  110    CONTINUE

         KKT = SMFRO(N, P, S)/STEP

         NIT = JJ
         FVALS(JJ) = FVAL
         KKTS(JJ) = KKT
         FEASV(JJ) = FEA
         CALL LOGFUN(JJ-1, FVAL, KKT, FEA, 0)

         IF (KKT .LT. GTOL) THEN
            CALL LOGFUN(JJ-1, FVAL, KKT, FEA, 1)
            GO TO 210
         END IF
  200 CONTINUE

  210 CONTINUE
      IF (IPOST .NE. 0 .AND. NIT .GE. 1) THEN
         CALL SMPOST(N, P, X, WP1, WP2, WP3, WEIG, ROW)
         CALL OBJFUN(NP, X, FVAL, GF)
         FEA = SMFEAS(N, P, X, WP1)
         CALL LOGFUN(NIT-1, FVAL, KKT, FEA, 2)
         FVALS(NIT) = FVAL
         KKTS(NIT) = KKT
         FEASV(NIT) = FEA
      END IF
      RETURN
      END


      SUBROUTINE SMSL21(N, P, X, MAXIT, GAM, GTOL, IPOST, OBJFUN,
     $                  LOGFUN, NIT, FVALS, KKTS, FEASV, FVAL, KKT,
     $                  FEA, GF, GR, GRAD, GRDP, S, Y, XP, XT, LAM,
     $                  WP1, WP2, WP3, WEIG, ROW)
c     SLPG for the l_{2,1} regularized objective f(X) + GAM*||X||_{2,1}.
c     Both the prox and the constraint multiplier are available in
c     closed form, so no inner iteration is needed.
      INTEGER N, P, MAXIT, IPOST, NIT
      DOUBLE PRECISION X(N,P), GAM, GTOL, FVAL, KKT, FEA
      DOUBLE PRECISION FVALS(MAXIT), KKTS(MAXIT), FEASV(MAXIT)
      DOUBLE PRECISION GF(N,P), GR(N,P), GRAD(N,P), GRDP(N,P)
      DOUBLE PRECISION S(N,P), Y(N,P), XP(N,P), XT(N,P)
      DOUBLE PRECISION LAM(P,P), WP1(P,P), WP2(P,P), WP3(P,P)
      DOUBLE PRECISION WEIG(P), ROW(P)
      EXTERNAL OBJFUN, LOGFUN
      INTEGER I, J, JJ, NP
      DOUBLE PRECISION L, STEP
      DOUBLE PRECISION SMFRO, SMDOT, SMFEAS, SMSTEP
      EXTERNAL SMFRO, SMDOT, SMFEAS, SMSTEP

      NP = N*P
      NIT = 0
      KKT = 0.0D0
      FEA = 0.0D0

      CALL OBJFUN(NP, X, FVAL, GF)
      CALL SMJA(N, P, X, GF, GR, WP1)
      L = SMFRO(N, P, GF) + SMFRO(N, P, GR)

      CALL SMLM21(N, P, X, GAM, LAM)
      CALL SMJC(N, P, X, LAM, GRAD, WP1)
      DO 20 J = 1, P
         DO 10 I = 1, N
            GRAD(I,J) = GR(I,J) + GRAD(I,J)
   10    CONTINUE
   20 CONTINUE

      DO 100 JJ = 1, MAXIT
         IF (JJ .LE. 5) THEN
            STEP = 0.001D0/L
         ELSE
            STEP = SMSTEP(SMDOT(N, P, S, S), SMDOT(N, P, S, Y),
     $                    1.0D5)
         END IF

         CALL SMCOPY(N, P, X, XP)
         DO 40 J = 1, P
            DO 30 I = 1, N
               XT(I,J) = X(I,J) - STEP*GRAD(I,J)
   30       CONTINUE
   40    CONTINUE
         CALL SMPL21(N, P, XT, STEP, GAM, 1.0D-16, X)
         CALL SMAMAP(N, P, X, WP1, WP2, ROW)

         DO 60 J = 1, P
            DO 50 I = 1, N
               S(I,J) = X(I,J) - XP(I,J)
   50       CONTINUE
   60    CONTINUE

         CALL OBJFUN(NP, X, FVAL, GF)
         CALL SMCOPY(N, P, GRAD, GRDP)
         CALL SMJA(N, P, X, GF, GR, WP1)

         CALL SMLM21(N, P, X, GAM, LAM)
         CALL SMJC(N, P, X, LAM, GRAD, WP1)
         DO 80 J = 1, P
            DO 70 I = 1, N
               GRAD(I,J) = GR(I,J) + GRAD(I,J)
               Y(I,J) = GRAD(I,J) - GRDP(I,J)
   70       CONTINUE
   80    CONTINUE

         KKT = SMFRO(N, P, S)/STEP
         FEA = SMFEAS(N, P, X, WP1)

         NIT = JJ
         FVALS(JJ) = FVAL
         KKTS(JJ) = KKT
         FEASV(JJ) = FEA
         CALL LOGFUN(JJ-1, FVAL, KKT, FEA, 0)

         IF (KKT .LT. GTOL) THEN
            CALL LOGFUN(JJ-1, FVAL, KKT, FEA, 1)
            GO TO 110
         END IF
  100 CONTINUE

  110 CONTINUE
      IF (IPOST .NE. 0 .AND. NIT .GE. 1) THEN
         CALL SMPOST(N, P, X, WP1, WP2, WP3, WEIG, ROW)
         CALL OBJFUN(NP, X, FVAL, GF)
         FEA = SMFEAS(N, P, X, WP1)
         CALL LOGFUN(NIT-1, FVAL, KKT, FEA, 2)
         FVALS(NIT) = FVAL
         KKTS(NIT) = KKT
         FEASV(NIT) = FEA
      END IF
      RETURN
      END
