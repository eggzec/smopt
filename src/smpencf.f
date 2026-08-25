c-----------------------------------------------------------------------
c     PENCF, a constraint dissolving penalty method for
c
c         min f(X)   subject to   X**T*X = I_p.
c
c     The search direction combines the projected gradient with a
c     penalty term BETA*JC(X, C(X)) that pushes the iterate back onto
c     the manifold, and feasibility is restored only once the iterate
c     has drifted appreciably away from it.
c-----------------------------------------------------------------------

      SUBROUTINE SMPCF(N, P, X, BETA, MAXIT, GTOL, IPOST, OBJFUN,
     $                 LOGFUN, NIT, FVALS, KKTS, FEASV, FVAL, KKT,
     $                 FEA, BETOUT, GF, GR, GC, GRP, S, Y, XP, WP1,
     $                 WP2, WP3, WEIG, ROW)
c     A negative BETA asks for the default 0.1*||grad f(X0)||_F, so an
c     explicit BETA of zero is honoured.  The value actually used is
c     returned in BETOUT.
      INTEGER N, P, MAXIT, IPOST, NIT
      DOUBLE PRECISION X(N,P), BETA, GTOL, FVAL, KKT, FEA, BETOUT
      DOUBLE PRECISION FVALS(MAXIT), KKTS(MAXIT), FEASV(MAXIT)
      DOUBLE PRECISION GF(N,P), GR(N,P), GC(N,P), GRP(N,P)
      DOUBLE PRECISION S(N,P), Y(N,P), XP(N,P)
      DOUBLE PRECISION WP1(P,P), WP2(P,P), WP3(P,P)
      DOUBLE PRECISION WEIG(P), ROW(P)
      EXTERNAL OBJFUN, LOGFUN
      INTEGER I, J, JJ, NP
      DOUBLE PRECISION L, STEP, BT
      DOUBLE PRECISION SMFRO, SMDOT, SMFEAS, SMSTEP
      EXTERNAL SMFRO, SMDOT, SMFEAS, SMSTEP

      NP = N*P
      NIT = 0
      KKT = 0.0D0
      FEA = 0.0D0

      CALL OBJFUN(NP, X, FVAL, GF)

      BT = BETA
      IF (BT .LT. 0.0D0) BT = 0.1D0*SMFRO(N, P, GF)
      BETOUT = BT

      CALL SMJA(N, P, X, GF, GR, WP1)
      CALL SMCMAP(N, P, X, WP2)
      CALL SMJC(N, P, X, WP2, GC, WP1)
      DO 20 J = 1, P
         DO 10 I = 1, N
            GR(I,J) = GR(I,J) + BT*GC(I,J)
   10    CONTINUE
   20 CONTINUE

      L = SMFRO(N, P, GF) + SMFRO(N, P, GR)

      DO 100 JJ = 1, MAXIT
         IF (JJ .LE. 3) THEN
            STEP = 0.01D0/L
         ELSE
            STEP = SMSTEP(SMDOT(N, P, S, Y), SMDOT(N, P, Y, Y),
     $                    1.0D10)
         END IF

         CALL SMCOPY(N, P, X, XP)
         DO 40 J = 1, P
            DO 30 I = 1, N
               X(I,J) = X(I,J) - STEP*GR(I,J)
   30       CONTINUE
   40    CONTINUE
         CALL SMFIX(N, P, X, WP1, WP2, ROW)

         DO 60 J = 1, P
            DO 50 I = 1, N
               S(I,J) = X(I,J) - XP(I,J)
   50       CONTINUE
   60    CONTINUE

         CALL OBJFUN(NP, X, FVAL, GF)
         CALL SMCOPY(N, P, GR, GRP)
         CALL SMJA(N, P, X, GF, GR, WP1)
         CALL SMCMAP(N, P, X, WP2)
         CALL SMJC(N, P, X, WP2, GC, WP1)
         DO 80 J = 1, P
            DO 70 I = 1, N
               GR(I,J) = GR(I,J) + BT*GC(I,J)
               Y(I,J) = GR(I,J) - GRP(I,J)
   70       CONTINUE
   80    CONTINUE

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
