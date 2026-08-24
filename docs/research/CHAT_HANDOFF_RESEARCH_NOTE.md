# Chat handoff — research note on representation geometry

**Date:** 29 July 2026

## Objective

Help revise a short research note provisionally titled:

> *Geometry and Accessibility of Predictive Information in Learned
> Representations*

The note is intended for a mathematically oriented potential supervisor. It
should become a clear 4–6 page research proposal, not a complete thesis chapter
or an inflated paper claim.

## File hierarchy

Use the attached files with this priority:

1. `RESEARCH_NOTE_GEOMETRY_ACCESSIBILITY.md`

   Editable working draft. Improve its reasoning, notation, structure, and
   concision.
2. `../history/STATO_TESI_POST_CONSOLIDAMENTO_20260728.md`

   Canonical human-readable source for empirical results, caveats, and current
   project state.
3. `../history/CONSOLIDATION_20260728.md`

   Technical provenance and analysis protocol. Consult it when a numerical or
   methodological claim needs verification.
4. `plot_ladder_normalized_last_concat512.png`

   Main empirical figure for the short note.

If the short note conflicts numerically with the state or consolidation
documents, do not guess: identify the conflict explicitly.

## Scientific position

The LOB is the real system in which the phenomenon was discovered, not the
ultimate scientific subject. The general question is:

> How do supervised and self-supervised predictive objectives preserve,
> allocate, and expose predictive variables in representations of partially
> observed stochastic dynamical systems?

The desired logical structure is:

```text
empirical phenomenon
    → mathematical separation of concepts
        → minimal analytic example
            → controlled stochastic replication
                → causal geometric intervention
                    → possible transfer to the real system
```

## Established empirical findings

- At `m/D = 1/32`, horizon-JEPA final-readout normalized accessibility is
  `0.0050`, versus numerical random-subspace null `0.0563`; supervised is
  `0.8971`, versus null `0.6118`.
- Full linear directional `R²`: horizon `0.2111`, supervised `0.3756`.
- MLP directional `R²`, final readout: horizon `0.3191`, supervised `0.3881`.
- MLP directional `R²`, temporal-mean readout: horizon `0.0494`, supervised
  `0.3917`.
- Horizon final-readout directional signal is mainly in token contrasts:
  common component approximately `0.041`, contrasts approximately `0.205`,
  full concatenation approximately `0.211`.
- Horizon and supervised are both strongly anisotropic. Their difference is
  target alignment, not isotropy alone.
- Held-out depth and imbalance support task-specific alignment and possible
  co-adaptation. Timing retains a broader supervised advantage, but has
  `R²(timing | 14 training targets) = 0.1817`.

## Required epistemic cautions

- Do not say that PCA was theoretically required to beat the random null.
- Do not claim that the phenomenon is already novel before a systematic
  literature review.
- Do not say that the training objective alone has been causally identified;
  optimizer, normalization, predictor, and implicit bias may contribute.
- Do not say that JEPA and supervised contain the same information.
- The `82.23%` horizon/supervised ratio applies only to the final pooled
  readout; after temporal averaging it is `12.62%`.
- Current held-out results are not a strict native-versus-fresh estimate of
  decoder co-adaptation.
- An MLP probe is an operational lower bound on predictive performance, not a
  measure of total information.
- A simulator can establish a possible mechanism, not by itself the cause of
  the real-data observation.

## Notation to preserve

Use the notation already present in the research note:

\[
X_t=(S_t,N_t),\qquad
O_t=g(S_t,N_t,\varepsilon_t),\qquad
Y_t=h(S_{t+1:t+H},\zeta_t),
\]

\[
\mathcal H_t=(O_{t-L+1},\ldots,O_t),\qquad
G_t=f_\theta(\mathcal H_t)\in\mathbb R^{K\times S\times d},
\]

\[
Z_t^{(P)}=P(G_t)\in\mathbb R^D.
\]

Do not silently rename symbols while revising a section. Note that \(S_t\) is
the target-relevant latent component, whereas the unindexed \(S\) in
\(\mathbb R^{K\times S\times d}\) is the number of token roles.

The formal setup should distinguish:

1. the latent component \(S_t\);
2. a sufficient statistic of \(\mathcal H_t\) for predicting \(Y_t\);
3. the learned full grid \(G_t\);
4. the pooled representation \(Z_t^{(P)}\).

On real data all of the following are unknown: the latent \(S_t\), a sufficient
predictive statistic of \(\mathcal H_t\), and the Bayes risk achievable from
\(\mathcal H_t\).

## Current mathematical core

For a decoded predictor \(\widehat Y=WZ\), an invertible change

\[
Z'=TZ,\qquad W'=WT^{-1}
\]

preserves the decoded function while potentially changing covariance,
conditioning, and finite-resource accessibility. This indeterminacy is not
claimed as novel. The research question is which representative is selected by
the training objective, architecture, normalization, optimizer, and explicit
regularization.

The minimal proposition uses

\[
Y=S+\epsilon,\qquad Z_{a,b}=(aS,bN),
\]

and shows that full predictive risk is unchanged for \(a,b\neq0\), while the
first PCA direction switches from signal to nuisance according to

\[
b^2\operatorname{Var}(N)
>
a^2\operatorname{Var}(S).
\]

## Proposed programme, not completed work

The candidate controlled process is:

\[
S_{t+1}=F_SS_t+\eta_t,\qquad
N_{t+1}=F_NN_t+\xi_t,
\]

\[
O_t=g(A_SS_t+A_NN_t+\varepsilon_t),\qquad
Y_t=BS_{t+h}+\zeta_t.
\]

The first goal is a population/linear analysis with known Bayes predictor,
sufficient state, and nuisance structure. Neural replication and the matched
H0/H-VIC/H-SIG intervention come later.

Topology is optional. It should enter only through a process with known latent
topology, such as a diffusion on \(S^1\) or a torus. Do not apply topological
data analysis post-hoc to the real point cloud merely as decoration.

## Desired collaboration style

- Review one section at a time.
- Explain every mathematical object before strengthening it.
- Separate definitions, empirical estimators, and causal interpretations.
- Prefer one defensible central claim over many diagnostics.
- Identify hidden assumptions and counterexamples.
- Keep the note readable by a mathematician with no LOB background.
- When proposing a change, state whether it is conceptual, notational,
  empirical, or rhetorical.

## Immediate task

First review the formal setup and the definition of Bayes risk while preserving
the document notation. Then revise the research note section by section. Do not
start new simulations or propose a full LeJEPA implementation until the
formalism and scope have been agreed.
