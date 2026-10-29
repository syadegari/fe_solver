# Planned solver control-flow refactor

Status: deferred design note. This records the motivation and candidate approaches
for future work; it does not change the current implementation contract or select
a framework. The formulation and implementation specification remain the references
for numerical behavior.

## Motivation

Refactor the solver so that the main analysis and Newton routines more closely
reflect their mathematical algorithms. Move logging and history collection behind
observers, encapsulate adaptive timestep and event-landing logic in a stateful
controller, and isolate line-search mechanics from the Newton loop. The goal is
not to hide numerical choices or remove production checks, but to separate
algorithmic decisions from observability, bookkeeping, and control infrastructure
so that correctness, diagnostics, and performance information are preserved without
obscuring the numerical flow.

The current analysis driver and `_newton_attempt` mix these responsibilities.
Repeated history updates, timer bookkeeping, log writes, retry counters, and event
calculations make it harder to compare the code with the algorithm on paper.
Prefer a few direct interfaces and ordinary control flow over layers of wrappers.

## Observability

Candidate approach: a small observer interface for solver logging, history
collection, startup information, and timing. Existing iteration and linear-system
callbacks offer starting points. A general event bus is not required.

The numerical routines should calculate residuals, tolerances, corrections, and
acceptance decisions explicitly, then report the relevant information to observers.
Useful observation points include attempt start, iteration evaluation, linear-solve
completion, line-search completion, increment acceptance, and attempt failure.
Observers can build histories and JSON records without interleaving formatting and
dictionary updates throughout the algorithm.

Preserve failed-attempt diagnostics, underlying failure causes, optional detailed
timings, and the timing boundaries needed for meaningful performance comparisons.
Avoid unnecessary array copies or expensive work when detailed diagnostics are
disabled. Observers must not mutate material history or decide convergence,
step growth, cutback, or state commitment. Required result and restart writes
must remain tied explicitly to accepted states and scheduled events.

Time measurement is distinct from time-step control: the former observes elapsed
wall time; the latter makes numerical decisions about the next integration interval.

## Time-step control

Candidate approach: a stateful controller owning committed time, the proposed step,
attempt/cutback counters, the event schedule, and termination conditions. It should
propose an interval and respond explicitly to acceptance or recoverable failure.
The analysis driver retains responsibility for solving and committing material state.

An **extended finite-state machine** is worth considering. A few states such as
ready, attempting, finished, and failed can be combined with numerical state
variables and guarded transitions:

| Transition | Numerical action |
| --- | --- |
| Ready to attempting | Select the endpoint, respecting the proposed step, mandatory events, final time, and step limits. |
| Attempting to ready after rejection | Keep committed time unchanged, reduce the proposal, and update retry limits. |
| Attempting to ready after acceptance | Advance after successful commitment, apply the growth rule, and identify due events. |
| Ready or attempting to termination | Finish at the target time or report an exhausted limit with its underlying cause. |

Preserve the distinction between an event-limited interval and the proposal used
for future growth, exact event landing within the existing tolerance, permitted
event intervals below `dt_min`, restart-restored proposals, and attempt limits.
Preserve the existing growth threshold and cutback rules. Expose the selected
interval and reasons to the caller; do not bury them in callbacks.

The state machine is a design option, not a requirement to add a framework. An
ordinary Python controller with explicit methods and a small transition table may
be sufficient. Avoid duplicate state that can disagree with committed solver time.

## Line search

Extract backtracking mechanics into a focused routine that receives the correction,
current iterate, and trial-evaluation operation. Return the accepted length and
trial response, or a recoverable failure. Keep the merit definition, fixed search
scales, sufficient-decrease condition, reduction factor, and search limits visible
and documented.

The Newton routine should still clearly select full versus modified Newton, decide
whether a tangent is needed, prepare or reuse the KKT solver, check convergence,
and invoke line search when enabled. Retain the current full-Newton evaluation of
candidate tangents and reuse of the accepted assembly. Modified Newton retains its
initial tangent and uses residual-only candidates. Preserve the undamped path too.

Every candidate integrates from the same committed material state at the start
of the increment. Search acceptance updates the iterate, not committed history.
An invalid candidate rejects that search length; exhausting the search requests
increment cutback. Keep these different failure scopes explicit.

## Completion criteria

The refactored analysis and Newton routines should be readable alongside their
formulation algorithm boxes. Explicit steps should cover interval selection,
trial evaluation, residual checks, tangent/context preparation or reuse, correction,
optional search, convergence, verification, commitment, and failure handling.

Preserve the numerical policies, committed/trial-state contract, production checks,
backend cleanup, and MPI ownership. Keep logging and output content available,
including diagnostics on unsuccessful attempts. Update formulation and specification
descriptions alongside the refactor without introducing new constitutive or
backend-specific decisions into the global algorithms.

Implement in separately reviewable steps: observability, time-step control, then
line-search extraction, revisiting that order if dependencies warrant it. Use
bounded comparisons with the pre-refactor solver to check accepted times, solution
and state fields, iteration/search/cutback histories, event handling, restart, and
failure behavior. Timing values will vary; preserve their meaning and coverage.
