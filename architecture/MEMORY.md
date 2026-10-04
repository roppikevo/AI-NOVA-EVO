# NOVA-EVO v0.2 — MEMORY

## 1. Purpose

Memory is the persistent learning layer of NOVA-EVO.

Its purpose is to allow the system to retain useful information from
previous computation, experiments, failures and successful solutions.

Memory is not limited to model weights.

NOVA-EVO must be able to learn from experience.

## 2. Memory Classes

NOVA-EVO defines four logical memory classes:

1. Short-term memory
2. Working memory
3. Long-term memory
4. Evolution memory

## 3. Short-Term Memory

Short-term memory contains information required during the current
computation.

Examples:

- current sequence,
- current hidden state,
- temporary activations,
- current task context,
- intermediate computation.

Short-term memory is normally volatile.

It may be discarded after computation finishes.

## 4. Working Memory

Working memory contains information required during an active task or
experiment.

Examples:

- current hypothesis,
- current architecture,
- current experiment state,
- current error,
- repair attempt,
- temporary measurements,
- current training progress.

Working memory may survive across multiple computation steps but does
not automatically become permanent knowledge.

## 5. Long-Term Memory

Long-term memory contains information that may be useful in future
experiments.

Examples:

- successful solutions,
- failed solutions,
- architecture patterns,
- optimization techniques,
- debugging solutions,
- training strategies,
- performance observations,
- resource-management strategies.

Long-term memory must be persistent.

## 6. Evolution Memory

Evolution memory records the history of architectural evolution.

It contains:

- generations,
- parents,
- mutations,
- candidates,
- experiment results,
- promotions,
- rejected candidates,
- failures,
- repairs,
- lineage.

Evolution memory allows NOVA-EVO to understand how an architecture
developed.

## 7. Knowledge Units

A knowledge item should contain structured information.

Conceptually:

```text
knowledge_id
type
problem
context
hypothesis
action
implementation
result
metrics
success
failure_reason
parent
generation
timestamp
reusability
