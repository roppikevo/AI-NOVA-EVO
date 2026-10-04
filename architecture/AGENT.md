# NOVA-EVO v0.2 — AUTONOMOUS AGENT

## 1. Purpose

The Autonomous Agent is the operational controller of NOVA-EVO.

It coordinates:

- Cognition,
- Memory,
- Evolution,
- Resource Manager,
- Experiment Engine,
- Knowledge Store,
- OxCoder Mentor.

The agent transforms goals into executable experiments.

The agent must be capable of operating without manual intervention
inside its defined permissions and resource limits.

## 2. Fundamental Loop

The agent operates according to:

GOAL
  |
  v
OBSERVE
  |
  v
UNDERSTAND
  |
  v
SEARCH MEMORY
  |
  v
PLAN
  |
  v
IMPLEMENT
  |
  v
VALIDATE
  |
  v
EXPERIMENT
  |
  v
MEASURE
  |
  v
LEARN
  |
  v
DECIDE NEXT ACTION
  |
  +----------------------+
                         |
                         v
                      REPEAT

## 3. Goal

Every autonomous cycle must have an explicit goal.

Examples:

- improve validation loss,
- reduce VRAM usage,
- increase training speed,
- test a new architecture,
- repair a failed candidate,
- investigate a performance regression,
- discover a better resource configuration.

The goal must be recorded.

## 4. Observation

Before acting, the agent should inspect:

- current generation,
- current parent,
- available resources,
- previous experiments,
- recent failures,
- relevant knowledge,
- current architecture,
- current candidate state.

The agent should avoid acting blindly.

## 5. Knowledge Search

Before generating a solution, the agent searches existing knowledge.

Possible outcomes:

KNOWN

A validated solution already exists.

RELATED

A related solution exists and may be adapted.

UNKNOWN

No useful previous solution exists.

The agent should prefer validated knowledge when appropriate.

## 6. Planning

The agent creates an explicit action plan.

A plan should contain:

- objective,
- hypothesis,
- proposed change,
- expected effect,
- resource estimate,
- validation method,
- experiment budget,
- success criteria,
- failure criteria.

The plan must be recorded before execution.

## 7. Implementation

The agent may create or modify code required for an experiment.

All autonomous code changes must initially occur inside the sandbox.

The agent must not directly modify protected generations.

## 8. Sandbox

The sandbox isolates generated work.

Conceptually:

```text
evo/engine/sandbox/
    candidate/
        source/
        tests/
        logs/
        results/
