# NOVA-EVO v0.2 — EXPERIMENT SYSTEM

## 1. Purpose

The Experiment System is the scientific execution layer of NOVA-EVO.

Its purpose is to transform hypotheses into measurable experiments.

Every important autonomous decision should be supported by experimental
evidence whenever practical.

The experiment system provides:

- reproducibility,
- controlled comparison,
- measurement,
- failure recording,
- resource measurement,
- lineage,
- knowledge generation.

## 2. Core Principle

An experiment must answer a question.

The fundamental structure is:

HYPOTHESIS
    |
    v
EXPERIMENT
    |
    v
MEASUREMENT
    |
    v
CONCLUSION
    |
    v
KNOWLEDGE

The system should avoid experiments that cannot change a decision.

## 3. Experiment Identity

Every experiment receives a unique ID.

Example:

```text
EXP-GEN4-0001
