# NOVA-EVO v0.2 — OXCODER MENTOR

## 1. Purpose

OxCoder Mentor is an external technical reasoning component available
to NOVA-EVO.

Its purpose is to provide technical hypotheses, implementation ideas,
debugging assistance and architectural suggestions when NOVA-EVO cannot
solve a problem using its own knowledge.

OxCoder is not the main autonomous intelligence.

NOVA-EVO remains responsible for:

- deciding when help is required,
- defining the problem,
- implementing proposed solutions,
- testing the solution,
- measuring the result,
- accepting or rejecting the proposal,
- storing the verified knowledge.

## 2. Fundamental Principle

OxCoder output is a hypothesis.

It is never automatically considered correct.

The governing rule is:

```text
OxCoder suggestion
       |
       v
implementation
       |
       v
validation
       |
       v
experiment
       |
       v
measurement
       |
       +---- successful ----> verified knowledge
       |
       +---- failed --------> failure knowledge
