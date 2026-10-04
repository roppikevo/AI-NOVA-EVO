# NOVA-EVO v0.2 — COGNITION

## 1. Purpose

Cognition is the computational core of NOVA-EVO.

Its purpose is to transform input information into internal representations
and useful outputs while minimizing computational and memory resources.

The cognition system must be evolvable.

NOVA-EVO distinguishes between:

- immutable interfaces,
- current architecture,
- experimental architecture,
- learned parameters.

## 2. Initial Cognition

The initial cognition system is based on the NOVA state-space architecture
with causal local context processing.

The initial architecture contains:

- token embedding,
- RMS normalization,
- causal depthwise convolution,
- gated state-space processing,
- residual connections,
- output projection.

The initial model is trained from random initialization.

## 3. Current Reference Architecture

```text
Token
  |
  v
Embedding
  |
  v
RMSNorm
  |
  +----------------------+
  |                      |
  v                      v
Causal Conv          State Update
  |                      |
  |                +-----+-----+
  |                |     |     |
  |                v     v     v
  |                f     i     g
  |                |     |     |
  |                +-- State --+
  |                      |
  +----------+-----------+
             |
             v
           Fusion
             |
             v
       Output Projection
             |
             v
          Residual
             |
             v
           Output
