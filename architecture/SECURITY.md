# NOVA-EVO v0.2 — SECURITY AND AUTONOMY BOUNDARY

## 1. Purpose

The Security System defines the boundaries within which NOVA-EVO may
operate autonomously.

The goal is controlled autonomy.

NOVA-EVO must be capable of experimenting, modifying code, repairing
failures and evolving architectures without allowing an experiment to
destroy protected history or exceed defined system resources.

## 2. Core Principle

Autonomy does not mean unrestricted access.

The governing principle is:

```text
AUTONOMOUS ACTION
       |
       v
POLICY CHECK
       |
       v
RESOURCE CHECK
       |
       v
SANDBOX
       |
       v
VALIDATION
       |
       v
EXPERIMENT
       |
       v
PROMOTION
