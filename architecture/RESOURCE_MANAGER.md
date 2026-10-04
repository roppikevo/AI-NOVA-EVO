# NOVA-EVO v0.2 — RESOURCE MANAGER

## 1. Purpose

The Resource Manager controls how NOVA-EVO uses available hardware.

Its objective is not maximum hardware utilization.

The objective is:

> Maximize useful result per unit of hardware resource.

The Resource Manager must dynamically manage:

- VRAM,
- RAM,
- SSD,
- CPU,
- GPU,
- compute time.

## 2. Hardware Layers

NOVA-EVO uses three primary memory tiers:

```text
HOT
 |
 v
VRAM
 |
 v
WARM
 |
 v
RAM
 |
 v
COLD
 |
 v
SSD
