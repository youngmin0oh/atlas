# ATLAS

Reference implementation of *ATLAS: Adaptive TDA-guided Landscape-Aware
Transistor Sizing* (ICCAD '26).

ATLAS is a Bayesian optimization loop for analog transistor sizing that reads
the topology of the design space before deciding where to sample. At each
iteration it fits one Gaussian process per specification, scores a large
candidate pool with the feasibility score

    h(x) = min_s (mu_s(x) - c_s) / sigma_s(x),

keeps the most promising fraction as the near-feasible set, and builds a Mapper
graph over it using h as the filter. The connected components of that graph are
treated as candidate feasible regions. Their number and geometry then feed the
acquisition function in three places: candidates that could merge two regions
or discover a new one get an exploration bonus, candidates inside a
heavily-sampled region get discounted, and local candidates are drawn around
region centroids in proportion to how little each region has been visited.

## Installation

```bash
pip install -r requirements.txt
```

You also need `ngspice` on the `PATH` (tested with 44.2) and the device models
for the two open PDKs. The netlists refer to them through `$PDK_ROOT`:

```bash
git clone https://github.com/google/gf180mcu-pdk
git clone https://github.com/google/skywater-pdk

export PDK_ROOT=/path/to/pdks     # must contain gf180/ and sky130A/
```

`PDK_ROOT` defaults to `eval_engines/pdk` if the variable is unset. The
expected layout is

```
$PDK_ROOT/gf180/ngspice/{design.ngspice, sm141064.ngspice}
$PDK_ROOT/sky130A/libs.tech/ngspice/corners/{tt,ff,ss,fs,sf}.spice
```

## Running

A single run, stopping at the first design that meets every target:

```bash
python run_experiment.py --circuit cascode_miller_opamp_gf180 \
    --method ATLAS --budget 500 --seed 42
```

Repeated runs of several methods on one circuit:

```bash
python run_benchmark.py --circuit cascode_miller_opamp_gf180 \
    --methods ATLAS WEIBO SCBO TuRBO MACE CMAES Random \
    --budget 500 --n-runs 10
```

Each run writes `config.json`, `iterations.jsonl` (per-iteration specs and a
`sim_time` / `gp_time` / `tda_time` / `acq_time` breakdown), `progress.log` and
`results.json` under `--log-root`.

To reproduce the component ablation of Table 5, disable one mechanism at a
time: `--no-region-discount`, `--no-centroid-guidance`, `--no-lambda-decay`.
The Mapper sensitivity sweep of Table 6 uses `--mapper-n-cubes`,
`--mapper-overlap`, `--mapper-dbscan-eps` and `--top-k-frac`.

## Circuits

| Circuit | File | PDK | Params | Specs |
|---|---|---|---|---|
| Two-stage Miller OTA | `two_stage_opamp_gf180.yaml` | GF180MCU | 7 | gain, UGBW, PM, Ibias, Vswing, Tsettle |
| Cascode Miller amp | `cascode_miller_opamp_gf180.yaml` | GF180MCU | 10 | gain, UGBW, PM, Ibias, Vswing, Tsettle |
| Comparator | `comparator_gf180.yaml` | GF180MCU | 6 | delay, power |
| LDO | `ldo_sky130.yaml` | SKY130 | 13 | Vdrop, PSRR (6), PM (2) |

A circuit file lists the parameter bounds, the target specifications as
`[low, high]` ranges, and the netlist for each process corner. The shipped
files pin `low == high`, so every run of a circuit targets the same
specifications and the runs differ only by their optimizer seed. Widening a
range makes `run_benchmark.py` draw a fresh target per run instead.

## Layout

```
topbo/
  optimizers.py   ATLAS and the baselines; the shared BO driver
  mapper.py       Mapper graph construction and bridge/frontier classification
  acquisition.py  weighted EI, the lambda schedule, the MACE ensemble
  gp_model.py     independent Matern 5/2 ARD GP per specification
  nn_model.py     probabilistic MLP ensemble, used when GP fitting gets slow
  circuit_eval.py SPICE wrapper: parameters in, specifications out
  utils.py        LHS, normalization, feasibility, figure of merit
  logger.py       per-run structured logging
eval_engines/     ngspice runner, netlists, circuit definitions
```

## Methods

`ATLAS` is the proposed method. The baselines are `WEIBO` (weighted EI with
per-spec GPs), `TuRBO` (trust-region BO with Thompson sampling), `SCBO`
(constrained TuRBO), `MACE` (EI/PI/UCB ensemble), `CMAES`, and `Random`. The
PPAAS reinforcement-learning baseline is not included here; use the authors'
own release.

## Settings

Defaults match the paper: 10 cover intervals at 0.3 overlap, DBSCAN with
`eps = 0.3` and `min_samples = 3`, a candidate pool of 50,000, the Mapper graph
refreshed every 3 iterations, an initial design of 3 points per parameter
capped at 50, an initial local radius of 0.2 decaying by 0.7 over the budget,
and the exploration weight held for the first 60% of the budget before decaying
quadratically. The threshold that separates bridge, frontier and interior
candidates is the median distance between Mapper nodes, so it adapts to the
graph rather than being tuned per circuit.

`--surrogate` selects the surrogate: `auto` (the default) starts on the GP and
falls back to the NN ensemble only once GP fitting exceeds `--gp-time-limit`,
`gp` pins the GP, and `nn` pins the ensemble. The reported experiments stay on
the GP throughout. Note that the NN path does not seed torch, so `nn` runs are
not bit-reproducible across invocations.

## Notes on this release

Two defects present while the paper experiments were run have been corrected
here, so numbers produced by this code can differ from the tables in the paper:

- The threshold epsilon* was derived from Mapper node positions in raw
  parameter units while candidate distances were measured in the normalized
  space. Every candidate therefore fell on the same side of the threshold and
  the bridge/frontier/interior split carried no signal. Node positions are now
  kept in the normalized space, as Section 3.3 describes.
- The constraint curriculum, which relaxes the targets handed to the surrogate
  while no feasible design exists, wrote into the same dictionary used to
  decide feasibility, so a run could stop on a design that met the relaxed
  targets rather than the specified ones. The curriculum now returns a separate
  working copy and feasibility, the figure of merit and early stopping are
  always judged against the target specifications.

The GP kernel is also pinned to Matern 5/2 with ARD rather than inherited from
the BoTorch default, which has since changed to RBF.

## Citation

```bibtex
@inproceedings{oh2026atlas,
  title     = {{ATLAS}: Adaptive {TDA}-guided Landscape-Aware Transistor Sizing},
  author    = {Oh, Youngmin and Won, Jihwan and Park, Yuntae and
               Hwang, Bosun and Kim, Suwan},
  booktitle = {IEEE/ACM International Conference on Computer-Aided Design (ICCAD)},
  year      = {2026},
  doi       = {10.1145/3831252.3834039}
}
```
