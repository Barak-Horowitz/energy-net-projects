# Competitive Energy Market RL Exercise

> **AI disclosure**
>
> This exercise was designed and implemented largely with AI assistance. It is
> not polished teaching material. Visualizations,
> interactive explanations, extensive refactoring, and other classroom aids
> were deliberately left out. The code is more complicated and less readable
> than ideal; part of the exercise is seeing whether students can understand,
> use, and improve an imperfect EnergyNet-based system, which is something you will have to deal with during your project while working with human or AI partners.
> This is not a homework - the only requirement is that you submit a passing result, and while the person with the best result might receive a prize, the primary purpose is simply to check that you are ready to tackle a real project in the lab.

## Part I — The problem

A learning electricity supplier, called the **main ISO**, must earn money while
competing with another learning ISO and serving a battery-equipped household
controller, called the **PCS**.

Each ISO chooses, for every half-hour:

- the retail price charged to the PCS,
- the price paid for electricity exported by the PCS, and
- how much generation to dispatch in advance.

The PCS buys from the cheaper ISO and sells genuine excess energy to the ISO
offering the higher payment. It may shift energy through its battery, but it
cannot buy and immediately resell energy in the same half-hour.

Dispatch always costs money. Demand above dispatch requires more expensive
emergency energy. The PCS only sells excess energy when an ISO offers at least
`0.06` per unit. The ISO may resell it externally for `0.12`, leaving at most
`0.06` profit per exported unit.

All three participants learn during training:

- the PCS and competitor begin from reusable checkpoints produced by much
  longer pretraining runs, then continue adapting during the experiment;
- the competitor is a serious learned market participant, not a deliberately
  bad policy;
- the main ISO begins from new random weights and must learn within the short
  experiment budget;
- updates made during one run are discarded, so later runs start from the same
  checkpoints.

Training uses earlier chronological data. Evaluation uses later dates once,
without further learning. The combined competitive training target is **five
CPU-minutes** using one compute thread. CPU time measures how much processing a
job actually receives rather than how long it waits on the clock. This lets you
launch several algorithm or hyperparameter experiments at once without shorter
budgets for jobs sharing the machine; parallel jobs may take longer in wall
time, but each receives the same compute target. Training stops after the
current environment step and reports the exact CPU and wall time used.

**Success has one definition:** the main ISO's total held-out profit must be
strictly greater than zero. Other reported metrics explain the result but do
not affect pass/fail.

> 🏠 **A small data-history note:** the household readings came from a London household
> electricity use dataset that Michael imported, normalized to smaller values, and no longer remembers the exact source of 🙂
> Similar smart-meter household datasets are widely available online, so the
> experiment can later be repeated with a clearly documented public source.

## Part II — Code and commands

Students mainly need:

- `competitive_alternating.py`: the EnergyNet subclass and competitive step;
- `run_competitive_alternating.py`: the short experiment entry point;
- `grading/submission_template.py`: the required submission interface.

The `internal/` folder contains accounting, preprocessing, parameters,
checkpoint loading, and evaluation plumbing that students do not need to study.
The core `energy_net/` package is not modified.

### Run the provided experiment

```powershell
python -m for_students.run_competitive_alternating
```

The competitor is a demanding reference point with substantially more prior
training. You do not need to outperform it: success means that your own ISO
earns positive held-out profit.

### Create your submission

Your submission is one Python source file. Start by copying
`for_students/grading/submission_template.py` and implement:

```python
def create_model(training_environment, seed):
    return newly_initialized_model
```

The returned model must implement methods compatible with the trusted
alternating trainer:

```python
model.set_env(environment, force_reset=True)
model.learn(total_timesteps, reset_num_timesteps=False, callback=callback)
action, state = model.predict(observation, deterministic=True)
```

The file may contain the complete model architecture, algorithm, helper
classes, and hyperparameters. The model must begin from fresh initialization:
do not load checkpoints, serialized weights, companion model files, or
pretrained policies.

Your file will be reviewed before it is run. Submissions must not:

- submit a model that has already learned. `create_model()` must construct a
  brand-new model with randomly initialized parameters. It must not read a
  saved model from disk or copy learned parameters from another model;
- read held-out data or files outside the approved training inputs;
- access the network or launch subprocesses;
- monkeypatch EnergyNet, accounting, rewards, clocks, or evaluator code; or
- continue learning during held-out evaluation.

### Evaluate your submission

Run:

```powershell
python -m for_students.grading.evaluate_submission path\to\submission.py
```

The evaluator imports your file in a separate process and constructs a fresh
model. Your ISO then trains in the same alternating market as the provided
experiment. The PCS and competitor begin from their reusable checkpoints, but
your ISO always begins from new weights.

Training stops after the fixed CPU budget. The resulting policy is frozen and
evaluated once on later chronological data, without additional learning.
Actions must have the exact ISO action shape, contain only finite values, and
remain within `[0, 1]`.

The evaluator reports the CPU and wall time, financial outcomes, energy
behavior, and final pass or fail result. The included data lets you test your
approach locally. Your final submission may also be tested on different
household days that you have not seen, so your strategy should generalize
rather than memorize the provided test period.

## Part III — Environment signals and rewards

The market operates in half-hour periods grouped into simulated days. Each ISO
makes one action for the whole upcoming day, while the PCS makes one battery
action during each half-hour.

### Main ISO and competitor ISO

Both ISOs use the same observation, action, and reward structure when they are
being trained.

**Observation:** an ISO receives the predicted household consumption for every
half-hour of the upcoming day. The environment also supports additional
per-period features if they are included in the prediction data, although the
provided setup currently uses only the consumption forecast. The ISO does not
directly observe the other ISO's action or the day's realized consumption
before choosing its action.

**Action:** for every half-hour of the upcoming day, an ISO chooses:

- the price it will pay for electricity exported by the PCS;
- the retail price it will charge the PCS; and
- the amount of generation it will dispatch in advance.

The PCS buys from the ISO offering the lower retail price and sells exports to
the ISO offering the higher payment. Exact price ties split the transaction
between the two ISOs.

**Reward:** an ISO receives its total profit for the simulated day:

```text
retail revenue
+ margin from reselling PCS exports
- cost of all generation dispatched in advance
- cost of emergency energy needed when won demand exceeds dispatch
```

Unused dispatch still costs money. Dispatching too little can require expensive
emergency energy, while uncompetitive prices can cause the ISO to lose the
transaction entirely.

### PCS

The PCS controls the household battery and acts once per half-hour.

**Observation:** the PCS receives:

- its current battery level and household consumption;
- the current lowest retail price available from the two ISOs;
- the current highest payment available for household exports;
- future consumption predictions and a mask showing which predictions are
  valid within the current day; and
- the battery's remaining charge and discharge capacity.

The PCS sees the best available market prices, not each ISO's offer separately.

**Action:** one continuous battery instruction. Positive actions charge the
battery, negative actions discharge it, and an action near zero leaves it
mostly unchanged. Battery limits and household consumption determine the
actual energy exchanged with the market.

**Reward:** the PCS receives its transaction money for that half-hour, minus a
penalty when household consumption still requires grid energy. Buying energy is
a cost to the PCS, selling genuine excess energy earns money, and battery use
can shift those transactions across the day.

### Alternating training and evaluation

Training rotates between the main ISO, the competitor, and the PCS. The main
ISO learns against the competitor's current policy; the competitor then learns
against the main ISO's current policy; and the PCS continues adapting to the
prices produced by both. Their updates share the same overall CPU-time budget.

For evaluation, the trained main ISO is frozen. Fresh copies of the original
pretrained PCS and competitor are loaded, and the later chronological data is
run once without any further learning. Only the main ISO's total profit
determines pass or fail.
