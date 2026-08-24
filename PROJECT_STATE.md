# Stato scientifico e operativo — Experiment 01

Aggiornato al **2026-08-24**. Questo documento riassume lo stato corrente del
progetto. Le specifiche congelate e i report di fase restano le fonti canoniche
per soglie, procedure e risultati numerici.

## Stato complessivo

| componente | stato | esito |
|---|---|---|
| dataset sidecar e gate CSV→NPZ | completo | equivalenza verificata su 8.039.246 righe |
| split train/validation/test per stock-day | completo | disgiunto e congelato |
| bundle production e preflight | completo | tutti i gate superati |
| Phase I | completa | `A1` tecnico secondario, con robusto gap di ceiling |
| Phase II | completa | localizzazione spettrale direzionale profonda |
| Phase III-R | completa | outcome reader-relative `R3` |
| diagnostica predicibilità `P→M` | completa | gate preregistrato `fail` |
| suite software | completa | 165 test superati |

Non sono previste esecuzioni di Phase II o III aggiuntive. Encoder, bundle,
split, seed, budget, target, soglie e risultati sono congelati.

## Conclusione scientifica

Il confronto distingue contenuto predittivo, interfaccia di readout e
accessibilità finite-sample.

1. Horizon-JEPA conserva contenuto direzionale operativo, ma lo espone peggio
   del supervised ai reader lineari con pochi dati etichettati.
2. Il segnale direzionale horizon-JEPA è anti-allineato alle prime direzioni di
   covarianza ed è fragile alla media temporale.
3. Il whitening progressivo riduce il gap, ma la perdita di robustezza richiede
   una trasformazione quasi completa dello spettro.
4. Un reader MLP recupera una parte maggiore del ceiling senza eliminare la
   difficoltà relativa ai bassi budget.
5. L'ipotesi meccanicistica più forte — allocazione spettrale monotona in
   funzione della predicibilità intrinseca — non supera il gate preregistrato.

Queste osservazioni supportano una differenza nella **geometria di
accessibilità**. Non dimostrano perdita d'informazione in senso
information-theoretic, causalità dell'obiettivo, efficacia causale del
whitening o una legge generale sulle rappresentazioni self-supervised.

## Phase I — accessibilità finite-sample

Report: [REPORT_EXPERIMENT_01.md](docs/results/phase1/REPORT_EXPERIMENT_01.md).

Tre diagnostiche convergono sul blocco direzionale:

- specificità finite-sample: gap normalizzato `0,5460`, contro `0,1838` per
  volatilità e `0,1528` per timing;
- fragilità al pooling: horizon-JEPA `0,2199 → 0,0701` passando da
  `last_concat512` a `meanK_concatS`, supervised `0,3853 → 0,3941`;
- anti-allineamento fra varianza e segnale predittivo direzionale.

Vanno mantenuti distinti:

- gap robusto di ceiling lineare operativo: `0,165405`, intervallo 95%
  `[0,160753, 0,168857]`;
- gap robusto di recovery normalizzata ai bassi budget;
- mediazione del gap tramite whitening progressivo.

Il whitening a `k=128` dimezza il gap senza eliminarlo; la non-robustezza si
raggiunge a `k=508`. La classificazione `A1` è tecnica e secondaria rispetto al
risultato di specificità: **A1 con robusto gap di ceiling**.

## Phase II — localizzazione spettrale

Report: [REPORT_EXPERIMENT_01_PHASE2.md](docs/results/phase2/REPORT_EXPERIMENT_01_PHASE2.md).

Massa predittiva direzionale cumulativa media su `last_concat512`:

| k | horizon-JEPA | supervised |
|---:|---:|---:|
| 8 | 0,0001 | 0,7518 |
| 16 | 0,0064 | 0,8742 |
| 32 | 0,1302 | 0,9166 |
| 64 | 0,3585 | 0,9397 |
| 128 | 0,6458 | 0,9748 |
| 256 | 0,8330 | 0,9903 |
| 508 | 1,0000 | 1,0000 |

Per horizon-JEPA direzionale, tutti i 100 sottospazi Haar superano le top PC a
`k=8` e `k=16` in tutti e tre i seed. A `k=128/256` le top PC tornano a
dominare il null. La spiegazione locale “banda 17:32 povera, 33:64 informativa”
non è robusta in tutti i rami e rimane post hoc.

Compute canonico: 718,5 secondi wall, 4,35 GiB peak RAM, zero failure.

## Phase III-R — dipendenza dal reader

Report: [REPORT_EXPERIMENT_01_PHASE3.md](docs/results/phase3r/REPORT_EXPERIMENT_01_PHASE3.md).

Outcome: **R3 — difficoltà persistente oltre linearità e conditioning di
secondo ordine**.

Ceiling MLP full-budget horizon-JEPA:

- direzionale nativo: `0,3448`, pari a `0,8602` del supervised;
- direzionale full-whitened: `0,3609`, pari a `0,9119` del supervised;
- volatilità full-whitened: `0,5362`, pari a `0,9938` del supervised.

Il reader non lineare recupera contenuto operativo ma non elimina la difficoltà
finite-sample relativa.

## Diagnostica predicibilità → allocazione

Report:
[REPORT_EXPERIMENT_01_PREDICTABILITY_ALLOCATION.md](docs/results/predictability_allocation/REPORT_EXPERIMENT_01_PREDICTABILITY_ALLOCATION.md).

La diagnostica usa deliberatamente un campione frazionario ma ampio e
stratificato:

- 100.000 endpoint train e 50.000 validation;
- tutti e sette i titoli;
- 1.527 stock-day train e 170 validation disgiunti;
- 99,941% degli stock-day validi rappresentati.

Decisione preregistrata: **fail**, senza failure tecniche.

```text
rho horizon-JEPA  0,2475 / 0,2132 / 0,2059
rho supervised   -0,1348 / -0,1838 / -0,1054
Delta rho medio   0,3636
low-P sotto null  2 / 2 / 3 horizon; 0 / 0 / 0 supervised
```

Il segnale relativo non basta a sostenere una relazione monotona forte fra
predicibilità intrinseca e massa top-spettrale.

## Asset canonici

| asset | percorso locale | stato |
|---|---|---|
| CSV raw | `data/lobench/raw/` | 7 file, circa 6,7 GB, hash verificati |
| dataset processato | `data/lobench_processed.npz` | 8.039.246 righe, 162 MiB |
| bundle production | `validation/experiment01_bundle_20260730` | completo, circa 253 GiB |
| output completi | `validation/experiment01/execution_20260730` | completi, circa 6,1 GiB |
| checkpoint multiseed | `checkpoints/multiseed` | 3 bracci × 3 seed canonici |
| risultati Git | `docs/results/` | report, figure e metadata, circa 3,8 MiB |
| archivio checkpoint | `dist/experiment01_canonical_checkpoints_ep020.tar` | 9 file, 84.213.760 byte |

Hash principali:

```text
bundle manifest
bdded4ebd03c29d47e5dfdba106590f24763cc06bb7e6e5ea379eb4b34201c0b

Phase-II manifest
1a30b67f6739a1a0440eae1866ee55f72cddf94248e5edf336a7e605461144c2

P→M manifest
31d348cee4374a8ee7cdd29d6d578b60a99b5f0dabca2a374a991adecfc84e61

checkpoint release archive
3e268b6fa73a122399e4b420e989a4d37112e2696efe55b4bf095892ab82ed06
```

## Struttura del codice

| percorso | responsabilità |
|---|---|
| `experiment01/` | implementazione delle tre fasi e della diagnostica P→M |
| `experiment01/reference/` | gate congelati di equivalenza e riproduzione |
| `training/` | training canonico dei tre bracci |
| `scripts/dataset/` | builder CSV→NPZ |
| `scripts/experiment01/` | CLI delle analisi congelate |
| `scripts/artifacts/` | verifica e packaging dei checkpoint |
| `docs/experiment01/` | specifiche, protocollo e audit |
| `docs/results/` | risultati leggeri pubblicabili |
| `docs/research/` | formulazione teorica |
| `tests/` | regressione e gate fail-closed |

## Contratto operativo

- Non modificare encoder, budget, seed, split, alpha grid, whitening-k, target,
  eligibility o soglie delle analisi congelate.
- Non usare il test per selezionare iperparametri.
- Conservare separati directional, volatility e timing.
- Usare `last_concat512` come readout primario e `meanK_concatS` come controllo
  dell'interfaccia di pooling.
- Trattare `data/`, `validation/`, `checkpoints/` e `dist/` come artefatti
  esterni hashati, non come sorgenti Git.

## Verifica

```bash
python -m pip install -r requirements.txt
python -m pytest -q
sha256sum -c docs/results/SHA256SUMS
```

Guida completa: [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).
