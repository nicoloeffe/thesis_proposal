# Experiment 01 — Phase I claim map

**Status:** interpretazione consolidata da artefatti congelati

**Data:** 2026-08-27

**Scope:** Phase I; nessun nuovo fit, feature extraction o training

## Domanda sperimentale

Dato un encoder congelato, quante etichette servono a un reader lineare per
recuperare i target futuri dalla sua rappresentazione?

Il confronto primario usa `last_concat512` e contrappone supervised e
horizon-JEPA sulle stesse righe etichettate. La covariance e il whitening sono
fittati sulle feature non etichettate del train; `alpha` è selezionato sulla
validation; il test fisso viene usato per la valutazione finale.

## Quantità distinte

1. **Ceiling lineare operativo.** Test R² del ridge addestrato con tutto il
   train etichettato. Non è il Bayes ceiling né una misura di informazione
   totale.
2. **Recovery normalizzata.** Per target e rappresentazione,
   `R²(budget) / R²(full train)`, se il ceiling test è almeno `0.01`.
3. **Finite-sample gap.** Recovery supervised meno recovery horizon-JEPA allo
   stesso budget.
4. **Whitening bridge.** Variazione del finite-sample gap dopo una trasformazione
   invertibile, fittata sul train e applicata prima dello stesso reader.

## Affermazioni, test e limiti

| proposizione | test operativo | risultato | stato difendibile |
|---|---|---:|---|
| supervised ha un ceiling lineare maggiore | differenza full-budget raw test R² | `0.165405`, intervallo computazionale `[0.160753, 0.168857]` | supportata per il reader dichiarato |
| horizon-JEPA è più costoso con poche etichette | differenza di recovery normalizzata | `0.546020` sulla griglia low-budget; `0.700972` ai budget decisivi | supportata, reader-relative |
| la penalità è maggiore per direzione | confronto descrittivo fra blocchi | normalizzato: `2.97×/3.57×`; raw: `1.99×/2.35×` | segno supportato; magnitudine scale-dependent; nessun interaction test |
| il conditioning media gran parte del gap | progressive whitening train-only | `55.6%` di riduzione a `k=128`; `92.6%` a `k=508` | supportata per ridge, regolarizzazione e budget dichiarati |
| a `k=508` il gap diventa statisticamente nullo | intervalli del gap | falso: entrambi i lower bound decisivi restano positivi | non supportata |
| supervised e horizon-JEPA contengono la stessa informazione | decoder universale/Bayes risk | non eseguito | non stabilita |
| supervised è end-to-end più label-efficient | conteggio di tutte le label di pretraining | non matched: supervised ha visto target-aligned labels | non stabilita |

## Semantica della classificazione tecnica

La regola storica definisce `robust` come:

```text
lower interval bound > 0  AND  mean gap >= δ
```

Quindi `k_nonrobust=508` significa che entrambi i budget decisivi non soddisfano
più il criterio composto a `δ=0.10`; non significa che gli intervalli
attraversino zero.

| δ | classificazione tecnica | interpretazione |
|---:|---|---|
| 0.05 | D | il gap residuo a full whitening non è uniformemente eliminato secondo la tassonomia |
| 0.10 | A1 | output al threshold primario preregistrato |
| 0.15 | A1 | sensitivity preregistrata |

`A1` resta quindi l'output congelato del classificatore al threshold primario,
ma è una label tecnica secondaria ed è sensibile a `δ`. La curva misurata non
cambia.

## Formulazione scientifica finale

> Nei dati, encoder e reader dichiarati, horizon-JEPA presenta un costo
> finite-sample lineare maggiore del supervised target-aligned e un ceiling
> lineare operativo inferiore. Il whitening train-only riduce fortemente il
> finite-sample gap, mostrando che la difficoltà è in larga parte dipendente
> dal condizionamento e dal reader. Phase I non identifica informazione totale,
> Bayes risk o superiorità label-efficient end-to-end.

## Artefatti sorgente

- `validation/experiment01/execution_20260730/phase1/results.parquet`
- `validation/experiment01/execution_20260730/summary/summary.json`
- `validation/experiment01/execution_20260730/summary/gap_summary_delta_005.parquet`
- `validation/experiment01/execution_20260730/summary/gap_summary_delta_010.parquet`
- `validation/experiment01/execution_20260730/summary/gap_summary_delta_015.parquet`
- `docs/results/phase1/16_critical_budget_metrics.parquet`
