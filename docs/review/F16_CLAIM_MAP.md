# Experiment 01 — Mappa delle affermazioni F16

## 1. Domanda scientifica

F16 chiede che cosa accade alla rappresentazione quando encoder con la stessa
architettura ricevono quantità crescenti di supervisione target-aligned. Sono
stati addestrati quattro budget per tre seed. Il test era già fissato e viene
aperto una sola volta dopo avere congelato checkpoint, reader alpha e
whitening-k.

F16 non identifica il solo effetto causale del numero di label: insieme al
volume cambiano l'esposizione al target e la traiettoria di ottimizzazione. È
quindi una diagnostica controllata del sistema studiato, non una legge
universale della supervisione.

## 2. I due assi di valutazione

- **Axis A, label-matched:** per ogni encoder F16 il reader usa lo stesso
  manifest etichettato adoperato per addestrare l'encoder. Il confronto con
  horizon-JEPA usa quel medesimo manifest. È il confronto primario.
- **Axis B, fixed-reader:** tutti gli encoder sono letti con il budget fisso
  `b_16`. Isola una diagnostica della rappresentazione, ma non è label-matched
  end-to-end.

Le quantità geometriche — proiezioni di ruolo, predictive mass, perdita sotto
pooling e whitening — appartengono ad Axis B. Nessuna di esse misura da sola il
contenuto informativo totale.

## 3. Claim primario — gap label-matched

### Test

Per ciascuno dei quattro budget e dei tre encoder seed, si confronta il test
R² direzionale su `last_concat512` fra F16-supervised e horizon-JEPA. Gli
intervalli raggruppano prima stock e poi stock-day; una sensibilità separata
lascia fuori uno stock alla volta.

### Risultato

- tutti i 12 gap F16-supervised meno horizon-JEPA sono positivi;
- tutti i 12 intervalli grouped al 95% escludono zero;
- tutti gli 84 gap leave-one-stock-out sono positivi.

### Stato

**Supportato nel dataset e nel protocollo dichiarati.** La supervisione
target-aligned produce rappresentazioni con performance direzionale maggiore
del braccio horizon-JEPA nel confronto label-matched osservato. I sette stock
non costituiscono però un campione ampio di mercati indipendenti.

## 4. Claim di forma — risposta smooth al volume

### Regola storica

La specifica definiva una famiglia come ordinata quando Spearman `rho >= 0.8`
in tutti i seed e dichiarava il pattern complessivo smooth se passavano almeno
quattro famiglie su sei.

Il boundary esatto `rho=0.8` era stato serializzato come
`0.7999999999999999`. La correzione numerica, applicata 6,48 minuti dopo
l'unlock del test, ha portato il conteggio da una a quattro famiglie passanti.
È una correzione matematica valida, ma il suo effetto decisionale deve essere
mostrato esplicitamente.

### Audit correttivo

`whitening_k128` non è una sesta evidenza indipendente: replica quasi
esattamente Axis B.

- massima differenza raw R²: `0.000340`;
- correlazione dei valori raw: `0.999991`;
- escursione massima dell'intera whitening ladder: `0.000920` R².

Dopo averla deduplicata, passano tre famiglie distinte su cinque, meno delle
quattro richieste dalla regola originale. Inoltre `rho=0.8` su quattro budget
consente una inversione di rango adiacente: soltanto Axis A è strettamente
monotona in tutti e tre i seed.

### Stato

**Non supportato dopo deduplicazione.** F16 non stabilisce una dose-response
smooth e continua al volume di label.

## 5. Forma empirica osservata — transizione precoce e saturazione

Il budget minimo contiene 7.116 righe etichettate, lo `0,108%` del train
completo. Già a questo budget la frazione media del percorso
horizon→supervised è:

| famiglia distinta | percorso medio completato | intervallo sui seed |
| --- | ---: | ---: |
| Axis A | 0,578 | 0,520–0,643 |
| Axis B | 0,451 | 0,381–0,550 |
| perdita al pooling | 0,863 | 0,789–0,927 |
| role retention | 0,893 | 0,855–0,921 |
| top-k predictive mass | 0,819 | 0,764–0,892 |

Nei budget successivi compaiono plateau e piccole inversioni, alcune entro la
risoluzione del gate di convergenza. La descrizione difendibile è quindi
**transizione rapida al budget minimo seguita da saturazione**, non legge
graduata proporzionale al volume.

Il flag storico “supervised-like at low volume” passa a `b_1` (28.446 righe),
non al budget minimo. È il superamento di una soglia preregistrata, non un test
di equivalenza col supervised canonico.

## 6. Flag secondari

| flag | lettura scientifica corrente |
| --- | --- |
| supervised-like a basso volume | passa a `b_1` come regola di soglia; non equivalenza |
| smooth label-volume dependence | non supportata dopo deduplicazione (`3/5`) |
| accessibility without measured geometry change | vero solo nel senso tecnico della regola di rango; non significa “geometria invariata” |
| low-budget optimization floor | non osservato secondo il gate congelato |
| directionality-specific coadaptation | non identificata |

L'ultimo flag non può essere letto come evidenza negativa. Per volatilità i
gap di normalizzazione ceiling-minus-horizon sono `0,0163`, `0,0206` e
`0,0125`, tutti sotto il floor interpretativo post-hoc `0,05`; i rapporti
ceiling-scaled sono quindi instabili.

## 7. Cosa F16 permette di affermare

La formulazione più forte compatibile con l'audit è:

> Nel sistema studiato, una quantità molto piccola di supervisione
> target-aligned è associata a un rapido spostamento di accessibilità e di
> diverse diagnostiche geometriche verso il supervised. Gli encoder F16
> superano horizon-JEPA nei confronti label-matched osservati. I quattro
> budget non stabiliscono però una legge smooth del volume, né isolano il
> numero di label da esposizione al target e ottimizzazione.

Per il simulatore questo suggerisce di includere la possibilità di una
selezione geometrica precoce o quasi-discontinua. Non va imposto come fatto un
parametro continuo che interpoli monotonamente tutte le proprietà fra JEPA e
supervised.

## 8. Integrità e artefatti

La revisione è una reaggregazione deterministica, in sola lettura, dei file
congelati. Non modifica risultati, checkpoint, alpha, soglie, selezioni o
l'outcome Phase-I `A1`.

- report corretto: `docs/results/f16/REPORT_EXPERIMENT_01_F16.md`;
- audit decisionale: `docs/results/f16/f16_corrective_reanalysis.json`;
- audit delle famiglie: `docs/results/f16/f16_family_audit.parquet`;
- saturazione al budget minimo: `docs/results/f16/f16_saturation_table.parquet`;
- manifest correttivo: `docs/results/f16/f16_corrective_manifest.json`.

La `f16_summary.json` originale resta conservata come output tecnico storico
post-amendment.
