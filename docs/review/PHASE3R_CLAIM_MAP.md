# Experiment 01 — Mappa delle affermazioni di Phase III-R

## 1. Domanda scientifica

Phase III-R prova ad allargare il reader rispetto al ridge lineare di Phase I.
Le domande sono due e devono restare separate:

1. un MLP aumenta la performance operativa a pieno budget?
2. a basso budget, il gap supervised–horizon persiste anche dopo avere
   allargato il reader e applicato whitening completo?

La prima domanda è identificata dai risultati. La seconda, nel regime
eseguito, non lo è.

## 2. Reader e protocollo effettivi

Il reader primario congelato è:

```text
Linear(D,256) -> GELU -> Dropout(0.10) -> Linear(256,T)
```

Usa tre encoder seed, tre subset seed e tre reader seed. Il weight decay è
scelto su validation fra `0`, `1e-5`, `1e-3`; il learning rate resta fisso a
`1e-3`. I target sono standardizzati sul solo subset etichettato e le
predizioni sono riportate alla scala originale prima del calcolo di R².

Il reader non applica standardizzazione coordinate-wise all'input, né
BatchNorm né LayerNorm. Confronta coordinate native centrate con coordinate
full-whitened train-only. Questa scelta era preregistrata, ma rende il reader
non scale-invariante fra i due sistemi di coordinate.

Phase III-R usa soltanto due budget bassi:

- `b_1_4`: circa 7.116–7.798 righe, secondo encoder;
- `b_1_2`: circa 14.226–15.590 righe.

Sono formalmente eleggibili secondo la specifica definitiva del 1 agosto
(`n_rows >= 4096`). La specifica precedente del 30 luglio, nella sezione MLP
opzionale, vietava però l'interpretazione sotto 8 giorni per stock. Phase III-R
è quindi conforme all'amendment successivo, ma opera deliberatamente nel
regime severamente overparameterized che il documento precedente considerava
non interpretabile.

## 3. Claim 1 — il ceiling operativo MLP aumenta

### Test

Confrontare a `full_train` il test R² del reader MLP con il ridge congelato,
separatamente per ramo, trasformazione e blocco target.

### Risultato direzionale primario

| ramo | coordinate | MLP R² | lift sul ridge | rapporto col supervised MLP |
| --- | --- | ---: | ---: | ---: |
| horizon-JEPA | native | 0.3448 | +0.1248 | 0.8602 |
| horizon-JEPA | full-whitened | 0.3609 | +0.1408 | 0.9119 |
| supervised | native | 0.3986 | +0.0132 | 1.0000 |
| supervised | full-whitened | 0.3946 | +0.0092 | 1.0000 |

Tutti i punteggi full-budget primari sono positivi. L'MLP recupera quindi
l'86,0% della performance supervised in coordinate native e il 91,2% dopo
whitening completo.

### Stato del claim

**Pass.** Un reader non lineare specifico aumenta il ceiling operativo di
horizon-JEPA. Non è una misura di mutual information o del Bayes ceiling e non
dimostra che tutti i reader non lineari produrrebbero lo stesso risultato.

## 4. Claim 2 — il gap persiste oltre non linearità e conditioning

### Regola tecnica congelata

Il classificatore preregistrato usa il gap fra recovery normalizzate:

```text
recovery = R²(budget) / R²(full_train)
gap = recovery_supervised - recovery_horizon
```

Le medie tecniche sui due budget bassi sono:

| reader/coordinate | gap normalizzato |
| --- | ---: |
| ridge native congelato | 0.6975 |
| MLP native | 1.5301 |
| MLP full-whitened | 4.9594 |

Applicando letteralmente le soglie congelate, il classificatore restituisce
`R3`.

### Diagnostica grezza che impedisce l'interpretazione meccanicistica

A `b_1_4`:

| ramo | coordinate | R² medio | frazione R² negativi |
| --- | --- | ---: | ---: |
| supervised | native | 0.255 | 0.000 |
| horizon-JEPA | native | -0.387 | 0.966 |
| supervised | full-whitened | -0.416 | 1.000 |
| horizon-JEPA | full-whitened | -2.416 | 1.000 |

Nel confronto decisivo full-whitened, entrambi i rami sono peggiori del
predictor che usa la media test in tutte le celle. Il gap 4.9594 è quindi una
differenza fra recovery molto negative, non una misura stabile di accessibilità.

Il pattern prosegue a `b_1_2`: horizon-JEPA full-whitened ha R² medio `-1.378`
e il 100% di celle negative; supervised ha media `-0.033` e il 52,8% di celle
negative.

Poiché il reader usa un learning rate fisso e nessuna standardizzazione
coordinate-wise, il fallimento congiunto può dipendere dall'interazione fra
scala, ottimizzazione e rapporto campioni/parametri. Il confronto non isola
quindi la difficoltà della mappa downstream dalla fragilità del reader.

### Stato del claim

**Non identificato nel regime eseguito.** `R3` viene conservato come
classificazione tecnica storica della regola preregistrata, ma non è la
conclusione scientifica primaria. Non possiamo affermare che la difficoltà
persista “oltre linearità e conditioning”.

Il fatto grezzo più debole resta vero: sotto l'esatto reader congelato,
horizon-JEPA è peggiore del supervised ai budget osservati. Non sappiamo però
attribuire questa differenza alla rappresentazione invece che
all'ottimizzazione reader×coordinate.

## 5. Claim 3 — diagnostica spettrale MLP

Il reduced design esegue soltanto horizon-JEPA direzionale su tre bracci:
`band_1_127`, `band_382_508` e `full_valid_rank`.

Al full budget:

| sottospazio | MLP R² |
| --- | ---: |
| PC 1:127 | 0.3497 |
| PC 382:508 | 0.0200 |
| rango completo | 0.3704 |

Il sottospazio 1:127 raggiunge circa il 94% del full-rank MLP R². Questo non
contraddice Phase II: la predictive mass è una statistica lineare di
cross-covarianza, mentre qui il reader è non lineare. Mostra invece che la
nozione di “dove sta il segnale” è reader-relative.

### Limiti

- manca il supervised nello stesso contrasto MLP spettrale;
- mancano le bande intermedie;
- mancano volatilità e timing;
- il reduced design ha rimosso il capacity sweep;
- non è una misura di predictive mass recuperata.

Questa parte è quindi **secondaria e descrittiva**.

## 6. Conclusione scientifica difendibile

La formulazione più forte è:

> Un MLP width-256 preregistrato aumenta sostanzialmente il ceiling operativo
> direzionale di horizon-JEPA, portandolo all'86–91% del supervised. Questo
> dimostra reader-relative recoverability a pieno budget. Il comportamento
> low-budget, soprattutto dopo whitening, non identifica invece una difficoltà
> persistente oltre il conditioning, perché entrambi i rami producono R²
> sistematicamente negativi nello stesso regime. `R3` resta una label tecnica
> della regola congelata, non un claim meccanicistico.

## 7. Cosa servirebbe per identificare il claim low-budget

Un nuovo esperimento, separato da Phase III-R, dovrebbe almeno:

- standardizzare gli input per coordinate in entrambi i sistemi;
- oppure tarare learning rate e dinamica di ottimizzazione separatamente su
  validation per trasformazione;
- imporre un gate di utilizzabilità raw, per esempio escludendo classificazioni
  in cui entrambi i rami hanno R² prevalentemente negativo;
- usare un budget con rapporto campioni/parametri meno estremo.

Non è necessario eseguirlo per sostenere i risultati principali di Phase I e
Phase II.

## 8. Artefatti sorgente

- `validation/experiment01/execution_20260730/phase3_reduced/phase3_results.parquet`
- `validation/experiment01/execution_20260730/phase3_reduced/phase3_report_metrics.parquet`
- `validation/experiment01/execution_20260730/phase3_reduced/phase3_ceiling_and_lift.parquet`
- `validation/experiment01/execution_20260730/phase3_reduced/phase3_spectral_bands.parquet`
- `validation/experiment01/execution_20260730/phase3_reduced/summary.json`

Nessun modello, risultato, soglia o outcome tecnico è modificato da questa
mappa.
