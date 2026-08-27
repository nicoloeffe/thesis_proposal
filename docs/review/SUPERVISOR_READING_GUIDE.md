# Experiment 01 — guida di lettura per il relatore

## Obiettivo dell'incontro

Experiment 01 è empiricamente completo. L'obiettivo dell'incontro non è
decidere se eseguire un'altra griglia LOB, ma concordare la formulazione
matematica del successivo sistema controllato.

La domanda centrale è:

> Come separare formalmente il contenuto predittivo invariante dalla sua
> accessibilità, che dipende da coordinate, pooling, reader e campione?

## Lettura essenziale — circa 30 minuti

1. [Cosa abbiamo fatto e cosa no](../research/COSA_ABBIAMO_FATTO_E_COSA_NO.md)
2. [Phase-I claim map](PHASE1_CLAIM_MAP.md)
3. [Research note](../research/RESEARCH_NOTE_GEOMETRY_ACCESSIBILITY.md),
   sezioni 1–4 e 6–7
4. [Phase I](../results/phase1/REPORT_EXPERIMENT_01.md), sezioni iniziali
5. [Phase-II claim map](PHASE2_CLAIM_MAP.md)
6. [Phase II](../results/phase2/REPORT_EXPERIMENT_01_PHASE2.md), “Diagnosi in
   breve”
7. [Phase-III-R claim map](PHASE3R_CLAIM_MAP.md)
8. [T2 token-role](../results/token_role/REPORT_EXPERIMENT_01_TOKEN_ROLE.md),
   “Result” e “Interpretation for the simulator”
9. [F16 claim map](F16_CLAIM_MAP.md)
10. [F16](../results/f16/REPORT_EXPERIMENT_01_F16.md), “Result” e
   “Minimum-budget saturation”

## Risultato da portare alla discussione

Nel sistema osservato, horizon-JEPA mantiene segnale direzionale decodificabile
ma lo rende meno accessibile del supervised target-aligned a reader lineari con
poche etichette. Tre osservazioni distinte sostengono questa descrizione:

- anti-allineamento fra varianza principale e massa predittiva;
- perdita selettiva sotto media temporale;
- riduzione del `55,6%` a `k=128` e del `92,6%` a `k=508` tramite whitening.

Il gap a `k=508` resta positivo: `k_nonrobust` indica il mancato raggiungimento
del criterio composto di effect size a `δ=0,10`, non una confidence interval
che attraversa zero. Al threshold primario `δ=0,10` il classificatore tecnico
restituisce `A1`, ma la sensitivity preregistrata produce `D` a `δ=0,05` e
`A1` a `δ=0,15`; la label resta scientificamente secondaria.

Un MLP aumenta il ceiling operativo horizon-JEPA all'86–91% del supervised.
La parte low-budget di Phase III-R non identifica però la persistenza oltre il
conditioning: dopo whitening entrambi i rami hanno fit sistematicamente
negativi al budget minimo. F16 mostra inoltre una transizione ampia già con lo
`0,108%` del train: i gap label-matched sono robustamente positivi, ma la legge
smooth al volume non sopravvive alla deduplicazione delle famiglie e soltanto
Axis A è strettamente monotona in tutti i seed.

## Correzione scientifica già incorporata

Il confronto storico fra componente token-role comune 128D e complemento 384D
non era dimension-matched e i relativi R² non erano additivi. Il successivo
null Haar strutturato non trova l'asse all-ones o il complemento eccezionali in
tutti i seed. Il simulatore non deve quindi assumere un meccanismo relazionale
Hadamard privilegiato.

## Claim da non usare

- “JEPA perde informazione”;
- “il segnale vive in poche componenti profonde”;
- “il whitening corregge causalmente l'encoder”;
- “R3 dimostra una difficoltà persistente oltre il conditioning”;
- “la media temporale e il complemento di ruolo misurano lo stesso fenomeno”;
- “F16 dimostra causalmente una legge universale del training supervisionato”;
- “F16 dimostra una dose-response smooth al volume di label”;
- “il test è una conferma esterna incontaminata”.

## Decisioni richieste al relatore

| decisione | alternativa A | alternativa B |
|---|---|---|
| nozione di contenuto | sufficienza/Bayes risk | equivalenza sotto decoder ricchi |
| primo modello | sistema lineare analitico | piccola rete fin dall'inizio |
| oggetto da derivare | soluzione selezionata dall'obiettivo | bound finite-sample del reader |
| primo intervento | persistenza e nuisance variance | H0/H-VIC/H-SIG |
| estensione topologica | rinviata | inclusa come secondo asse |

La raccomandazione corrente è partire dal sistema lineare con stato predittivo
e nuisance separati, Bayes predictor noto e famiglia di fattorizzazioni
encoder–predictor esplicita. Solo dopo avere isolato la selezione della
geometria conviene introdurre una piccola rete o un regolarizzatore.

## Stato di riproducibilità

- dataset canonico: 8.039.246 endpoint, equivalenza CSV→NPZ verificata;
- split stock-day: disgiunto e hashato;
- encoder canonici: tre bracci × tre seed;
- Phase I, II, III-R, T2 e F16: complete;
- revisione F16 read-only: risultati congelati, flag smooth corretto dopo
  deduplicazione;
- test software: 190 passati;
- risultati leggeri e checksum: versionati in Git;
- dataset, bundle e checkpoint: esterni, hashati e documentati.

## Approfondimento tecnico

Per un audit completo seguire, nell'ordine:

1. [specifica Phase I](../experiment01/SPEC_EXPERIMENT_01_SAMPLE_EFFICIENCY_20260730.md);
2. [contratto di implementazione](../experiment01/EXPERIMENT_01_IMPLEMENTATION.md);
3. [training audit](../experiment01/TRAINING_PROTOCOL.md);
4. [specifica Phase III-R](../experiment01/SPEC_EXPERIMENT_01_PHASE3_READER_ACCESSIBILITY_20260801.md);
5. [specifica T2](../experiment01/SPEC_EXPERIMENT_01_TOKEN_ROLE_MATCHED_NULL_20260826.md);
6. [specifica F16](../experiment01/SPEC_EXPERIMENT_01_F16_LABEL_MATCHED.md);
7. [riproducibilità](../REPRODUCIBILITY.md) e inventario checksum.
