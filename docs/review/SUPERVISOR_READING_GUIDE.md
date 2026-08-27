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
2. [Research note](../research/RESEARCH_NOTE_GEOMETRY_ACCESSIBILITY.md),
   sezioni 1–4 e 6–7
3. [Phase I](../results/phase1/REPORT_EXPERIMENT_01.md), sezioni iniziali
4. [Phase II](../results/phase2/REPORT_EXPERIMENT_01_PHASE2.md), “Diagnosi in
   breve”
5. [T2 token-role](../results/token_role/REPORT_EXPERIMENT_01_TOKEN_ROLE.md),
   “Result” e “Interpretation for the simulator”
6. [F16](../results/f16/REPORT_EXPERIMENT_01_F16.md), “Result” e
   “Dose-response checks”

## Risultato da portare alla discussione

Nel sistema osservato, horizon-JEPA mantiene segnale direzionale decodificabile
ma lo rende meno accessibile del supervised target-aligned a reader lineari con
poche etichette. Tre osservazioni distinte sostengono questa descrizione:

- anti-allineamento fra varianza principale e massa predittiva;
- perdita selettiva sotto media temporale;
- riduzione, ma non eliminazione immediata, del gap tramite whitening.

Un MLP aumenta il ceiling operativo horizon-JEPA. F16 mostra inoltre che alcune
proprietà di accessibilità cambiano ordinatamente con il volume di supervisione
target-aligned. Non tutte le metriche geometriche seguono però la stessa curva.

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
- “la media temporale e il complemento di ruolo misurano lo stesso fenomeno”;
- “F16 dimostra causalmente una legge universale del training supervisionato”;
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
- test software: 187 passati;
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
