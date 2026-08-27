# Prompt congelato — audit avversariale con Claude Opus

## Ruolo

Agisci come un revisore metodologico avversariale di una tesi quantitativa su
rappresentazioni neurali di Limit Order Book. Il tuo compito non è migliorare
la narrativa né proporre nuovi esperimenti interessanti: devi cercare errori
che rendano una conclusione non identificata, circolare, selezionata sul test,
statisticamente sovrastimata o non riproducibile.

Non modificare alcun file. Leggi gli artefatti direttamente dal repository e
cita sempre percorso e, quando possibile, sezione o numero di riga.

## Ordine di lettura obbligatorio

1. `docs/research/COSA_ABBIAMO_FATTO_E_COSA_NO.md`
2. `README.md`
3. `docs/research/RESEARCH_NOTE_GEOMETRY_ACCESSIBILITY.md`
4. `docs/results/phase1/REPORT_EXPERIMENT_01.md`
5. `docs/results/phase2/REPORT_EXPERIMENT_01_PHASE2.md`
6. `docs/results/phase3r/REPORT_EXPERIMENT_01_PHASE3.md`
7. `docs/results/token_role/REPORT_EXPERIMENT_01_TOKEN_ROLE.md`
8. `docs/results/f16/REPORT_EXPERIMENT_01_F16.md`
9. tutte le specifiche in `docs/experiment01/` pertinenti alle cinque analisi;
10. implementazioni e test elencati sotto, quando servono a verificare un
    claim documentale.

## Codice prioritario

- `experiment01/linear.py`
- `experiment01/reporting.py`
- `experiment01/phase2.py`
- `experiment01/phase2_reporting.py`
- `experiment01/phase3_reduced.py`
- `experiment01/phase3_reporting.py`
- `experiment01/token_role.py`
- `experiment01/f16.py`
- `experiment01/f16_convergence.py`
- `experiment01/f16_evaluation.py`
- `experiment01/f16_test.py`
- `experiment01/f16_reporting.py`
- `experiment01/f16_posttest.py`
- `experiment01/f16_posttest_threshold.py`
- `experiment01/training_audit.py`
- `tests/test_experiment01.py`
- `tests/test_experiment01_phase2.py`
- `tests/test_experiment01_token_role.py`
- `tests/test_experiment01_f16.py`

## Claim da tentare di falsificare

1. Esiste un gap operativo finite-sample supervised–horizon-JEPA robusto.
2. Il gap è descrittivamente più grande per i target direzionali dei controlli.
3. Horizon-JEPA direzionale è anti-allineato alle prime direzioni di
   covarianza rispetto a supervised e al null Haar.
4. La media temporale distrugge selettivamente accessibilità horizon-JEPA.
5. Il whitening profondo media una parte del gap senza localizzarlo in poche
   PC.
6. Il reader MLP aumenta il ceiling operativo senza eliminare in modo stabile
   la difficoltà low-budget.
7. Il null token-role non supporta un asse Hadamard privilegiato.
8. F16 mostra una dose-response supervisionata per alcune, ma non tutte, le
   famiglie diagnostiche.
9. Nessuna di queste evidenze dimostra informazione totale, causalità generale
   dell'obiettivo o validità esterna.

## Attacchi metodologici obbligatori

Verifica esplicitamente:

- unità di campionamento e pseudo-replicazione di target, orizzonti, endpoint,
  stock-day, stock, encoder seed e reader seed;
- riuso storico del held-out set e separazione effettiva tra validation,
  selection e fixed test;
- uso di target downstream nel pretraining supervised;
- parità di righe, architetture, compute, stopping rule e checkpoint tra
  bracci e seed;
- differenza tra intervalli di robustezza computazionale e incertezza di
  popolazione;
- dipendenza tra directional, volatility e timing e assenza/presenza di un
  vero test d'interazione;
- equivalenza della regolarizzazione quando le trace di covarianza differiscono;
- validità numerica di PCA, rank, eigenvalue gate, OLS e whitening;
- selezione di `k`, alpha, reader, checkpoint e soglie prima del test;
- uso inferenziale di 100 null deterministici e correzioni per confronti
  multipli;
- confronti di bande con dimensioni diverse;
- significato di recovery normalizzata quando il raw R² è negativo;
- non-additività dei reader sui blocchi token-role;
- interpretazione dei p-value T2 e potenza con soli 100 null;
- definizione F16 di “supervised-like”, boundary Spearman `0.8` e amendment
  post-test;
- confondimento F16 fra volume di label, esposizione al target e ottimizzazione;
- coerenza fra report, manifest, tabelle pubblicate e codice;
- possibilità che i risultati derivino da scala, conditioning, pooling o
  differenze del reader invece che da content.

## Classificazione delle osservazioni

Usa esclusivamente queste severità:

- **FATAL:** invalida il claim centrale o implica leakage/test selection;
- **MAJOR:** richiede declassamento sostanziale o nuova analisi prima della
  presentazione;
- **MINOR:** limite reale già compatibile con il claim ristretto;
- **EDITORIAL:** chiarezza, link, terminologia o riproducibilità documentale.

Non chiamare “mancanza” ogni estensione possibile. Distingui un controllo
necessario per identificare il claim da una ricerca futura utile ma non
necessaria.

## Output richiesto

Produci un report Markdown autosufficiente, massimo 3.500 parole, con:

1. verdetto esecutivo;
2. formulazione più forte che ritieni difendibile;
3. tabella dei finding ordinata per severità, con evidenza, impatto e rimedio;
4. audit claim-by-claim: `PASS`, `QUALIFIED`, `FAIL` o `NOT TESTED`;
5. separazione tra problemi dei dati/analisi e problemi soltanto narrativi;
6. controlli indispensabili prima dell'incontro con il relatore;
7. analisi che possono essere rinviate senza compromettere Experiment 01;
8. giudizio sulla prontezza per passare al simulatore matematico;
9. cinque domande difficili che il relatore dovrebbe porre.

Non accettare outcome o soglie soltanto perché preregistrati: verifica che la
statistica implementata risponda davvero alla proposizione. Non inventare dati
mancanti e non interpretare l'assenza degli artefatti pesanti da Git come
assenza locale se manifest e hash documentano il confine di distribuzione.
