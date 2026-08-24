# Stato del progetto — LOB representation / Experiment 01

Aggiornato al **2026-08-24**. Questo è il documento operativo autorevole. I
report di fase restano le fonti scientifiche canoniche per numeri, soglie e
interpretazioni preregistrate.

## Sintesi esecutiva

Il progetto ha completato la pipeline principale di Experiment 01:

| componente | stato | esito |
|---|---|---|
| bundle production e preflight | completo | tutti i gate superati |
| Phase I | completa | `A1` tecnico secondario, con robusto gap di ceiling |
| Phase II | completa | localizzazione spettrale direzionale profonda |
| Phase III-R | completa | `R3` reader-relative |
| diagnostica predicibilità `P→M` | completa | gate preregistrato `fail` |
| test repository | completi | 165 passed |

La lettura scientifica complessiva è:

1. horizon-JEPA conserva informazione direzionale, ma la organizza in modo
   fortemente sfavorevole alle direzioni di massima varianza e al pooling
   semplice;
2. il supervised rende il segnale direzionale molto più accessibile ai reader
   lineari e ai bassi budget;
3. whitening profondo riduce il gap lineare, ma la non-robustezza richiede
   whitening quasi completo;
4. un reader MLP preregistrato non elimina la difficoltà relativa: Phase III-R
   classifica il risultato come `R3`;
5. la successiva ipotesi forte secondo cui la predicibilità intrinseca ordina
   monotonamente l'allocazione spettrale non supera la soglia preregistrata.

Questi risultati sono compatibili con una geometria di accessibilità diversa
fra gli encoder; non provano perdita informativa, causalità dell'obiettivo o una
soglia SNR discontinua.

## Stato del repository

La cartella top-level `legacy/` è stata rimossa intenzionalmente il 2026-08-24.
Conteneva 85 file archiviati e non era importata dal codice attivo. Dopo la
rimozione, l'intera suite continua a passare.

Restano due moduli con nome storico:

- `experiment01/legacy.py`;
- `experiment01/phase2_legacy.py`.

Sono gate di riproduzione storica usati dalla pipeline attiva e non dipendono
dalla directory eliminata. Non vanno rimossi soltanto per il nome.

`training/historical/` conserva le definizioni compatibili con i checkpoint
pre-fix usate da alcuni probe esplorativi. Non è codice di training corrente,
ma non è neppure un duplicato eliminabile: il nome esplicita ora questo ruolo.

La documentazione non corrente è raccolta in `docs/`: protocolli e audit di
Experiment 01, snapshot storici e note di ricerca sono separati. La root
mantiene soltanto i documenti correnti, `.gitignore` e il contratto standard
`requirements.txt`; non contiene più script o JSON sciolti.

Mappa della struttura tracciata:

| percorso | responsabilità |
|---|---|
| `experiment01/` | implementazione Phase I–III e diagnostica P→M |
| `experiment01/historical/` | estrazione e gate post-P0 corretti, hashati |
| `training/` | training corrente |
| `training/historical/` | compatibilità con checkpoint esplorativi pre-fix |
| `scripts/dataset/` | builder canonico CSV→NPZ |
| `scripts/evaluation/` | probe ed orchestrazione esplorativa/storica |
| `scripts/experiment01/` | entrypoint CLI riproducibili di Experiment 01 |
| `docs/experiment01/` | contratto, protocollo e audit |
| `docs/history/` | snapshot datati, non correnti |
| `docs/research/` | note di ricerca |
| `tests/` | regressione e gate fail-closed |

L'indice documentale è `docs/README.md`. Gli output locali `out_id*` sono stati
spostati in `validation/intrinsic_dimension/`; lo stdout storico
`log.out` è ora `logs/supervised_grid_v1_stdout.log`.

Lo stato precedente alla rimozione è recuperabile da:

```text
commit  f77dc7468b10fdb7bc7272d41fcc49348341e80a
tag     project-snapshot-2026-08-24-experiment01
```

## Asset e riproducibilità

### Presenti

| asset | path | dimensione/stato |
|---|---|---|
| CSV raw canonici | `data/lobench/raw/` | 7 file, circa 6,7 GB, hash verificati |
| dataset processato | `data/lobench_processed.npz` | 8.039.246 righe, 162 MiB |
| bundle production | `validation/experiment01_bundle_20260730` | completo, circa 253 GiB |
| esecuzione Phase I–III | `validation/experiment01/execution_20260730` | completa, circa 6,1 GiB |
| readout post-P0 | `validation/readouts_v2_20260728` | completi |
| checkpoint 3 rami × 3 seed | `checkpoints/multiseed` | presenti, circa 1,8 GiB |
| diagnostica `P→M` | `validation/experiment01/predictability_allocation_20260819` | completa, circa 51 MiB |
| risultati per revisione | `docs/results/` | report/summary/figure tracciati, circa 3,7 MiB |
| checkpoint canonici | `dist/experiment01_canonical_checkpoints_ep020.tar` | 9 file, 84,2 MB, pronto per release |

Hash principali:

```text
production bundle manifest
bdded4ebd03c29d47e5dfdba106590f24763cc06bb7e6e5ea379eb4b34201c0b

Phase-II manifest
1a30b67f6739a1a0440eae1866ee55f72cddf94248e5edf336a7e605461144c2

P→M run manifest
31d348cee4374a8ee7cdd29d6d578b60a99b5f0dabca2a374a991adecfc84e61
```

### Sorgenti raw ripristinati

I sette CSV raw canonici sono stati ripristinati in `data/lobench/raw/` dopo la
rimozione della directory top-level `legacy/`. Gli SHA-256 coincidono, file per
file, con quelli registrati nell'audit del 2026-07-30:

| simbolo | SHA-256 |
|---|---|
| `sz000001` | `cfc88e926c06b87f7e82506ec0973d07afde838d1b949353c21a6c7ab049842b` |
| `sz000002` | `eaf43ffda67970fb467e38fdc0984784a94c2e141f1e90c9525d18fef77e3465` |
| `sz000651` | `527e082a61f30f42e4ce5ec117cb2d99f42b3eeb6798de4f8237a9d2b14fea59` |
| `sz000858` | `d9ad8f2f341e3868c59bcc1e382e761038ea3ddb86c1f28c89c59f8ef136b14f` |
| `sz002415` | `2c801af4e923e3abf1bc2fec35ddbc9289027e9ceb2f95d17d61975cba60073a` |
| `sz300147` | `60bfb8fee288b028f773b389066696ed18878d3e1c26ffeffdd9636738f97062` |
| `sz300750` | `7ed3d0b250871c19fb5829a4270777c028a6847845e264dfbef9541bf25ac938` |

La riproducibilità da CSV è quindi nuovamente disponibile tramite
`scripts/dataset/build_encoder_dataset_lobench.py`. Non è stata eseguita alcuna
rigenerazione: NPZ, sidecar e bundle production congelati restano invariati.
I riferimenti a `legacy/data/lobench/raw/` nei documenti datati descrivono
soltanto il vecchio percorso.

### Artefatto pre-P0 invalido ridotto

I 180 dump NPZ in
`validation/ladder_readouts_INVALID_20260728/readouts/` sono stati eliminati il
2026-08-24 dopo conferma esplicita, recuperando circa **103 GiB**. Erano prodotti
con uno split errato, non potevano alimentare risultati o claim e non erano
versionati.

Restano circa 40 MiB sotto `validation/ladder_readouts_INVALID_20260728/`:
README dell'incidente, manifest, target e output aggregati. Sono conservati
unicamente come audit dell'errore P0 e sono esclusi da ogni pipeline corrente.

## Phase I — risultato congelato

Report canonico pubblicato:
[`docs/results/phase1/REPORT_EXPERIMENT_01.md`](docs/results/phase1/REPORT_EXPERIMENT_01.md).

Il risultato principale è la convergenza di tre diagnostiche direzionali:

1. anti-allineamento spettrale: a `m/D=1/32`, horizon-JEPA recupera `0,0050`
   del proprio score direzionale lineare completo, contro null Haar `0,0563`;
2. fragilità al pooling: horizon-JEPA passa da R² `0,2199` su
   `last_concat512` a `0,0701` su `meanK`, mentre supervised rimane circa
   stabile (`0,3853 → 0,3941`);
3. specificità finite-sample: gap normalizzato `0,5460` direzionale, contro
   `0,1838` volatilità e `0,1528` timing.

Effetti da mantenere distinti:

- gap robusto di ceiling lineare operativo: `0,165405`, intervallo 95%
  `[0,160753, 0,168857]`;
- gap robusto di recovery normalizzata ai bassi budget;
- mediazione tramite whitening progressivo.

Whitening:

- `k_50gap = 128`;
- `k_nonrobust = 508`;
- a `k=128` il gap si dimezza ma resta robusto;
- la non-robustezza richiede whitening quasi completo.

Classificazione tecnica secondaria: **A1 con robusto gap di ceiling**.

## Phase II — localizzazione spettrale

Report canonico pubblicato:
[`docs/results/phase2/REPORT_EXPERIMENT_01_PHASE2.md`](docs/results/phase2/REPORT_EXPERIMENT_01_PHASE2.md).

Gate storico PCA e parity full-rank con Phase I: superati. Failure tecniche: 0.

Su `last_concat512`, massa predittiva direzionale cumulativa media:

| k | horizon-JEPA | supervised |
|---:|---:|---:|
| 8 | 0,0001 | 0,7518 |
| 16 | 0,0064 | 0,8742 |
| 32 | 0,1302 | 0,9166 |
| 64 | 0,3585 | 0,9397 |
| 128 | 0,6458 | 0,9748 |
| 256 | 0,8330 | 0,9903 |
| 508 | 1,0000 | 1,0000 |

Per horizon-JEPA direzionale, tutti i 100 sottospazi Haar superano top-PCA a
`k=8` e `k=16` in tutti e tre i seed. A `k=128/256`, top-PCA torna a dominare
il null. I controlli sono molto meno estremi a `k=8`: volatilità `0,5601`,
timing `0,8141`.

La specifica spiegazione locale “banda 17:32 povera, 33:64 informativa” non è
robusta per entrambi i rami decisivi e resta post hoc. Phase II non modifica
Phase I e non prova causalità spettrale.

Compute canonico: 718,5 secondi wall, 4,35 GiB peak RAM, zero failure.

## Phase III-R — reader e conditioning

Report canonico pubblicato:
[`docs/results/phase3r/REPORT_EXPERIMENT_01_PHASE3.md`](docs/results/phase3r/REPORT_EXPERIMENT_01_PHASE3.md).

La Phase III originale da 21.456 modelli è stata fermata prima del freeze di
selezione e prima dell'accesso al test. L'emendamento compute-feasible Phase
III-R ha congelato 1.296 modelli senza cambiare le semantiche scientifiche.

Outcome: **R3 — difficoltà persistente oltre linearità e conditioning di
secondo ordine**. L'outcome Phase-I A1 resta invariato.

Gap normalizzato direzionale ai bassi budget:

- ridge nativo congelato: `0,6975`;
- MLP nativo: `1,5301`;
- MLP full-whitened: `4,9594`.

Ceiling MLP full-budget horizon-JEPA:

- direzionale nativo: `0,3448`, pari a `0,8602` del supervised;
- direzionale full-whitened: `0,3609`, pari a `0,9119` del supervised;
- volatilità full-whitened: `0,5362`, pari a `0,9938` del supervised.

Il reader non lineare recupera parte del ceiling ma non elimina la difficoltà
finite-sample relativa. Non è lecito concludere perdita d'informazione o che il
whitening sia un intervento causale sull'encoder.

## Diagnostica predicibilità → allocazione

Report canonico pubblicato:
[`docs/results/predictability_allocation/REPORT_EXPERIMENT_01_PREDICTABILITY_ALLOCATION.md`](docs/results/predictability_allocation/REPORT_EXPERIMENT_01_PREDICTABILITY_ALLOCATION.md).

Campione deliberatamente frazionario:

- 100.000 train + 50.000 validation;
- 2,048% degli endpoint validi;
- tutti e sette i titoli;
- 1.527 stock-day train e 170 validation disgiunti;
- copertura 99,941% degli stock-day validi.

Decisione preregistrata: **fail**. Il processo è completo e senza failure; è
fallita la soglia scientifica forte `rho_horizon > 0,6` in ogni seed.

```text
rho horizon-JEPA  0,2475 / 0,2132 / 0,2059
rho supervised   -0,1348 / -0,1838 / -0,1054
Delta rho medio   0,3636  (gate superato)
low-P sotto null  2 / 2 / 3 horizon; 0 / 0 / 0 supervised
```

Quindi esiste un segnale relativo di anti-allineamento low-P, ma non una forte
relazione monotona globale fra predicibilità intrinseca e massa top-spettrale.
Il claim meccanicistico forte non è supportato; Phase I–III restano invariati.

## Contratto operativo corrente

- Non rigenerare il bundle production senza i CSV raw canonici.
- Non modificare risultati, seed, soglie o split delle fasi congelate.
- Non usare il test per scegliere alpha, whitening-k o capacità del reader.
- Conservare separati directional, volatility e timing.
- Usare `last_concat512` come readout primario e `meanK_concatS` come
  diagnostica di pooling.
- Considerare i file in `validation/` e `checkpoints/` come artefatti esterni
  hashati, non come sorgenti Git.

## Verifica software

Comando:

```bash
../rocm_env/bin/python -m pytest -q
```

Ultima verifica: **165 passed in 18,77 s** dopo l'aggiunta del packager
fail-closed dei checkpoint canonici.

Non risultano import attivi verso la directory eliminata. Le occorrenze residue
di “legacy” appartengono a moduli/gate storici o a documenti datati.

## Prossimo passo ragionevole

Prima di nuovi training o di usare l'intero dataset, il controllo più economico
è una stability analysis della diagnostica `P→M`:

1. subset annidati 25%, 50%, 75%, 100% dei 150.000 endpoint;
2. bootstrap raggruppato per stock-day;
3. stabilità di `rho`, `Delta rho` e conteggi sotto-null.

Il collo di bottiglia inferenziale sono i 17 target correlati (participation
ratio circa `8,2`), non il numero assoluto di endpoint. Una nuova estrazione più
grande è giustificata solo se la stability analysis mostra instabilità
sostanziale.
