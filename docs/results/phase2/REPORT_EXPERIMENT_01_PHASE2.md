# Experiment 01 — Phase II spectral diagnostics

## Stato e risultato

Phase II è completata come analisi diagnostica preregistrata. Phase I non è stata modificata: soglie, risultati e outcome tecnico **A1** restano congelati. Non sono stati avviati MLP, nuovi training, VICReg, simulatori o Phase III.

Il gate storico PCA post-P0 passa su 3960 celle con errore assoluto massimo `5.662e-15` (tolleranza `5.0e-10`). Il gate aggiuntivo full-rank Phase I↔Phase II passa per tutte le 18 feature e 23 target, con errore massimo `6.914e-12`.

### Diagnosi in breve

- **Specificità direzionale netta.** Su `last_concat512`, horizon-JEPA colloca soltanto 0.0001 e 0.0064 della massa direzionale cumulativa nelle prime 8 e 16 PC; supervised ne colloca rispettivamente 0.7518 e 0.8742. Per horizon-JEPA a k=8 i controlli sono molto meno estremi: volatilità 0.5601, timing 0.8141.
- **Top-PCA underperformance localizzata.** Per horizon-JEPA direzionale, tutti i 100 sottospazi Haar superano top-PCA a k=8 e k=16 in ciascuno dei tre encoder seed. La transizione è eterogenea a k=32/64 e top-PCA domina il null a k=128/256. Supervised top-PCA è al percentile 100 del null a tutte le profondità riportate.
- **Meccanismo coerente con whitening profondo, non prova causale.** Horizon-JEPA raggiunge solo 0.6458 della massa direzionale a k=128 e 0.8330 a k=256, contro 0.9748 e 0.9903 per supervised. Questa dispersione fornisce una spiegazione spettrale coerente del gap finite-sample e della profondità di whitening, ma il ponte resta descrittivo perché il whitening riscala il full rank anziché troncarlo.
- **La storia locale 17:32→33:64 non regge come spiegazione generale.** Il confronto paired non è robustamente positivo per entrambi i rami decisivi; la non-monotonia resta non spiegata da quella singola coppia di bande.

## Protocollo effettivo

- PCA/covarianza: fit esclusivamente sulle feature non etichettate del train canonico completo, separatamente per ramo × encoder seed × readout.
- Cross-covarianza e predictive mass: train canonico; direzioni oltre il rank numerico sono marcate invalide e mai invertite.
- Alpha ridge: selezionato esclusivamente su validation; test fisso usato solo per la valutazione finale.
- Null: 100 sottospazi Haar deterministici per dimensione, reader min-norm diagnostico; nessuna estrazione selezionata sul test.
- Readout primario: `last_concat512`; secondario: `meanK_concatS`. Blocchi directional, volatility e timing sempre separati.
- I confronti ridge usano `lambda = alpha * trace(covariance) / dimension` sul design etichettato pertinente.

## Localizzazione della predictive mass

La tabella seguente riporta la frazione cumulativa media della predictive mass sui target indipendenti. Gli intervalli sono gerarchici sui tre encoder seed.

| block | branch | k | mass | 95% CI |
| --- | --- | --- | --- | --- |
| directional | jepa_horizon | 8 | 0.0001 | [0.0001, 0.0001] |
| directional | jepa_horizon | 16 | 0.0064 | [0.0043, 0.0088] |
| directional | jepa_horizon | 32 | 0.1302 | [0.0620, 0.1669] |
| directional | jepa_horizon | 64 | 0.3585 | [0.3029, 0.4439] |
| directional | jepa_horizon | 128 | 0.6458 | [0.6127, 0.6785] |
| directional | jepa_horizon | 256 | 0.8330 | [0.8196, 0.8437] |
| directional | jepa_horizon | 508 | 1.0000 | [1.0000, 1.0000] |
| directional | supervised | 8 | 0.7518 | [0.7057, 0.7813] |
| directional | supervised | 16 | 0.8742 | [0.8578, 0.8863] |
| directional | supervised | 32 | 0.9166 | [0.9007, 0.9249] |
| directional | supervised | 64 | 0.9397 | [0.9337, 0.9431] |
| directional | supervised | 128 | 0.9748 | [0.9742, 0.9754] |
| directional | supervised | 256 | 0.9903 | [0.9896, 0.9911] |
| directional | supervised | 508 | 1.0000 | [1.0000, 1.0000] |
| timing | jepa_horizon | 8 | 0.8141 | [0.8044, 0.8216] |
| timing | jepa_horizon | 16 | 0.8271 | [0.8222, 0.8320] |
| timing | jepa_horizon | 32 | 0.8656 | [0.8581, 0.8791] |
| timing | jepa_horizon | 64 | 0.9129 | [0.9035, 0.9198] |
| timing | jepa_horizon | 128 | 0.9381 | [0.9357, 0.9406] |
| timing | jepa_horizon | 256 | 0.9728 | [0.9700, 0.9743] |
| timing | jepa_horizon | 508 | 1.0000 | [1.0000, 1.0000] |
| timing | supervised | 8 | 0.8022 | [0.7760, 0.8229] |
| timing | supervised | 16 | 0.8942 | [0.8740, 0.9043] |
| timing | supervised | 32 | 0.9493 | [0.9360, 0.9627] |
| timing | supervised | 64 | 0.9686 | [0.9628, 0.9765] |
| timing | supervised | 128 | 0.9860 | [0.9826, 0.9886] |
| timing | supervised | 256 | 0.9944 | [0.9937, 0.9948] |
| timing | supervised | 508 | 1.0000 | [1.0000, 1.0000] |
| volatility | jepa_horizon | 8 | 0.5601 | [0.5413, 0.5703] |
| volatility | jepa_horizon | 16 | 0.6345 | [0.5994, 0.6779] |
| volatility | jepa_horizon | 32 | 0.6901 | [0.6733, 0.6988] |
| volatility | jepa_horizon | 64 | 0.8619 | [0.8557, 0.8728] |
| volatility | jepa_horizon | 128 | 0.9364 | [0.9332, 0.9424] |
| volatility | jepa_horizon | 256 | 0.9744 | [0.9696, 0.9775] |
| volatility | jepa_horizon | 508 | 1.0000 | [1.0000, 1.0000] |
| volatility | supervised | 8 | 0.8564 | [0.8431, 0.8654] |
| volatility | supervised | 16 | 0.9116 | [0.9057, 0.9185] |
| volatility | supervised | 32 | 0.9393 | [0.9299, 0.9494] |
| volatility | supervised | 64 | 0.9581 | [0.9516, 0.9676] |
| volatility | supervised | 128 | 0.9785 | [0.9729, 0.9855] |
| volatility | supervised | 256 | 0.9924 | [0.9898, 0.9945] |
| volatility | supervised | 508 | 1.0000 | [1.0000, 1.0000] |

Le curve complete, comprese `meanK_concatS` e `jepa_masked`, sono in `predictive_mass_intervals.parquet` e nella figura 01. La predictive mass non è identificata con R² out-of-sample: è una diagnostica population-style stimata sul train e viene confrontata separatamente con ladder e bande sul test.

## Top-PCA versus sottospazi Haar

Per il blocco direzionale primario, il percentile top-PCA e la frazione dei 100 null che lo superano sono:

| branch | k | top-PCA R2 | Haar R2 mean | top percentile mean | empirical p mean | seed range p |
| --- | --- | --- | --- | --- | --- | --- |
| jepa_horizon | 8 | 0.0000 | 0.0024 | 0.0 | 1.000 | [1.00, 1.00] |
| jepa_horizon | 16 | 0.0014 | 0.0093 | 0.0 | 1.000 | [1.00, 1.00] |
| jepa_horizon | 32 | 0.0237 | 0.0300 | 27.7 | 0.723 | [0.32, 1.00] |
| jepa_horizon | 64 | 0.0773 | 0.0706 | 57.0 | 0.430 | [0.00, 0.76] |
| jepa_horizon | 128 | 0.1538 | 0.1271 | 100.0 | 0.000 | [0.00, 0.00] |
| jepa_horizon | 256 | 0.1917 | 0.1806 | 100.0 | 0.000 | [0.00, 0.00] |
| supervised | 8 | 0.2947 | 0.2152 | 100.0 | 0.000 | [0.00, 0.00] |
| supervised | 16 | 0.3408 | 0.2883 | 100.0 | 0.000 | [0.00, 0.00] |
| supervised | 32 | 0.3574 | 0.3345 | 100.0 | 0.000 | [0.00, 0.00] |
| supervised | 64 | 0.3655 | 0.3589 | 100.0 | 0.000 | [0.00, 0.00] |
| supervised | 128 | 0.3773 | 0.3726 | 100.0 | 0.000 | [0.00, 0.00] |
| supervised | 256 | 0.3824 | 0.3810 | 100.0 | 0.000 | [0.00, 0.00] |

I risultati sono riportati per ogni encoder seed in `random_null_summary.parquet`; la media tra seed qui sopra è soltanto descrittiva. Bottom-k, min-norm OLS e ridge trace-normalized tarato su validation sono conservati in `phase2_results.parquet`.

## Bande spettrali e non-monotonia k=8,16,32,64

Il test post hoc preregistrato confronta in modo paired la banda 17:32 con 33:64. Una differenza positiva significa che 33:64 contiene più predictive mass o produce R² band-only maggiore.

| branch | metric | difference | 95% CI | robust |
| --- | --- | --- | --- | --- |
| jepa_horizon | band_only_test_r2_33_64_minus_17_32 | 0.0310 | [0.0002, 0.0489] | True |
| jepa_masked | band_only_test_r2_33_64_minus_17_32 | 0.0016 | [0.0009, 0.0021] | True |
| supervised | band_only_test_r2_33_64_minus_17_32 | -0.0087 | [-0.0110, -0.0046] | False |
| jepa_horizon | predictive_mass_fraction_33_64_minus_17_32 | 0.1046 | [-0.0210, 0.2176] | False |
| jepa_masked | predictive_mass_fraction_33_64_minus_17_32 | 0.0152 | [0.0074, 0.0214] | True |
| supervised | predictive_mass_fraction_33_64_minus_17_32 | -0.0194 | [-0.0303, -0.0098] | False |

Conclusione della verifica: **non supportata in modo robusto per entrambi i rami decisivi; resta una spiegazione post hoc non confermata**. Questa diagnosi non modifica l’interpretazione né l’outcome di Phase I. I risultati per encoder seed sono in `nonmonotonicity_per_encoder.parquet`; tutte le bande, leave-band-out e null matched sono in `spectral_bands.parquet`.

## Ponte con il whitening Phase I

Il ponte usa senza rifit `k_50gap = 128`, `k_nonrobust = 508` e i gap congelati a k=0,8,16,32,64,128,256,508. Il whitening parziale a 128 dimezza il gap ma non lo elimina; la non-robustezza richiede whitening quasi completo. Phase II non assume né conclude che il problema sia concentrato in poche PC.

| block | budget | k | Phase-I gap |
| --- | --- | --- | --- |
| directional | 0.125 | 0 | 0.7096 |
| directional | 0.125 | 8 | 0.6487 |
| directional | 0.125 | 16 | 0.6481 |
| directional | 0.125 | 32 | 0.6666 |
| directional | 0.125 | 64 | 0.5523 |
| directional | 0.125 | 128 | 0.2949 |
| directional | 0.125 | 256 | 0.1425 |
| directional | 0.125 | 508 | 0.0355 |
| directional | 0.25 | 0 | 0.6923 |
| directional | 0.25 | 8 | 0.6667 |
| directional | 0.25 | 16 | 0.6689 |
| directional | 0.25 | 32 | 0.6863 |
| directional | 0.25 | 64 | 0.5744 |
| directional | 0.25 | 128 | 0.3272 |
| directional | 0.25 | 256 | 0.1779 |
| directional | 0.25 | 508 | 0.0687 |
| timing | 0.125 | 0 | 0.1578 |
| timing | 0.125 | 8 | 0.3911 |
| timing | 0.125 | 16 | 0.3606 |
| timing | 0.125 | 32 | 0.4157 |
| timing | 0.125 | 64 | 0.4838 |
| timing | 0.125 | 128 | 0.4117 |
| timing | 0.125 | 256 | 0.2196 |
| timing | 0.125 | 508 | 0.0838 |
| timing | 0.25 | 0 | 0.1325 |
| timing | 0.25 | 8 | 0.3067 |
| timing | 0.25 | 16 | 0.2735 |
| timing | 0.25 | 32 | 0.3129 |
| timing | 0.25 | 64 | 0.4734 |
| timing | 0.25 | 128 | 0.5671 |
| timing | 0.25 | 256 | 0.4506 |
| timing | 0.25 | 508 | 0.2183 |
| volatility | 0.125 | 0 | 0.2979 |
| volatility | 0.125 | 8 | 0.2728 |
| volatility | 0.125 | 16 | 0.2591 |
| volatility | 0.125 | 32 | 0.2777 |
| volatility | 0.125 | 64 | 0.3209 |
| volatility | 0.125 | 128 | 0.3482 |
| volatility | 0.125 | 256 | 0.2901 |
| volatility | 0.125 | 508 | 0.1585 |
| volatility | 0.25 | 0 | 0.2582 |
| volatility | 0.25 | 8 | 0.2437 |
| volatility | 0.25 | 16 | 0.2270 |
| volatility | 0.25 | 32 | 0.2349 |
| volatility | 0.25 | 64 | 0.2912 |
| volatility | 0.25 | 128 | 0.3521 |
| volatility | 0.25 | 256 | 0.3797 |
| volatility | 0.25 | 508 | 0.3409 |

La figura 04 sovrappone i gap Phase I alle frazioni cumulative di massa Phase II senza rifittare Phase I.

## Specificità e limiti

Directional, volatility e timing sono riportati separatamente e con identica procedura. `meanK_concatS` resta l’analisi secondaria dell’interazione con la fragilità al pooling. Le bande e i null sono diagnostiche descrittive; non introducono nuove soglie decisionali e non ridefiniscono A1/A2/B/D.

Il null Haar usa il min-norm OLS diagnostico per preservare la parità con il vecchio PCA ladder post-P0. Il ridge tarato è prodotto per top-k, bottom-k, band-only e leave-band-out, ma non viene usato per selezionare o classificare estrazioni Haar.

## Compute e failure

- Runtime core interno: `702.3` s; wall time canonico esterno: `718.5` s.
- Peak RAM canonica (`GNU time -v`): `4.35` GiB. Il campionamento interno è conservato in metadata ma non viene usato come stima del picco.
- Failure tecniche: 0.
- Cache: statistiche sufficienti in coordinate PCA; nessun rifit sui 270 GB per k, banda o sottospazio.

## Artefatti

Gli artefatti canonici sono `phase2_results.parquet`, `predictive_mass.parquet`, `random_subspace_null.parquet`, `spectral_bands.parquet`, `phase1_phase2_bridge.parquet`, `failures.parquet`, `metadata.json`, le tabelle diagnostiche, le figure e `manifest.json`. Tutti gli hash sono registrati nel manifest.
