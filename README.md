# RSNA Knee Abnormality Detection - Simple Two Stage Solution

Simple research baseline for the [RSNA Knee Abnormality Detection competition](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection). It is structurally inspired by [Konrad Habel's 8th place RSNA Intracranial Aneurysm Detection solution](https://github.com/KonradHabel/rsna), whose original repository is MIT licensed.

The original solution uses precise aneurysm slice localizers for Stage 1. This adaptation replaces localized supervision with report-derived study-level weak supervision and attention MIL. Qwen3 is used only during training to extract labels from reports; inference is image-only.

```text
                        TRAIN ONLY
Report ---------------------------------> Qwen3
                                            |
                                            v
                                      weak labels
                                            |
                                            v

MRI Study
   |
   +-- Sagittal
   +-- Coronal
   +-- Axial
          |
          v
    2.5D windows
          |
          v
    ConvNeXt DINOv3
          |
          +--------------------+
          |                    |
          v                    v
    Attention MIL       slice features
          |                    |
          v                    v
    Stage1 logits       Stage2 Transformer
                               |
                               v
                        final 12 logits
```

Stage 1 treats one study as a bag and its 2.5D windows as instances. Target-specific attention learns which windows matter for each abnormality and mainly teaches the image encoder from weak labels. Stage 2 is the final sequence model: a small DeBERTa trained from configuration over the resulting ConvNeXt features.

Place the competition CSV files and `train_series` / `test_series` folders in `data/`. DICOMs must be under `<StudyInstanceUID>/<SeriesInstanceUID>/*.dcm` within each folder. The default `convnext_base.dinov3_lvd1689m` identifier is available in timm 1.0.30 and outputs 1024 features. Training needs the pretrained ConvNeXt weights and the DeBERTa config; model identifiers can be replaced by local paths in the script configurations.

```bash
python label_reports.py      # validate on gold reports
# Set Configuration.mode = "full" in label_reports.py, then run it again:
python label_reports.py
python preprocess_data.py
python create_folds.py
python train_stage1.py
python eval_stage1.py
python extract_features.py
python train_stage2.py
python eval_stage2.py
python eval_end_to_end.py
python inference.py
```

Edit `Configuration.fold` in the scripts to train another fold. With five folds, run Stage 1, feature extraction, and Stage 2 separately for each fold you want to evaluate. The local `data/` folder is empty in this repository; the competition data and model weights must be obtained separately.

## Scripts Order

1. `label_reports.py`

   Runs local `Qwen/Qwen3-4B` with thinking disabled and extracts strict `1`, `0`, or `null` labels from multilingual reports. `validate` measures coverage, positive rates, accuracy, F1 and AUC on gold studies; `full` supports simple CSV resume and writes `data/report_labels_qwen3.csv`.

2. `preprocess_data.py`

   Selects one representative series for each plane/fluid-sensitive slot, sorts slices along the DICOM slice normal, robustly normalizes each MRI series, and stores a compact `[6, 24, 224, 224]` uint8 cache. Missing slots are zero-filled and recorded in `slot_mask`. Raw uncompressed cache size is at most about 7.2 MB per study; the NPZ is compressed. Reruns reuse each study cache when the chosen DICOM files are unchanged (names, sizes, and modification times).

3. `create_folds.py`

   Merges metadata and Qwen labels, converts unknowns to `-1`, and replaces all weak labels with gold labels wherever gold is available. It creates deterministic round-robin folds in `data/train_5_folds.csv` and records `label_source`.

4. `train_stage1.py` / `eval_stage1.py`

   Trains and evaluates a 2.5D ConvNeXt with target-specific attention MIL. Each fixed-size study item contains two windows from each of six slots; loss and AUC ignore unknown targets. Stage 1 selects its checkpoint on an inner fold and never sees its outer evaluation fold during training or selection.

5. `extract_features.py`

   Loads `best_stage1.pth`, builds every step-1 and step-2 2.5D window, and stores float16 ConvNeXt embeddings in `features_dict.pkl` with matching `slot_dict.pkl`. Per-study files in `feature_cache/` make interrupted runs resumable; a study is recomputed if the size or modification time of its preprocessed NPZ or the Stage 1 checkpoint changes. Step distances are positions in the 24-slice cache, not guaranteed adjacent original DICOM slices; boundary indices clamp instead of wrapping.

6. `train_stage2.py` / `eval_stage2.py`

   Trains a three-layer DeBERTa from config over image embeddings plus slot embeddings, with mean and CLS pooling. Validation reports both steps; the step-1 score selects the checkpoint because inference uses step 1. Training also exports the encoder and Transformer together as `weights_inference.pth`.

7. `eval_end_to_end.py`

   Repeats the complete validation path directly from raw DICOM: series selection, physical slice sorting, normalization, 2.5D encoding, and Transformer prediction. No cached features are used.

8. `inference.py`

   Runs the same image-only path on `test.csv`, `test_series.csv`, and `test_series`, then writes the exact 12-target `submission.csv`. Reports and Qwen are not used at inference.

The model is trained and evaluated by `StudyInstanceUID` folds. For fold 0, Stage 1 trains on three folds, selects on one inner fold, and its image encoder produces features for the held-out outer fold. Stage 2 training features from the three Stage 1 training folds remain in-sample, so a train/evaluation feature-distribution shift is possible. Gold-only AUC is a noisy diagnostic on a very small set; the main validation AUC includes report-derived targets and measures agreement with those weak labels.

Slice order follows increasing patient-space position within a plane. This does not standardize medial/lateral anatomy across left and right knees; no laterality correction is part of this baseline. Long series can have large physical gaps between the 24 cached positions, so the two window steps describe neighboring cached positions rather than always neighboring original slices.

Training artifacts are written under:

```text
model/<encoder>/fold-0/
├── best_stage1.pth
├── stage1_split.json
├── log_stage1.txt
├── features_dict.pkl
├── slot_dict.pkl
├── feature_cache/          # resumable per-study embeddings
├── config_stage2.pkl
├── best_stage2.pth
├── weights_inference.pth
└── log_stage2.txt
```

The generated caches stay under ignored `data/` and `model/` paths. If preprocessing or feature-extraction code changes in a way that affects results, increment the corresponding `CACHE_VERSION` in that script (or remove its generated cache files).

## Possible next experiment

Use Stage 1 attention to identify high-attention windows and create pseudo slice-localizations. Once those are reliable, a negative report label combined with high attention or a high local score could become a hard-negative candidate analogous to `f_dict_hard.pkl` in the reference solution. Neither experiment is part of this baseline.

## License

MIT. The retained Konrad Habel copyright notice acknowledges the original MIT code whose organization and small model components were adapted here.
