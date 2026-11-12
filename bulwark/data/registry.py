"""Every dataset Bulwark knows about: where it comes from, its terms, and how to get it.

Four flags record the maintainers' reading of each dataset's terms or, where
`weights_release_basis` says so, their own dated decision. None of them is legal advice.

* `commercial_ok`: the published terms appear to permit commercial use. Every manifest
  row carries it. The commercially clean retrain is the `configs/train/*-clean.yaml`
  chain (heads, LoRA fine-tune, teacher cache, distillation): every stage sets
  `commercial_only: true`, which trains on rows with `commercial_ok=True` only, and
  refuses to start from a checkpoint or teacher features made without it (see
  docs/training.md). It covers Bulwark's data, not the encoder's upstream pretraining.
  Other datasets are used for research-only weights.
* `weights_release_ok` says whether weights trained on the dataset may be published at
  all (True / False), or that this has not been determined yet (None). A checkpoint is
  ready for release only when every dataset it was trained on is True.
  `weights_release_basis` says on what that True or False rests, with its date: the
  dataset's terms, its owners' written permission, or the maintainers' own decision
  (`MAINTAINERS_RELEASE_DECISION`, which is not a statement by the dataset's owners).
  `bulwark release-check` prints it with its verdict. True clears weights released for
  non-commercial research; a licence that allows commercial use also needs
  `commercial_ok`.
* `kaggle_storage_ok` and `azure_storage_ok` say whether a private copy may be kept on
  that third party's storage: a private Kaggle dataset, or the private Azure Blob
  container in the project's own subscription. Each provider is decided on its own;
  clearing one does not clear the other. True / False once the terms or signed agreement
  have been read and the maintainers have decided (`terms` quotes the clause, or records
  the dated decision), None until then. `STORAGE_FIELDS` maps each provider to its flag.
  infra/kaggle/stage_data.py (run by infra/kaggle/push_data.sh with `--storage kaggle`,
  and by infra/azure/upload.sh with `--storage azure`) stages the shards, feature files
  and manifest rows only of datasets cleared for that provider, unless the operator names
  one explicitly (ALLOW_UNCLEARED). scripts/purge_cloud_copies.sh applies the Azure Blob
  flag to Blob: it deletes the shard blobs of a dataset not cleared for it instead of
  replacing them.

`terms` quotes the clause relied on, with where and when it was checked. `citation` is
copied from the dataset's own page or paper, never written from memory. Every dataset that
has been ingested has one; fill it in (and `terms`) before ingesting a dataset that has not.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    name: str
    title: str
    url: str
    licence: str
    commercial_ok: bool
    access: str
    size_gb: float
    modalities: tuple[str, ...]
    # Taxonomy labels the adapter writes as ground truth (tests keep this in sync).
    provides: tuple[str, ...]
    group: str
    download_url: str = ""
    adapter: bool = False
    notes: str = ""
    weights_release_ok: bool | None = None
    # On what `weights_release_ok` rests when it is True or False (tests require one then).
    weights_release_basis: str = ""
    # Set True only after the maintainers have read the licence or signed agreement and
    # approved a private copy with that provider; `terms` records the dated decision.
    kaggle_storage_ok: bool | None = None
    azure_storage_ok: bool | None = None
    terms: str = ""
    citation: str = ""
    # Where the ingested copy came from, when it is not `download_url` or `url`, and that
    # copy's `bulwark.data.sources.base.copy_fingerprint` (hex). An adapter that checks the
    # fingerprint records `source_copy` only for a copy that matches it.
    source_copy: str = ""
    source_copy_fingerprint: str = ""


# Third-party storage a private copy may be kept on -> the DatasetSpec flag that clears it.
STORAGE_FIELDS = {"kaggle": "kaggle_storage_ok", "azure": "azure_storage_ok"}

# `weights_release_basis` of the datasets the maintainers cleared for the v0 weights
# release themselves (MODEL_CARD.md, "Release status"). It records Inferensys's decision
# only; it never stands for the dataset owners' consent.
MAINTAINERS_RELEASE_DECISION = (
    "Inferensys's decision of 2026-10-30, made on the dataset's research terms, that v0 weights "
    "trained on the dataset may be published for non-commercial research only, under CC BY-NC "
    "4.0. It is not a statement by the dataset's owners. Users must also follow the dataset's "
    "terms. If its owners object, the weights trained on it are withdrawn and the models "
    "retrained without it."
)


DATASETS: tuple[DatasetSpec, ...] = (
    # --- Driver state video -------------------------------------------------------------
    DatasetSpec(
        name="dmd",
        title="DMD - Driver Monitoring Dataset (Vicomtech)",
        url="https://dmd.vicomtech.org/",
        licence="CC BY-NC-ND 4.0, academic use",
        commercial_ok=False,
        access="registration form, then download links by email",
        size_gb=250.0,
        modalities=("rgb",),
        provides=(
            "activity",
            "attention_direction",
            "blink",
            "context",
            "drink",
            "eyes_closed",
            "face_visible",
            "modality",
            "perclos",
            "phone",
            "phone_at_ear",
            "yawn",
        ),
        group="driver_video",
        adapter=True,
        notes=(
            "41 h, 3 cameras (face, body, hands). OpenLABEL JSON annotations. Only RGB is "
            "released. Only the face camera is ingested, so the hands-on-wheel annotations "
            "are not used. alertness='microsleep' is derived from the eye-state annotations "
            "(eyes closed without a break for 1 s or more inside one clip window, 1.6 s at "
            "30 fps; a closure split across two windows can be missed) and kept as a derived "
            "label in labels_teacher, not ground truth. Participants can ask Vicomtech to remove "
            "their data; removals are applied through bulwark/data/removed_subjects.yaml."
        ),
        terms=(
            '"This dataset can only be used for academic purposes"; published under CC BY-NC-ND '
            "4.0 (dmd.vicomtech.org, checked 2026-10-29). Bulwark is developed by a company, "
            "Inferensys, which trains on DMD internally for non-commercial research. These terms "
            "do not settle whether that use is allowed, nor whether weights trained on DMD may be "
            "released for research only (including whether they count as adapted material under "
            "ND). Releasing the v0 weights trained on DMD for non-commercial research is "
            "Inferensys's decision of 2026-10-30, not a statement by Vicomtech. "
            "Storage: private copies on Kaggle (datasets visible only to the maintainers' "
            "account, used only to train on Kaggle's GPUs; not redistribution) approved by the "
            "maintainers on 2026-10-29; Azure Blob not yet approved."
        ),
        weights_release_ok=True,
        weights_release_basis=MAINTAINERS_RELEASE_DECISION,
        kaggle_storage_ok=True,
        citation=(
            "Ortega, J. D., Kose, N., Cañas, P., Chao, M.-A., Unnervik, A., Nieto, M., Otaegui, "
            "O., Salgado, L. (2020). DMD: A Large-Scale Multi-modal Driver Monitoring Dataset for "
            "Attention and Alertness Analysis. In: A. Bartoli & A. Fusiello (eds), Computer Vision "
            "— ECCV 2020 Workshops (pg. 387–405). Springer International Publishing."
        ),
    ),
    DatasetSpec(
        name="drive_and_act",
        title="Drive&Act (Fraunhofer IOSB)",
        url="https://www.driveandact.com/",
        licence="Research use only, Fraunhofer IOSB",
        commercial_ok=False,
        access="direct ZIP download, no login",
        size_gb=60.0,
        modalities=("nir", "rgb"),
        provides=(
            "activity",
            "attention_direction",
            "context",
            "drink",
            "food",
            "modality",
            "phone",
            "phone_at_ear",
        ),
        group="driver_video",
        adapter=True,
        notes=(
            "12 h, 15 drivers, 5 NIR views + Kinect. 34 mid-level activities in 3 s chunks. "
            "The 5 NIR views and Kinect colour and IR are ingested, not Kinect depth. Splits "
            "follow the official split_0 subject split; the inner-mirror view of the test "
            "drivers is a cross-view holdout, and the rest of that view is excluded."
        ),
        terms=(
            '"Copyright Fraunhofer IOSB. Usage for research only." (driveandact.com, checked '
            "2026-10-29). Nothing is said about trained models. Bulwark is developed by a "
            "company, Inferensys, which trains on Drive&Act internally for non-commercial "
            "research. These terms do not settle whether that use is allowed, nor whether weights "
            "trained on Drive&Act may be released for research only. Releasing the v0 weights "
            "trained on Drive&Act for non-commercial research is Inferensys's decision of "
            "2026-10-30, not a statement by Fraunhofer IOSB. "
            "Storage: private copies on Kaggle (datasets visible only to the maintainers' "
            "account, used only to train on Kaggle's GPUs; not redistribution) approved by the "
            "maintainers on 2026-10-29; Azure Blob not yet approved."
        ),
        weights_release_ok=True,
        weights_release_basis=MAINTAINERS_RELEASE_DECISION,
        kaggle_storage_ok=True,
        # The site asks users to cite this paper; entry from mlanthology.org (ICCV 2019).
        citation=(
            "@inproceedings{martin2019iccv-drive,\n"
            "  title     = {{Drive&Act: A Multi-Modal Dataset for Fine-Grained Driver Behavior "
            "Recognition in Autonomous Vehicles}},\n"
            "  author    = {Martin, Manuel and Roitberg, Alina and Haurilet, Monica and Horne, "
            "Matthias and Reiss, Simon and Voit, Michael and Stiefelhagen, Rainer},\n"
            "  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer "
            "Vision},\n"
            "  year      = {2019},\n"
            "  url       = {https://mlanthology.org/iccv/2019/martin2019iccv-drive/}\n"
            "}"
        ),
    ),
    DatasetSpec(
        name="uta_rldd",
        title="UTA Real-Life Drowsiness Dataset",
        url="https://sites.google.com/view/utarldd/home",
        licence="No licence stated (all rights reserved; no written permission held)",
        commercial_ok=False,
        access="Google Drive folders (5 folds, 2 parts each)",
        size_gb=111.3,
        modalities=("rgb",),
        provides=("alertness", "context", "kss", "modality"),
        group="driver_video",
        adapter=True,
        notes=(
            "60 subjects x 3 self-reported states: 0 alert (KSS 1-3), 5 low vigilance (KSS 6-7), "
            "10 drowsy (KSS 8-9), ~10 min each. Held out entirely; it is the only KSS source."
        ),
        terms=(
            "The project page states no licence or terms of use. It says 36 of the 60 "
            'participants allowed their faces to be published (its "Image Publishable" '
            'spreadsheet lists them) and that the identity of the subjects "cannot be revealed '
            'in any way"; documents that report results must cite the paper below '
            "(sites.google.com/view/utarldd/home, checked 2026-10-29)."
        ),
        citation=(
            "@inproceedings{ghoddoosian2019realistic,\n"
            "  title     = {A Realistic Dataset and Baseline Temporal Model for Early Drowsiness "
            "Detection},\n"
            "  author    = {Ghoddoosian, Reza and Galib, Marnim and Athitsos, Vassilis},\n"
            "  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern "
            "Recognition Workshops},\n"
            "  pages     = {0--0},\n"
            "  year      = {2019}\n"
            "}"
        ),
    ),
    DatasetSpec(
        name="nthu_ddd",
        title="NTHU Driver Drowsiness Detection",
        url="http://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/",
        licence="EULA, academic use",
        commercial_ok=False,
        access="signed EULA sent to NTHU CV lab",
        size_gb=30.0,
        modalities=("rgb", "nir"),
        provides=(
            "activity",
            "alertness",
            "attention_direction",
            "context",
            "eyeglasses",
            "head_nod",
            "low_light",
            "modality",
            "sunglasses",
            "yawn",
        ),
        group="driver_video",
        adapter=True,
        notes=(
            "Per-frame drowsiness/eye/head/mouth strings. Day and night (IR) scenarios. The eye "
            "track marks 'sleepy eyes', not closure, so it is kept as metadata, not as "
            "eyes_closed."
        ),
        terms="Signed EULA; its clauses on derived models and company use are not yet reviewed.",
        # The page's "Citation" section, as archived by the Internet Archive on 2026-10-10
        # (the live site refused connections on 2026-10-29). The page's quotation marks are
        # mis-encoded, so plain ones are used here; the rest is verbatim.
        citation=(
            'Ching-Hua Weng, Ying-Hsiu Lai, Shang-Hong Lai, "Driver Drowsiness Detection via a '
            'Hierarchical Temporal Deep Belief Network", In Asian Conference on Computer Vision '
            "Workshop on Driver Drowsiness Detection from Video, Taipei, Taiwan, Nov. 2016"
        ),
    ),
    DatasetSpec(
        name="yawdd",
        title="YawDD - Yawning Detection Dataset",
        url="https://ieee-dataport.org/open-access/yawdd-yawning-detection-dataset",
        licence="Research use; publication of images restricted per subject",
        commercial_ok=False,
        access="IEEE DataPort (free account)",
        size_gb=5.0,
        modalities=("rgb",),
        provides=("activity", "context", "eyeglasses", "modality", "sunglasses", "yawn"),
        group="driver_video",
        adapter=True,
        notes=(
            "Mirror and dash cameras. Mirror files are named Normal/Talking/Yawning per video; "
            "only Normal/Talking give clip-level yawn labels (negatives)."
        ),
        terms=(
            '"You can use all videos for research. Also, you can display the screenshots of some '
            '(not all) videos in your own publications." Where the "Allow picture to be shared '
            'publicly?" column of the Participants Information file says "No", pictures from '
            "that video may not be shown in publications or anywhere else publicly. No licence "
            "is named (ieee-dataport.org, checked 2026-10-29)."
        ),
        # The page's dataset citation (IEEE style) and the paper it points to, verbatim.
        citation=(
            'Shabnam Abtahi, Mona Omidyeganeh, Shervin Shirmohammadi, Behnoosh Hariri, "YawDD: '
            'Yawning Detection Dataset", IEEE Dataport, August 1, 2020, doi:10.21227/e1qm-hb90\n'
            "\n"
            "S. Abtahi, M. Omidyeganeh, S. Shirmohammadi, and B. Hariri, “YawDD: A Yawning "
            "Detection Dataset”, Proc. ACM Multimedia Systems, Singapore, March 19 -21 2014, "
            "pp. 24-28. DOI: 10.1145/2557642.2563678"
        ),
    ),
    DatasetSpec(
        name="dad",
        title="DAD - Driver Anomaly Detection (TUM)",
        url="https://www.ce.cit.tum.de/mmk/dad/",
        licence="Free for research purposes (DAD_License.pdf)",
        commercial_ok=False,
        access="direct download",
        size_gb=560.0,
        modalities=("nir", "depth"),
        provides=("activity",),
        group="driver_video",
        notes="Use a ~100 GB subset of front/top IR. Normal vs. anomalous driving.",
    ),
    DatasetSpec(
        name="syndd",
        title="SynDD1 / SynDD2 (AI City naturalistic driving)",
        url="https://data.mendeley.com/datasets/ptcp7rp3wb/4",
        licence="By request",
        commercial_ok=False,
        access="request on Mendeley Data",
        size_gb=60.0,
        modalities=("rgb",),
        provides=("activity",),
        group="driver_video",
        notes="Real recordings despite the name. 16 distracted-driving activity classes, 3 cameras.",
    ),
    # --- Driver distraction images -------------------------------------------------------
    DatasetSpec(
        name="state_farm",
        title="State Farm Distracted Driver Detection (Kaggle)",
        url="https://www.kaggle.com/c/state-farm-distracted-driver-detection",
        licence="Kaggle competition rules (not yet reviewed)",
        commercial_ok=False,
        access=(
            "kaggle competitions download -c state-farm-distracted-driver-detection "
            "(accept rules first)"
        ),
        size_gb=4.0,
        modalities=("rgb",),
        provides=(
            "activity",
            "attention_direction",
            "context",
            "drink",
            "face_visible",
            "modality",
            "phone",
            "phone_at_ear",
        ),
        group="driver_images",
        adapter=True,
        notes="22k labelled images, 26 drivers, 10 classes, subject ids in driver_imgs_list.csv.",
        terms=(
            "Kaggle competition rules. Whether they allow use outside the competition, or the "
            "release of models trained on the data, is not yet reviewed (the rules page could "
            "not be read without signing in to Kaggle, 2026-10-29)."
        ),
    ),
    DatasetSpec(
        name="auc_v2",
        title="AUC Distracted Driver Dataset v2",
        url="https://heshameraqi.github.io/distraction_detection",
        licence=(
            "By request; non-commercial; to develop distracted-driver detection (other research "
            "needs the MI-AUC group's consent)"
        ),
        commercial_ok=False,
        access="signed licence agreement emailed to the authors",
        size_gb=2.3,
        modalities=("rgb",),
        provides=("activity", "context", "drink", "modality", "phone", "phone_at_ear"),
        group="driver_images",
        adapter=True,
        notes=(
            "14,478 images of 44 drivers in State Farm's 10 classes from two cameras: camera 1 "
            "10,555 train / 1,123 test, camera 2 2,000 train / 800 test (counted 2026-10-29). "
            "The authors split train and test by driver, but one camera-1 driver is on both "
            "sides: 83 of her c0 images are in test and her other classes in train, so Bulwark "
            "moves those 83 to train (camera 1 here: 10,638 train / 1,040 test). No other "
            "driver was found on both sides, but the check compared thumbnails within each "
            "camera: drivers recorded again in other clothes (the dataset page says some have "
            "several sessions) were not looked for, and the two cameras' drivers were not "
            "matched. No per-image "
            "driver id is published, so each side is one subject group: no val split, test "
            "results rest on one group (indicative, no subject-level interval), and one driver "
            "cannot be removed on their own. The archive's v1 images (camera 1, split at random) "
            "are not used: every v2 camera-1 image is a byte-identical copy of one."
        ),
        terms=(
            "License Agreement V2, signed and returned; access granted by the authors on "
            "2026-10-29. \"Any commercial use of the dataset is strictly prohibited\", and "
            "commercial use includes testing commercial systems. The dataset may not be "
            "transferred, \"permanently or temporarily\", distributed or broadcast to third "
            "parties without prior authorization of the MI-AUC group. Use to develop "
            "distracted-driver detection is permitted; other research uses need the group's "
            "prior consent. Trained models are not mentioned. Releasing the v0 weights trained "
            "on AUC Distracted Driver v2 for non-commercial research is Inferensys's decision "
            "of 2026-10-30, not a statement by the MI-AUC group. "
            "Storage: a private copy on Kaggle (a dataset visible only to the maintainers' "
            "account, used only to train on Kaggle's GPUs) approved by the maintainers on "
            "2026-10-29; the group has not authorized the copy. Azure Blob not approved: no "
            "copy is kept there."
        ),
        weights_release_ok=True,
        weights_release_basis=MAINTAINERS_RELEASE_DECISION,
        kaggle_storage_ok=True,
        azure_storage_ok=False,
        # The dataset page publishes no BibTeX, and the paper's page (Wiley) could not be
        # read (HTTP 403, 2026-10-29). This is the paper the V2 licence agreement asks users
        # to cite, verbatim but for its typographic quotation marks; the dataset page lists
        # it in the same words.
        citation=(
            'H. Eraqi, Y. Abouelnaga, M. Saad, M. Moustafa, "Driver Distraction Identification '
            'with an Ensemble of Convolutional Neural Networks", Journal of Advanced '
            "Transportation, Machine Learning in Transportation (MLT) Issue, 2019."
        ),
    ),
    DatasetSpec(
        name="100_driver",
        title="100-Driver",
        url="https://100-driver.github.io/",
        licence="Signed licensing form",
        commercial_ok=False,
        access="licensing form to the authors",
        size_gb=50.0,
        modalities=("rgb", "nir"),
        provides=("activity", "phone", "drink", "food"),
        group="driver_images",
        notes="470k images, 100 drivers, 4 cameras, day and night, 22 classes.",
        terms="Signed agreement; its clauses on derived models and company use are not yet reviewed.",
    ),
    DatasetSpec(
        name="zenodo_driver",
        title="Multi-Class Driver Behavior Image Dataset (Zenodo 14908802)",
        url="https://zenodo.org/records/14908802",
        licence="Labelled CC BY 4.0, but appears to include re-encoded State Farm images",
        commercial_ok=False,
        access="direct download",
        size_gb=2.6,
        modalities=("rgb",),
        provides=("activity", "context", "modality", "phone", "phone_at_ear"),
        group="driver_images",
        download_url=(
            "https://zenodo.org/records/14908802/files/"
            "Multi-Class%20Driver%20Behavior%20Image%20Dataset.zip?download=1"
        ),
        adapter=True,
        notes=(
            "7,276 images, 5 classes. Excluded from v0 by the maintainers on 2026-10-29 "
            "(SplitPolicy.excluded_datasets): the data use agreement below forbids sharing with "
            "third parties and asks to review publications 30 days ahead, and the project holds "
            "no signed copy. No driver ids, so it would otherwise be train-only; part looks like "
            "upscaled State Farm (checked 2026-10-28). 113 images carry an EXIF rotation "
            "(checked 2026-10-29)."
        ),
        terms=(
            "The Zenodo record is labelled CC BY 4.0 (checked 2026-10-29), but it ships a "
            "'DATA USE AGREEMENT.docx' (30,218 bytes, sha256 1fdd9450...a156, read 2026-10-29) "
            "between Daffodil International University and a recipient. It restricts use to "
            "'academic research, public safety, and the development of technologies aimed at "
            "improving road safety and driver behavior detection'; says the recipient 'shall "
            "not redistribute, sell, or otherwise share the Dataset with third parties without "
            "explicit written permission from DIU' and shall not disclose it to any third party "
            "without prior written consent (s. 2(b), 4(b)); and asks for a copy of any proposed "
            "publication 30 days before submission (s. 7(b)). The State Farm images it appears "
            "to contain are under State Farm's own terms. Given these clauses, no copy is kept "
            "on Kaggle or Azure Blob."
        ),
        weights_release_ok=False,
        weights_release_basis=(
            "The maintainers' decision of 2026-10-29 to exclude the dataset from v0, on its data "
            "use agreement (see the terms): it forbids sharing the dataset with third parties, "
            "and the project holds no signed copy."
        ),
        kaggle_storage_ok=False,
        azure_storage_ok=False,
        citation=(
            "@dataset{arafat_sahin_2025_14908802,\n"
            "  author       = {Arafat Sahin, Afridi and\n"
            "                  Moon, Nazmun Nessa and\n"
            "                  Kafy, Arafath and\n"
            "                  Shakil, Shariear},\n"
            "  title        = {Multi-Class Driver Behavior Image Dataset},\n"
            "  month        = feb,\n"
            "  year         = 2025,\n"
            "  publisher    = {Zenodo},\n"
            "  doi          = {10.5281/zenodo.14908802},\n"
            "  url          = {https://doi.org/10.5281/zenodo.14908802},\n"
            "}"
        ),
    ),
    DatasetSpec(
        name="ndd_lowlight",
        title="Novel Driver Distractions (low-light NoIR)",
        url="https://data.mendeley.com/datasets/ykmr99nrsg/2",
        licence="CC BY-NC 3.0",
        commercial_ok=False,
        access="direct download (Mendeley Data)",
        size_gb=3.0,
        modalities=("nir",),
        provides=("activity",),
        group="driver_images",
        notes="70 drivers, NoIR camera with IR LEDs, 10 classes.",
    ),
    # --- Eyes, gaze, head pose -------------------------------------------------------------
    DatasetSpec(
        name="mrl_eye",
        title="MRL Eye Dataset",
        url="https://mrl.cs.vsb.cz/eyedataset.html",
        licence="No licence stated (all rights reserved; no written permission held)",
        commercial_ok=False,
        access="direct download",
        size_gb=0.4,
        modalities=("nir",),
        provides=("eye_openness", "eyeglasses", "eyes_closed", "low_light", "modality"),
        group="eyes_gaze",
        download_url="https://mrl.cs.vsb.cz/data/eyedataset/mrlEyes_2018_01.zip",
        adapter=True,
        notes="84,898 IR eye crops from 37 people. Labels are encoded in file names.",
        terms=(
            "The official page states no licence or terms of use; it says the dataset is "
            "publicly available and asks users to refer to the paper below for details. The "
            "live site was unreachable on 2026-10-28 and 2026-10-29, so this was read from the "
            "Internet Archive's copy of https://mrl.cs.vsb.cz/eyedataset.html (snapshot "
            "2026-05-16). The original authors' terms govern, whatever licence a mirror shows. "
            "Releasing the v0 weights trained on MRL Eye for non-commercial research is "
            "Inferensys's decision of 2026-10-30, not a statement by the dataset's authors. "
            "Storage: private copies on Kaggle (datasets visible only to the maintainers' "
            "account, used only to train on Kaggle's GPUs; not redistribution) approved by the "
            "maintainers on 2026-10-29; Azure Blob not yet approved."
        ),
        weights_release_ok=True,
        weights_release_basis=MAINTAINERS_RELEASE_DECISION,
        kaggle_storage_ok=True,
        # The page's [bibtex] link (publications/fusek_isvc2018.bib), as archived by the
        # Internet Archive on 2024-07-17; copied verbatim.
        citation=(
            "@ARTICLE{Fusek2018433,\n"
            "author={Fusek, R.},\n"
            "title={Pupil localization using geodesic distance},\n"
            "journal={Lecture Notes in Computer Science (including subseries Lecture Notes in "
            "Artificial Intelligence and Lecture Notes in Bioinformatics)},\n"
            "year={2018},\n"
            "volume={11241 LNCS},\n"
            "pages={433-444},\n"
            "doi={10.1007/978-3-030-03801-4_38},\n"
            "document_type={Conference Paper},\n"
            "}"
        ),
        source_copy=(
            "Kaggle mirror https://www.kaggle.com/datasets/akashshingha850/mrl-eye-dataset "
            "(original file names), downloaded 2026-10-28 because mrl.cs.vsb.cz was "
            "unreachable: 84,898 images, the official count. Image fingerprint "
            "(bulwark.data.sources.base.copy_fingerprint) "
            "sha256:2cc111083a9168cb63f364b89088efef5c7bad0bfe71e67d5c4a72e80a6b228a. On "
            "2026-10-29 it was compared with the official archive mrlEyes_2018_01.zip as "
            "captured by the Internet Archive on 2026-04-23 (341,866,898 bytes, "
            "sha256:17ff89928005cb6e63a53fe020e419060279ff59e57555cfa49693fc1865fdfc): the "
            "same 84,898 images with the same fingerprint, so the images are identical."
        ),
        source_copy_fingerprint="2cc111083a9168cb63f364b89088efef5c7bad0bfe71e67d5c4a72e80a6b228a",
    ),
    DatasetSpec(
        name="cew",
        title="CEW - Closed Eyes in the Wild",
        url="http://parnec.nuaa.edu.cn/_upload/tpl/02/db/731/template731/pages/xtan/ClosedEyeDatabases.html",
        licence="Research use by a researcher only; no commercial use or redistribution",
        commercial_ok=False,
        access="direct download",
        size_gb=0.1,
        modalities=("rgb",),
        provides=("eye_openness", "eyes_closed", "face_visible", "modality"),
        group="eyes_gaze",
        adapter=True,
        notes="2,423 face images (1,192 closed, 1,231 open) from LFW-style web photos.",
        terms=(
            '"The dataset is provided for research purposes to a researcher only and not for '
            'any commercial use." "Please do not release the data or redistribute this link to '
            'anyone else without our permission." Copyright 2014, Xiaoyang Tan (dataset page, '
            "checked 2026-10-29). Trained models are not mentioned: ask the authors before "
            "publishing weights."
        ),
        # Copied from the dataset page, which asks users to cite it this way.
        citation=(
            "F.Song, X.Tan, X.Liu and S.Chen, Eyes Closeness Detection from Still Images with "
            "Multi-scale Histograms of Principal Oriented Gradients, Pattern Recognition, 2014."
        ),
    ),
    DatasetSpec(
        name="eth_xgaze",
        title="ETH-XGaze",
        url="https://ait.ethz.ch/xgaze",
        licence="CC BY-NC-SA 4.0 with extra conditions",
        commercial_ok=False,
        access="request form",
        size_gb=130.0,
        modalities=("rgb",),
        provides=("gaze_yaw", "gaze_pitch", "head_yaw", "head_pitch"),
        group="eyes_gaze",
        notes="1.1M face patches with gaze and head pose. Use the 224 px release.",
    ),
    DatasetSpec(
        name="mpiifacegaze",
        title="MPIIFaceGaze",
        url="https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/its-written-all-over-your-face-full-face-appearance-based-gaze-estimation",
        licence="CC BY-NC-SA 4.0",
        commercial_ok=False,
        access="direct download",
        size_gb=1.0,
        modalities=("rgb",),
        provides=("gaze_yaw", "gaze_pitch"),
        group="eyes_gaze",
        notes="15 people, 45k laptop-webcam images.",
    ),
    DatasetSpec(
        name="gaze360",
        title="Gaze360",
        url="http://gaze360.csail.mit.edu/",
        licence="Research use",
        commercial_ok=False,
        access="request form",
        size_gb=25.0,
        modalities=("rgb",),
        provides=("gaze_yaw", "gaze_pitch"),
        group="eyes_gaze",
        notes="Unconstrained 360-degree gaze, 238 people, indoor and outdoor.",
    ),
    DatasetSpec(
        name="300w_lp",
        title="300W-LP and BIWI head pose",
        url="http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm",
        licence="Research use",
        commercial_ok=False,
        access="direct download",
        size_gb=3.0,
        modalities=("rgb",),
        provides=("head_yaw", "head_pitch", "head_roll"),
        group="eyes_gaze",
        notes="Large-pose synthetic face set with 3DMM head pose.",
    ),
    # --- Body, hands, objects, PPE ----------------------------------------------------------
    DatasetSpec(
        name="coco",
        title="COCO 2017 + COCO-WholeBody",
        url="https://cocodataset.org/ and https://github.com/jin-s13/COCO-WholeBody",
        licence="COCO annotations CC BY 4.0 (Flickr images mixed); WholeBody research only",
        commercial_ok=False,
        access="direct download",
        size_gb=25.0,
        modalities=("rgb",),
        provides=("phone", "phone_at_ear", "food", "drink", "face_visible"),
        group="body_objects",
        notes="Person crops with a phone, food or drink object nearby, plus whole-body keypoints.",
    ),
    DatasetSpec(
        name="open_images",
        title="Open Images V7 (person / phone / food subset)",
        url="https://storage.googleapis.com/openimages/web/index.html",
        licence="Annotations CC BY 4.0, images listed as CC BY 2.0",
        commercial_ok=True,
        access="FiftyOne zoo or the downloader script",
        size_gb=30.0,
        modalities=("rgb",),
        provides=("phone", "food", "drink", "cigarette", "sunglasses", "helmet"),
        group="body_objects",
        notes="Download only images with Person plus a relevant object class.",
    ),
    DatasetSpec(
        name="sh17",
        title="SH17 - PPE detection",
        url="https://github.com/ahmadmughees/SH17dataset",
        licence="CC BY-NC-SA 4.0",
        commercial_ok=False,
        access="Kaggle or the Pexels download script in the repo",
        size_gb=13.0,
        modalities=("rgb",),
        provides=(
            "context",
            "eyeglasses",
            "face_mask",
            "gloves",
            "helmet",
            "modality",
            "safety_vest",
        ),
        group="body_objects",
        adapter=True,
        notes=(
            "8,099 images, 17 classes, YOLO-format labels. Pexels photo series show the same "
            "people across images and no ids are published, so train-only."
        ),
        terms=(
            '"The SH17 dataset is released under CC BY-NC-SA 4.0 license." The README adds: '
            '"This dataset, scrapped through the Pexels website, is intended for educational, '
            "research, and analysis purposes only. You may be able to use the data for training "
            'of the Machine learning models only." Users must also follow the Pexels licence '
            "(github.com/ahmadmughees/SH17dataset README, checked 2026-10-29)."
        ),
        # The README's "Citation" section, verbatim.
        citation=(
            "@article{AHMAD2024,\n"
            "    title = {SH17: A dataset for human safety and personal protective equipment "
            "detection in manufacturing industry},\n"
            "    journal = {Journal of Safety Science and Resilience},\n"
            "    year = {2024},\n"
            "    issn = {2666-4496},\n"
            "    doi = {https://doi.org/10.1016/j.jnlssr.2024.09.002},\n"
            "    url = {https://www.sciencedirect.com/science/article/pii/S266644962400077X},\n"
            "    author = {Hafiz Mughees Ahmad and Afshin Rahimi},\n"
            "}"
        ),
    ),
    DatasetSpec(
        name="roboflow_forklift_ppe",
        title="Roboflow Universe forklift / PPE projects (CC BY 4.0 only)",
        url="https://universe.roboflow.com/",
        licence="CC BY 4.0 per project (unverified; check each project and its image sources)",
        commercial_ok=False,
        access="Roboflow API key",
        size_gb=5.0,
        modalities=("rgb",),
        provides=("helmet", "safety_vest", "context"),
        group="body_objects",
        notes=(
            "Operator and warehouse context. Projects often re-host other datasets, so the "
            "licence has to be recorded per project (per clip) from a reviewed allowlist."
        ),
    ),
    # --- General human video ----------------------------------------------------------------
    DatasetSpec(
        name="stock_video",
        title="Pexels / Pixabay stock video",
        url="https://www.pexels.com/api/ and https://pixabay.com/api/docs/",
        licence=(
            "Pexels / Pixabay content licence (use for ML training and for identifiable people "
            "not yet verified)"
        ),
        commercial_ok=False,
        access="free API key",
        size_gb=60.0,
        modalities=("rgb",),
        provides=("context",),
        group="general_video",
        notes=(
            "Queries: driving, night driving, truck driver, tired, yawning, texting, forklift, "
            "factory worker."
        ),
    ),
    DatasetSpec(
        name="kinetics700_subset",
        title="Kinetics-700 subset",
        url="https://github.com/cvdfoundation/kinetics-dataset",
        licence="Annotations CC BY 4.0; videos remain YouTube copyright",
        commercial_ok=False,
        access="CVDF mirror tarballs",
        size_gb=40.0,
        modalities=("rgb",),
        provides=("activity", "yawn"),
        group="general_video",
        notes=(
            "Classes: yawning, texting, drinking, smoking, eating, driving car, talking on "
            "phone, applying makeup."
        ),
    ),
    DatasetSpec(
        name="charades",
        title="Charades (subset)",
        url="https://prior.allenai.org/projects/charades",
        licence="Non-commercial",
        commercial_ok=False,
        access="direct download",
        size_gb=15.0,
        modalities=("rgb",),
        provides=("activity",),
        group="general_video",
        notes="Indoor everyday actions, including eating, drinking, phone use, sitting at a desk.",
    ),
    DatasetSpec(
        name="synthetic_wan",
        title="Bulwark synthetic clips (Wan 2.2 TI2V-5B)",
        url="https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B",
        licence=(
            "Apache-2.0 model; the model card claims no rights over outputs, subject to its "
            "use restrictions"
        ),
        commercial_ok=True,
        access="generated by this project (scripts/synth_wan.py)",
        size_gb=10.0,
        modalities=("rgb", "nir"),
        provides=(),
        group="synthetic",
        adapter=True,
        notes=(
            "Prompted gap-fill: microsleep at night, IR cabin look, forklift and crane cabs. "
            "Prompt labels are weak labels at teacher weight, never ground truth; train-only."
        ),
    ),
)

BY_NAME = {spec.name: spec for spec in DATASETS}


def get_spec(name: str) -> DatasetSpec:
    try:
        return BY_NAME[name]
    except KeyError as exc:
        raise KeyError(f"Unknown dataset {name!r}. Known: {sorted(BY_NAME)}") from exc


def uncleared_for_release(names: Iterable[str]) -> list[str]:
    """Datasets among `names` not cleared for publishing trained weights (their
    `weights_release_ok` is not True)."""
    return sorted({name for name in names if get_spec(name).weights_release_ok is not True})


GROUP_TITLES = {
    "driver_video": "Driver state video",
    "driver_images": "Driver distraction images",
    "eyes_gaze": "Eyes, gaze and head pose",
    "body_objects": "Body, hands, objects and PPE",
    "general_video": "General human video",
    "synthetic": "Synthetic",
}


def _flag(value: bool | None) -> str:
    return "unknown" if value is None else ("yes" if value else "no")


def _release_flag(spec: DatasetSpec) -> str:
    """The Release column: a decision the maintainers made themselves says so."""
    flag = _flag(spec.weights_release_ok)
    return f"{flag} (maintainers' decision)" if spec.weights_release_basis == MAINTAINERS_RELEASE_DECISION else flag


def datasets_markdown() -> str:
    lines = [
        "# Bulwark datasets",
        "",
        "Generated by `bulwark datasets --markdown`. Do not edit by hand; edit "
        "`bulwark/data/registry.py`.",
        "",
        "`Commercial`: the maintainers' reading of the dataset's published terms is that "
        "commercial use is permitted. This is not legal advice; check the licence yourself "
        "before relying on it. "
        "Every clip in the manifest carries this flag. The commercially clean retrain is the "
        "`configs/train/*-clean.yaml` chain: every stage trains only on clips with this flag "
        "(`commercial_only: true`) and refuses checkpoints or teacher features made without it "
        "(see [training.md](training.md)). It covers Bulwark's data, not the encoder's upstream "
        "pretraining.",
        "",
        "`Release`: whether weights trained on the dataset may be published: `yes`, `no` or "
        "`unknown` (not yet determined). A `yes` or `no` rests on the dataset's terms, its "
        "owners' written permission or the maintainers' own dated decision, and Terms relied on "
        "says which (\"Weights release\"). `yes (maintainers' decision)` is Inferensys's decision, "
        "made on the dataset's research terms, that v0 weights trained on it may be published "
        "for non-commercial research; "
        "it is not a statement by the dataset's owners, and users must also follow the "
        "dataset's terms. A checkpoint is ready for release only when every dataset it was "
        "trained on is `yes`, and `bulwark release-check` prints the basis with its verdict.",
        "",
        "`Kaggle` and `Azure`: whether a private copy may be kept on that third party's "
        "storage: a private Kaggle dataset, or the private Azure Blob container in the "
        "project's own subscription. Each provider is decided on its own. `yes` or `no` (the "
        "terms or signed agreement have been read and the maintainers have decided; Terms "
        "relied on quotes the clause or records the dated decision) or `unknown` (not decided "
        "yet). `infra/kaggle/push_data.sh` (Kaggle) and `infra/azure/upload.sh` (Azure Blob) "
        "refuse the shards, feature files and manifest rows of datasets that are not `yes` for "
        "their provider unless the operator names them (`ALLOW_UNCLEARED`), and "
        "`scripts/purge_cloud_copies.sh` deletes the shard blobs of a dataset that is not `yes` "
        "for Azure instead of replacing them.",
        "",
        "`Labels`: the taxonomy labels (`bulwark/data/taxonomy.yaml`) the adapter writes as "
        "ground truth.",
        "",
    ]
    header = (
        "| Dataset | Licence | Commercial | Release | Kaggle | Azure | Access | Size (GB) | "
        "Adapter | Labels | Notes |"
    )
    for group, title in GROUP_TITLES.items():
        lines += [f"## {title}", "", header, "|---|---|---|---|---|---|---|---:|---|---|---|"]
        for spec in DATASETS:
            if spec.group != group:
                continue
            lines.append(
                f"| [{spec.title}]({spec.url.split(' ')[0]}) | {spec.licence} | "
                f"{'yes' if spec.commercial_ok else 'no'} | {_release_flag(spec)} | "
                f"{_flag(spec.kaggle_storage_ok)} | {_flag(spec.azure_storage_ok)} | "
                f"{spec.access} | {spec.size_gb:g} | {'yes' if spec.adapter else 'not yet'} | "
                f"{', '.join(spec.provides) or '-'} | {spec.notes} |"
            )
        lines.append("")
    total = sum(spec.size_gb for spec in DATASETS)
    lines += [
        f"Total if everything is downloaded: about {total:,.0f} GB (DAD alone is 560 GB; take a "
        "subset).",
        "",
        "## Terms relied on",
        "",
    ]
    for spec in DATASETS:
        if spec.terms or spec.source_copy or spec.weights_release_basis:
            lines.append(f"- **{spec.title}.** {spec.terms or 'Not yet reviewed.'}")
            if spec.weights_release_basis:
                lines.append(f"  Weights release ({_flag(spec.weights_release_ok)}): {spec.weights_release_basis}")
            if spec.source_copy:
                lines.append(f"  Copy used: {spec.source_copy}")
    lines += [
        "",
        "## Citations",
        "",
        "Copied from each dataset's own page or paper. Cite these when you use the data.",
        "",
    ]
    for spec in DATASETS:
        if spec.citation:
            fence = "```bibtex" if spec.citation.startswith("@") else "```text"
            lines += [f"### {spec.title}", "", fence, spec.citation, "```", ""]
    return "\n".join(lines).rstrip("\n") + "\n"
