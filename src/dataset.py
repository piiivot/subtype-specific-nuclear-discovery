"""
Dataset classes for MIME training.

- LMDBReader: reads patch data from an LMDB database
- EightClassDatasetKSplit: k-split cross-validation dataset (single-label)
- EightClassDatasetKSplitMIME: same as above, but uses multi-label pseudo-labels
"""

import csv
import pickle
import lmdb
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from PIL import Image
from torch.utils.data import Dataset


# ─────────────────────────────────────────
# LMDB Reader
# ─────────────────────────────────────────

class LMDBReader:
    """
    Reads patch data from an LMDB database.

    Supports both pickle and raw numpy formats.
    Designed for use with multiple DataLoader workers (lazy initialization).
    """

    def __init__(self, lmdb_path: str, use_pickle: bool = True, patch_size: int = 224):
        """
        Args:
            lmdb_path: path to the directory containing data.mdb
            use_pickle: if True, values are deserialized with pickle;
                        otherwise raw uint8 numpy arrays are expected
            patch_size: spatial size of each patch (used only when use_pickle=False)
        """
        self.lmdb_path = lmdb_path
        self.use_pickle = use_pickle
        self.patch_size = patch_size
        self.env = None
        self._length = None

    def _ensure_env(self):
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_path,
                max_readers=126,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )

    def __len__(self) -> int:
        if self._length is None:
            self._ensure_env()
            with self.env.begin() as txn:
                self._length = txn.stat()["entries"]
        return self._length

    def get_all_keys(self) -> List[str]:
        self._ensure_env()
        keys = []
        with self.env.begin() as txn:
            for key, _ in txn.cursor():
                keys.append(key.decode("utf-8"))
        return keys

    def get_patch(self, key: str) -> Optional[Dict]:
        """
        Returns a dict with at least {'image': PIL.Image or np.ndarray}.
        Returns None if the key does not exist.
        """
        self._ensure_env()
        with self.env.begin() as txn:
            value = txn.get(key.encode("utf-8"))
        if value is None:
            return None

        if self.use_pickle:
            data = pickle.loads(value)
            if "image" in data and isinstance(data["image"], np.ndarray):
                data["image"] = np.array(data["image"], copy=True)
            return data
        else:
            img_array = np.frombuffer(value, dtype=np.uint8).copy()
            img_array = img_array.reshape(self.patch_size, self.patch_size, 3)
            parts = key.rsplit("_", 2)
            return {
                "image": img_array,
                "x": int(parts[1]) if len(parts) == 3 else 0,
                "y": int(parts[2]) if len(parts) == 3 else 0,
                "wsi_filename": parts[0] if len(parts) == 3 else key,
            }

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None

    # Support pickling for multi-process DataLoader
    def __getstate__(self):
        state = self.__dict__.copy()
        state["env"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


# ─────────────────────────────────────────
# K-Split Dataset (single-label)
# ─────────────────────────────────────────

class EightClassDatasetKSplit(Dataset):
    """
    Eight-class lymphoma classification dataset with k-split cross-validation.

    Reads patches from an LMDB database and assigns class labels based on a
    fold CSV file. The fold CSV must contain the columns:
        fold        integer fold index (0 to num_folds-1)
        case_id     slide/case identifier (prefix of each LMDB key)
        class_name  one of the eight class names listed in CLASS_NAMES

    The folds to include are determined by ``use_folds`` or automatically from
    ``except_fold_idx``, ``cal_fold_idx``, and ``mode``:
        mode="train"        all folds except except_fold_idx and cal_fold_idx
        mode="val"          [except_fold_idx]
        mode="calibration"  [cal_fold_idx]

    Each __getitem__ returns (image_tensor, int_label).
    """

    CLASS_NAMES = [
        "FL-G1",
        "FL-G2",
        "FL-G3a",
        "FL-G3b",
        "DLBCL-CD5+",
        "DLBCL-GC",
        "DLBCL-nonGC",
        "Reactive",
    ]

    CLASS_TO_LABEL = {name: i for i, name in enumerate(CLASS_NAMES)}

    def __init__(
        self,
        data_path: str,
        fold_csv_path: str,
        use_folds: Optional[List[int]] = None,
        except_fold_idx: Optional[int] = None,
        cal_fold_idx: Optional[int] = None,
        num_folds: int = 10,
        mode: Optional[str] = None,
        transform=None,
        use_pickle: bool = True,
        patch_size: int = 64,
        use_cache: bool = True,
        **kwargs,
    ):
        if not data_path:
            raise ValueError("data_path must be specified.")
        if not fold_csv_path:
            raise ValueError("fold_csv_path must be specified.")

        # Determine which folds to use
        if use_folds is None:
            if except_fold_idx is None or cal_fold_idx is None:
                raise ValueError(
                    "Specify either use_folds or both except_fold_idx and cal_fold_idx."
                )
            all_folds = set(range(num_folds))
            if mode == "val":
                use_folds = [except_fold_idx]
            elif mode == "calibration":
                use_folds = [cal_fold_idx]
            else:  # "train" or None
                use_folds = sorted(all_folds - {except_fold_idx, cal_fold_idx})
            print(f"[EightClassDatasetKSplit] mode={mode}, use_folds={use_folds}")

        if not use_folds:
            raise ValueError("use_folds is empty.")

        self.transform = transform
        self.use_folds = sorted(use_folds)
        self.data_path = data_path
        self.use_pickle = use_pickle
        self.patch_size = patch_size
        self._reader = None  # lazy initialization per worker

        allowed_case_ids = self._load_case_ids(fold_csv_path, self.use_folds)
        folds_str = ",".join(map(str, self.use_folds))
        print(f"[EightClassDatasetKSplit] folds [{folds_str}]: {len(allowed_case_ids)} cases")

        # Try loading from cache
        folds_suffix = "_".join(map(str, self.use_folds))
        cache_path = Path(data_path) / f".ksplit_{folds_suffix}_cache.npz"

        if use_cache and cache_path.exists():
            try:
                cache = np.load(cache_path, allow_pickle=True)
                self.keys = cache["keys"].tolist()
                self.labels = cache["labels"].tolist()
                print(f"[EightClassDatasetKSplit] Loaded {len(self.keys)} patches from cache.")
                return
            except Exception as e:
                print(f"[EightClassDatasetKSplit] Cache load failed: {e}. Rebuilding...")

        # Build index from LMDB
        reader = LMDBReader(data_path, use_pickle=use_pickle, patch_size=patch_size)
        all_keys = reader.get_all_keys()
        reader.close()

        case_id_to_class = self._load_case_id_to_class(fold_csv_path, self.use_folds)
        self.keys = []
        self.labels = []
        for key in all_keys:
            case_id = self._key_to_case_id(key)
            if case_id in allowed_case_ids:
                label = self.CLASS_TO_LABEL[case_id_to_class[case_id]]
                self.keys.append(key)
                self.labels.append(label)

        print(f"[EightClassDatasetKSplit] folds [{folds_str}]: {len(self.keys)} patches")

        if use_cache:
            try:
                np.savez_compressed(
                    cache_path,
                    keys=np.array(self.keys, dtype=object),
                    labels=np.array(self.labels, dtype=np.int32),
                )
                print(f"[EightClassDatasetKSplit] Cache saved to {cache_path}")
            except Exception as e:
                print(f"[EightClassDatasetKSplit] Cache save failed: {e}")

    # ------------------------------------------------------------------
    # CSV helpers
    # ------------------------------------------------------------------

    def _load_case_ids(self, csv_path: str, use_folds: List[int]) -> Set[str]:
        fold_set = set(use_folds)
        case_ids: Set[str] = set()
        with open(csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if int(row["fold"]) in fold_set:
                    case_ids.add(row["case_id"])
        return case_ids

    def _load_case_id_to_class(self, csv_path: str, use_folds: List[int]) -> Dict[str, str]:
        fold_set = set(use_folds)
        mapping: Dict[str, str] = {}
        with open(csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if int(row["fold"]) in fold_set:
                    mapping[row["case_id"]] = row["class_name"]
        return mapping

    def _key_to_case_id(self, key: str) -> str:
        """
        Extract case ID from an LMDB key.
        Key format: "{case_id}.svs_{x}_{y}"  →  case_id = "{case_id}"
        """
        base = key.split("_")[0] if "_" in key else key
        return base.replace(".svs", "").replace(".ndpi", "")

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        if self._reader is None:
            self._reader = LMDBReader(self.data_path, use_pickle=self.use_pickle, patch_size=self.patch_size)

        patch = self._reader.get_patch(self.keys[index])
        if patch is None:
            raise ValueError(f"Key not found in LMDB: {self.keys[index]}")

        image = patch["image"]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, self.labels[index]


# ─────────────────────────────────────────
# MIME Dataset (multi-label pseudo-labels)
# ─────────────────────────────────────────

class EightClassDatasetKSplitMIME(EightClassDatasetKSplit):
    """
    MIME variant of EightClassDatasetKSplit.

    Instead of returning integer class labels, returns multi-label float
    vectors (pseudo-labels) of shape [num_classes].

    At initialization all pseudo-labels are set from the observed single-class
    labels (one-hot). During training the trainer calls ``update_pseudo_label``
    to add extra positive labels discovered by the model.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fold_csv_path = kwargs.get("fold_csv_path", "")

        # Initialize pseudo-labels as one-hot from observed labels
        num_samples = len(self.keys)
        num_classes = len(self.CLASS_NAMES)
        self.pseudo_labels = np.zeros((num_samples, num_classes), dtype=np.float32)
        for i, label_idx in enumerate(self.labels):
            self.pseudo_labels[i, label_idx] = 1.0

        folds_str = ",".join(map(str, self.use_folds))
        print(
            f"[EightClassDatasetKSplitMIME] Initialized {num_samples} samples "
            f"with one-hot pseudo-labels (folds: [{folds_str}])"
        )

    def __getitem__(self, index: int) -> Tuple[Any, np.ndarray]:
        if self._reader is None:
            self._reader = LMDBReader(self.data_path, use_pickle=self.use_pickle, patch_size=self.patch_size)

        patch = self._reader.get_patch(self.keys[index])
        if patch is None:
            raise ValueError(f"Key not found in LMDB: {self.keys[index]}")

        image = patch["image"]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, self.pseudo_labels[index]

    def update_pseudo_label(self, idx: int, class_idx: int, value: float):
        """Set pseudo_labels[idx, class_idx] = value."""
        self.pseudo_labels[idx, class_idx] = value

    def get_pseudo_label_stats(self) -> Dict[str, float]:
        """Return mean and std of the number of positive labels per sample."""
        counts = self.pseudo_labels.sum(axis=1)
        return {
            "avg_num_labels": float(counts.mean()),
            "std_num_labels": float(counts.std()),
        }
