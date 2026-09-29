"""LeRobot training entry point with a fast index-only dataset lookup.

LeRobot 0.5.1 installs an image-to-tensor transform before building the
absolute-to-relative frame index. Hugging Face Datasets then iterates full
rows for ``dataset["index"]`` and decodes every camera image unnecessarily.
This wrapper keeps the regular transform for training samples while reading
the index column through an unformatted view during dataset setup.
"""

from __future__ import annotations

from lerobot.datasets.dataset_reader import DatasetReader


def _build_index_mapping_without_image_decode(self: DatasetReader) -> None:
    self._absolute_to_relative_idx = None
    if self.episodes is not None and self.hf_dataset is not None:
        index_column = self.hf_dataset.with_format(None)["index"]
        self._absolute_to_relative_idx = {
            int(absolute_idx): relative_idx
            for relative_idx, absolute_idx in enumerate(index_column)
        }


def main() -> None:
    # Patch only this subprocess. Do not edit site-packages or alter the
    # transformed dataset used by the training DataLoader.
    DatasetReader._build_index_mapping = _build_index_mapping_without_image_decode
    from lerobot.scripts.lerobot_train import main as lerobot_main

    lerobot_main()


if __name__ == "__main__":
    main()
