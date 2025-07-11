import os
import sys
import torch
import torch.nn.functional as F
import torchaudio
import speechbrain as sb
import numpy as np
from speechbrain.dataio.batch import PaddedBatch
import warnings

warnings.filterwarnings("ignore")


class SBAudioDatset:
    def __init__(self, tr_dir, cv_dir, tt_dir, batch_size, num_workers):
        self.train = None
        self.val = None
        self.test = None
        self.tr_dir = tr_dir
        self.cv_dir = cv_dir
        self.tt_dir = tt_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.create()

    def create(self,):
        """Creates data processing pipeline"""

        # 1. Define datasets
        train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(csv_path=self.tr_dir)
        valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(csv_path=self.cv_dir)
        test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(csv_path=self.tt_dir)
        datasets = [train_data, valid_data, test_data]

        # 2. Provide audio pipelines
        @sb.utils.data_pipeline.takes("mix_wav")
        @sb.utils.data_pipeline.provides("mix_sig")
        def audio_pipeline_mix(mix_wav):
            mix_sig = sb.dataio.dataio.read_audio(mix_wav)
            return mix_sig

        @sb.utils.data_pipeline.takes("s1_wav")
        @sb.utils.data_pipeline.provides("s1_sig")
        def audio_pipeline_s1(s1_wav):
            s1_sig = sb.dataio.dataio.read_audio(s1_wav)
            return s1_sig

        @sb.utils.data_pipeline.takes("s2_wav")
        @sb.utils.data_pipeline.provides("s2_sig")
        def audio_pipeline_s2(s2_wav):
            s2_sig = sb.dataio.dataio.read_audio(s2_wav)
            return s2_sig

        sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_mix)
        sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_s1)
        sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_s2)
        sb.dataio.dataset.set_output_keys(
            datasets, ["id", "mix_sig", "s1_sig", "s2_sig"]
        )

        # 3. Provide audio dataloader
        self.train = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=PaddedBatch,
            worker_init_fn=lambda x: np.random.seed(
                int.from_bytes(os.urandom(4), "little") + x
            ),
        )

        self.val = torch.utils.data.DataLoader(
            valid_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=PaddedBatch,
            worker_init_fn=lambda x: np.random.seed(
                int.from_bytes(os.urandom(4), "little") + x
            ),
        )

        self.test = torch.utils.data.DataLoader(
            test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=PaddedBatch,
            worker_init_fn=lambda x: np.random.seed(
                int.from_bytes(os.urandom(4), "little") + x
            ),
        )


# if __name__ == "__main__":
#     dataloaders = SBAudioDatset(
#         '/home/likai/data1/nichang-sepformer/local/WSJ02Mix/wsj_tr.csv',
#         '/home/likai/data1/nichang-sepformer/local/WSJ02Mix/wsj_cv.csv',
#         '/home/likai/data1/nichang-sepformer/local/WSJ02Mix/wsj_tt.csv',
#         10, 40)

#     trainloader = dataloaders.train
#     valdataloader = dataloaders.val
#     testloader = dataloaders.test
#     for batch in trainloader:
#         import pdb; pdb.set_trace()
