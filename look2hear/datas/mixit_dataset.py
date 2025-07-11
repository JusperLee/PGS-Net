import torch
from torch.utils import data
import json
import os
import numpy as np
import soundfile as sf
from random import sample


def normalize_tensor_wav(wav_tensor, eps=1e-8, std=None):
    mean = wav_tensor.mean(-1, keepdim=True)
    if std is None:
        std = wav_tensor.std(-1, keepdim=True)
    return (wav_tensor - mean) / (std + eps)


class MixITDataset(data.Dataset):
    dataset_name = "MixIT"

    def __init__(
        self, json_dir, n_src=4, sample_rate=8000, segment=4.0, normalize_audio=False
    ):
        super().__init__()
        # Task setting
        self.json_dir = json_dir
        self.sample_rate = sample_rate
        self.normalize_audio = False
        self.EPS = 1e-8
        if segment is None:
            self.seg_len = None
        else:
            self.seg_len = int(segment * sample_rate)
        self.n_src = n_src
        self.like_test = self.seg_len is None
        # Load json files
        mix_json = os.path.join(json_dir, "mix.json")
        sources_json = [
            os.path.join(json_dir, source + ".json")
            for source in [f"s{n+1}" for n in range(n_src)]
        ]
        with open(mix_json, "r") as f:
            mix_infos = json.load(f)
        sources_infos = []
        for src_json in sources_json:
            with open(src_json, "r") as f:
                sources_infos.append(json.load(f))
        # Filter out short utterances only when segment is specified
        orig_len = len(mix_infos)
        drop_utt, drop_len = 0, 0
        if not self.like_test:
            for i in range(len(mix_infos) - 1, -1, -1):  # Go backward
                if mix_infos[i][1] < self.seg_len:
                    drop_utt = drop_utt + 1
                    drop_len = drop_len + mix_infos[i][1]
                    del mix_infos[i]
                    for src_inf in sources_infos:
                        del src_inf[i]

        print(
            "Drop {} utts({:.2f} h) from {} (shorter than {} samples)".format(
                drop_utt, drop_len / sample_rate / 36000, orig_len, self.seg_len
            )
        )
        self.mix = mix_infos
        self.sources = sources_infos

    def __len__(self):
        return len(self.mix)

    def __getitem__(self, idx):
        """Gets a mixture/sources pair.
        Returns:
            mixture, vstack([source_arrays])
        """
        # Random start
        if self.mix[idx][1] == self.seg_len or self.like_test:
            rand_start = 0
        else:
            rand_start = np.random.randint(0, self.mix[idx][1] - self.seg_len)
        if self.like_test:
            stop = None
        else:
            stop = rand_start + self.seg_len
        # Load mixture
        x, _ = sf.read(self.mix[idx][0], start=rand_start, stop=stop, dtype="float32")
        seg_len = torch.as_tensor([len(x)])
        # Load sources
        source_arrays = []
        for src in self.sources:
            if src[idx] is None:
                # Target is filled with zeros if n_src > default_nsrc
                s = np.zeros((seg_len,))
            else:
                s, _ = sf.read(
                    src[idx][0], start=rand_start, stop=stop, dtype="float32"
                )
            source_arrays.append(s)

        index = set(range(self.n_src))
        mix1_index = sample(list(index), self.n_src // 2)
        mix1 = [source_arrays[i] for i in mix1_index]
        mix2 = [source_arrays[i] for i in set(index).difference(set(mix1_index))]
        import pdb

        pdb.set_trace()
        moms = torch.from_numpy(np.vstack(mix1, mix2))
        sources = torch.from_numpy(np.vstack(source_arrays))
        mixture = torch.from_numpy(x)

        if self.normalize_audio:
            m_std = mixture.std(-1, keepdim=True)
            mixture = normalize_tensor_wav(mixture, eps=self.EPS, std=m_std)
            sources = normalize_tensor_wav(sources, eps=self.EPS, std=m_std)
            moms = normalize_tensor_wav(moms, eps=self.EPS, std=m_std)

        return mixture, sources, moms, self.mix[idx][0].split("/")[-1]
