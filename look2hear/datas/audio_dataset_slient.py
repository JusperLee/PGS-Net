import torch
from torch.utils import data
import json
import os
import numpy as np
import soundfile as sf


def normalize_tensor_wav(wav_tensor, eps=1e-8, std=None):
    mean = wav_tensor.mean(-1, keepdim=True)
    if std is None:
        std = wav_tensor.std(-1, keepdim=True)
    return (wav_tensor - mean) / (std + eps)


class AudioSlientDataset(data.Dataset):
    """Dataset class for the wsj0-mix source separation dataset.

    Args:
        json_dir (str): The path to the directory containing the json files.
        sample_rate (int, optional): The sampling rate of the wav files.
        segment (float, optional): Length of the segments used for training,
            in seconds. If None, use full utterances (e.g. for test).
        n_src (int, optional): Number of sources in the training targets.

    References
        "Deep clustering: Discriminative embeddings for segmentation and
        separation", Hershey et al. 2015.
    """

    dataset_name = "wsj0-mix"

    def __init__(
        self,
        json_dir,
        n_src=2,
        sample_rate=8000,
        segment=4.0,
        normalize_audio=False,
        gauss=False,
        slient=2,
    ):
        super().__init__()
        # Task setting
        self.json_dir = json_dir
        self.sample_rate = sample_rate
        self.normalize_audio = False
        self.EPS = 1e-8
        self.gauss = gauss
        self.slient = slient
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
        sources = torch.from_numpy(np.vstack(source_arrays))
        mixture = torch.from_numpy(x)

        # noise
        # if self.gauss:
        #     noise = np.random.randn(int(self.sample_rate * self.slient))
        #     noise = noise / np.sqrt(np.sum(noise**2)) * np.sqrt(
        #         np.sum(mixture.numpy()**2))
        #     noise_energy = np.random.uniform(-40. - 20)
        #     noise = torch.from_numpy(noise * np.power(10, noise_energy / 20.))
        #     mixture = torch.cat((noise, mixture), dim=0).float()
        #     new_sources = []
        #     for i in range(sources.shape[0]):
        #         new_sources.append(torch.cat((noise, sources[i]), dim=0).float())
        #     new_sources = torch.from_numpy(np.vstack(new_sources))
        # else:
        #     noise = torch.zeros(int(self.sample_rate * self.slient))
        #     mixture = torch.cat((noise, mixture), dim=0)
        #     new_sources = []
        #     for i in range(sources.shape[0]):
        #         new_sources.append(torch.cat((noise, sources[i]), dim=0))
        #     new_sources = torch.from_numpy(np.vstack(new_sources))

        # noise = torch.zeros(int(self.sample_rate * self.slient))
        # mixture = torch.cat((mixture, noise), dim=0)
        # new_sources = []
        # for i in range(sources.shape[0]):
        #     new_sources.append(torch.cat((sources[i], noise), dim=0))
        # new_sources = torch.from_numpy(np.vstack(new_sources))

        noise = torch.zeros(int(self.sample_rate * self.slient))
        mixture = torch.cat((noise, mixture), dim=0)
        new_sources = []
        for i in range(sources.shape[0]):
            new_sources.append(torch.cat((noise, sources[i]), dim=0))
        new_sources = torch.from_numpy(np.vstack(new_sources))
        return mixture, new_sources, self.mix[idx][0].split("/")[-1]
