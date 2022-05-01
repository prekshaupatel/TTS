#!/usr/bin/env python3

# Referred from data_prep.py in jv_openslr35 in ESPnet
# https://github.com/espnet/espnet/blob/master/egs2/jv_openslr35/
# asr1/local/data_prep.py



#!/usr/bin/env python3

# Copyright 2021 Wen-Chin Huang and Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Evaluate log-F0 RMSE between generated and groundtruth audios based on World."""

"""Evaluate MCD between generated and groundtruth audios with SPTK-based mcep."""

import argparse
import fnmatch
import logging
import multiprocessing as mp
import os

from typing import Dict
from typing import List
from typing import Tuple

import librosa
import numpy as np
import pysptk
import soundfile as sf

from fastdtw import fastdtw
from scipy import spatial


def sptk_extract(
    x: np.ndarray,
    fs: int,
    n_fft: int = 512,
    n_shift: int = 256,
    mcep_dim: int = 25,
    mcep_alpha: float = 0.41,
    is_padding: bool = False,
) -> np.ndarray:
    """Extract SPTK-based mel-cepstrum.
    Args:
        x (ndarray): 1D waveform array.
        fs (int): Sampling rate
        n_fft (int): FFT length in point (default=512).
        n_shift (int): Shift length in point (default=256).
        mcep_dim (int): Dimension of mel-cepstrum (default=25).
        mcep_alpha (float): All pass filter coefficient (default=0.41).
        is_padding (bool): Whether to pad the end of signal (default=False).
    Returns:
        ndarray: Mel-cepstrum with the size (N, n_fft).
    """
    # perform padding
    if is_padding:
        n_pad = n_fft - (len(x) - n_fft) % n_shift
        x = np.pad(x, (0, n_pad), "reflect")

    # get number of frames
    n_frame = (len(x) - n_fft) // n_shift + 1

    # get window function
    win = pysptk.sptk.hamming(n_fft)

    # check mcep and alpha
    if mcep_dim is None or mcep_alpha is None:
        mcep_dim, mcep_alpha = _get_best_mcep_params(fs)

    # calculate spectrogram
    mcep = [
        pysptk.mcep(
            x[n_shift * i : n_shift * i + n_fft] * win,
            mcep_dim,
            mcep_alpha,
            eps=1e-6,
            etype=1,
        )
        for i in range(n_frame)
    ]

    return np.stack(mcep)


def _get_basename(path: str) -> str:
    return os.path.splitext(os.path.split(path)[-1])[0]


def _get_best_mcep_params(fs: int) -> Tuple[int, float]:
    if fs == 16000:
        return 23, 0.42
    elif fs == 22050:
        return 34, 0.45
    elif fs == 24000:
        return 34, 0.46
    elif fs == 44100:
        return 39, 0.53
    elif fs == 48000:
        return 39, 0.55
    else:
        raise ValueError(f"Not found the setting for {fs}.")


def calculate(
    file_list: List[str],
    gt_file_list: List[str],
    args: argparse.Namespace,
    mcd_dict: Dict,
):
    """Calculate MCD."""
    for i, gen_path in enumerate(file_list):
        gt_path = gt_file_list[i]
        gt_basename = _get_basename(gt_path)

        # load wav file as int16
        gen_x, gen_fs = sf.read(gen_path, dtype="int16")
        gt_x, gt_fs = sf.read(gt_path, dtype="int16")

        fs = gen_fs
        if gen_fs != gt_fs:
            gt_x = librosa.resample(gt_x.astype(np.float), gt_fs, gen_fs)

        # extract ground truth and converted features
        gen_mcep = sptk_extract(
            x=gen_x,
            fs=fs,
            n_fft=args.n_fft,
            n_shift=args.n_shift,
            mcep_dim=args.mcep_dim,
            mcep_alpha=args.mcep_alpha,
        )
        gt_mcep = sptk_extract(
            x=gt_x,
            fs=fs,
            n_fft=args.n_fft,
            n_shift=args.n_shift,
            mcep_dim=args.mcep_dim,
            mcep_alpha=args.mcep_alpha,
        )

        # DTW
        _, path = fastdtw(gen_mcep, gt_mcep, dist=spatial.distance.euclidean)
        twf = np.array(path).T
        gen_mcep_dtw = gen_mcep[twf[0]]
        gt_mcep_dtw = gt_mcep[twf[1]]

        # MCD
        diff2sum = np.sum((gen_mcep_dtw - gt_mcep_dtw) ** 2, 1)
        mcd = np.mean(10.0 / np.log(10.0) * np.sqrt(2 * diff2sum), 0)
        logging.info(f"{gt_basename} {mcd:.4f}")
        mcd_dict[gt_basename] = mcd


        
def get_parser() -> argparse.Namespace:
    """Get argument parser."""
    parser = argparse.ArgumentParser(description="Evaluate Mel-cepstrum distortion.")
    parser.add_argument(
        "--outdir",
        type=str,
        help="Path of directory to write the results.",
    )

    # analysis related
    parser.add_argument(
        "--mcep_dim",
        default=None,
        type=int,
        help=(
            "Dimension of mel cepstrum coefficients. "
            "If None, automatically set to the best dimension for the sampling."
        ),
    )
    parser.add_argument(
        "--mcep_alpha",
        default=None,
        type=float,
        help=(
            "All pass constant for mel-cepstrum analysis. "
            "If None, automatically set to the best dimension for the sampling."
        ),
    )
    parser.add_argument(
        "--n_fft",
        default=1024,
        type=int,
        help="The number of FFT points.",
    )
    parser.add_argument(
        "--n_shift",
        default=256,
        type=int,
        help="The number of shift points.",
    )
    parser.add_argument(
        "--nj",
        default=16,
        type=int,
        help="Number of parallel jobs.",
    )
    parser.add_argument(
        "--verbose",
        default=1,
        type=int,
        help="Verbosity level. Higher is more logging.",
    )
    parser.add_argument("--org", help="original wav directory", type=str, required=True)
    parser.add_argument("--gen", help="generated wav directory", type=str, required=True)
    parser.add_argument("--file", help="pairings file", type=str, required=True)

    return parser


def main():
    """Run log-F0 RMSE calculation in parallel."""
    args = get_parser().parse_args()

    with open(args.file, "r", encoding="utf-8") as inf:
        data_lines = inf.read().splitlines()

    sr = 16000

    gt_files = list()
    gen_files = list()

    for line in data_lines:
        l_list = line.split("\t")
        org = l_list[0].split(".")[0] + '.wav'
        gen = l_list[1].split(".")[0] + '.wav'
        
        gt_files.append(os.path.join(args.org, org))
        gen_files.append(os.path.join(args.gen, gen))

    # Get and divide list
    if len(gen_files) == 0:
        raise FileNotFoundError("Not found any generated audio files.")
    if len(gen_files) > len(gt_files):
        raise ValueError(
            "#groundtruth files are less than #generated files "
            f"(#gen={len(gen_files)} vs. #gt={len(gt_files)}). "
            "Please check the groundtruth directory."
        )
    logging.info("The number of utterances = %d" % len(gen_files))
    file_lists = np.array_split(gen_files, args.nj)
    file_lists = [f_list.tolist() for f_list in file_lists]

    # multi processing
    with mp.Manager() as manager:
        mcd_dict = manager.dict()
        processes = []
        for f in file_lists:
            p = mp.Process(target=calculate, args=(f, gt_files, args, mcd_dict))
            p.start()
            processes.append(p)

        # wait for all process
        for p in processes:
            p.join()

        # convert to standard list
        mcd_dict = dict(mcd_dict)

        # calculate statistics
        mean_mcd = np.mean(np.array([v for v in mcd_dict.values()]))
        std_mcd = np.std(np.array([v for v in mcd_dict.values()]))
        logging.info(f"Average: {mean_mcd:.4f} ± {std_mcd:.4f}")

    # write results
    if args.outdir is None:
        args.outdir = '.'
    os.makedirs(args.outdir, exist_ok=True)
    with open(f"{args.outdir}/utt2mcd", "w") as f:
        for utt_id in sorted(mcd_dict.keys()):
            mcd = mcd_dict[utt_id]
            f.write(f"{utt_id} {mcd:.4f}\n")
    with open(f"{args.outdir}/mcd_avg_result.txt", "w") as f:
        f.write(f"#utterances: {len(gen_files)}\n")
        f.write(f"Average: {mean_mcd:.4f} ± {std_mcd:.4f}")

    logging.info("Successfully finished MCD evaluation.")


if __name__ == "__main__":
    main()

