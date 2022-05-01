#!/usr/bin/env python3

# Referred from data_prep.py in jv_openslr35 in ESPnet
# https://github.com/espnet/espnet/blob/master/egs2/jv_openslr35/
# asr1/local/data_prep.py



#!/usr/bin/env python3

# Copyright 2021 Wen-Chin Huang and Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Evaluate log-F0 RMSE between generated and groundtruth audios based on World."""

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
import pyworld as pw
import soundfile as sf

from fastdtw import fastdtw
from scipy import spatial


def world_extract(
    x: np.ndarray,
    fs: int,
    f0min: int = 40,
    f0max: int = 800,
    n_fft: int = 512,
    n_shift: int = 256,
    mcep_dim: int = 25,
    mcep_alpha: float = 0.41,
) -> np.ndarray:
    """Extract World-based acoustic features.
    Args:
        x (ndarray): 1D waveform array.
        fs (int): Minimum f0 value (default=40).
        f0 (int): Maximum f0 value (default=800).
        n_shift (int): Shift length in point (default=256).
        n_fft (int): FFT length in point (default=512).
        n_shift (int): Shift length in point (default=256).
        mcep_dim (int): Dimension of mel-cepstrum (default=25).
        mcep_alpha (float): All pass filter coefficient (default=0.41).
    Returns:
        ndarray: Mel-cepstrum with the size (N, n_fft).
        ndarray: F0 sequence (N,).
    """
    # extract features
    x = x.astype(np.float64)
    f0, time_axis = pw.harvest(
        x,
        fs,
        f0_floor=f0min,
        f0_ceil=f0max,
        frame_period=n_shift / fs * 1000,
    )
    sp = pw.cheaptrick(x, f0, time_axis, fs, fft_size=n_fft)
    if mcep_dim is None or mcep_alpha is None:
        mcep_dim, mcep_alpha = _get_best_mcep_params(fs)
    mcep = pysptk.sp2mc(sp, mcep_dim, mcep_alpha)

    return mcep, f0


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
    f0_rmse_dict: Dict[str, float],
):
    """Calculate log-F0 RMSE."""
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
        gen_mcep, gen_f0 = world_extract(
            x=gen_x,
            fs=fs,
            f0min=args.f0min,
            f0max=args.f0max,
            n_fft=args.n_fft,
            n_shift=args.n_shift,
            mcep_dim=args.mcep_dim,
            mcep_alpha=args.mcep_alpha,
        )
        gt_mcep, gt_f0 = world_extract(
            x=gt_x,
            fs=fs,
            f0min=args.f0min,
            f0max=args.f0max,
            n_fft=args.n_fft,
            n_shift=args.n_shift,
            mcep_dim=args.mcep_dim,
            mcep_alpha=args.mcep_alpha,
        )

        # DTW
        _, path = fastdtw(gen_mcep, gt_mcep, dist=spatial.distance.euclidean)
        twf = np.array(path).T
        gen_f0_dtw = gen_f0[twf[0]]
        gt_f0_dtw = gt_f0[twf[1]]

        # Get voiced part
        nonzero_idxs = np.where((gen_f0_dtw != 0) & (gt_f0_dtw != 0))[0]
        gen_f0_dtw_voiced = np.log(gen_f0_dtw[nonzero_idxs])
        gt_f0_dtw_voiced = np.log(gt_f0_dtw[nonzero_idxs])

        # log F0 RMSE
        log_f0_rmse = np.sqrt(np.mean((gen_f0_dtw_voiced - gt_f0_dtw_voiced) ** 2))
        logging.info(f"{gt_basename} {log_f0_rmse:.4f}")
        f0_rmse_dict[gt_basename] = log_f0_rmse


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
        "--f0min",
        default=40,
        type=int,
        help="Minimum f0 value.",
    )
    parser.add_argument(
        "--f0max",
        default=800,
        type=int,
        help="Maximum f0 value.",
    )
    parser.add_argument(
        "--nj",
        default=1,
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
        log_f0_rmse_dict = manager.dict()
        processes = []
        # for f in file_lists:
        #     calculate(f, gt_files, args, log_f0_rmse_dict)
        for f in file_lists:
            p = mp.Process(target=calculate, args=(f, gt_files, args, log_f0_rmse_dict))
            p.start()
            processes.append(p)

        # wait for all process
        for p in processes:
            p.join()

        # convert to standard list
        log_f0_rmse_dict = dict(log_f0_rmse_dict)

        # calculate statistics
        mean_log_f0_rmse = np.mean(np.array([v for v in log_f0_rmse_dict.values()]))
        std_log_f0_rmse = np.std(np.array([v for v in log_f0_rmse_dict.values()]))
        logging.info(f"Average: {mean_log_f0_rmse:.4f} ± {std_log_f0_rmse:.4f}")

    # write results
    if args.outdir is None:
        args.outdir = '.'
    with open(f"{args.outdir}/utt2log_f0_rmse", "w") as f:
        for utt_id in sorted(log_f0_rmse_dict.keys()):
            log_f0_rmse = log_f0_rmse_dict[utt_id]
            f.write(f"{utt_id} {log_f0_rmse:.4f}\n")
    with open(f"{args.outdir}/log_f0_rmse_avg_result.txt", "w") as f:
        f.write(f"#utterances: {len(gen_files)}\n")
        f.write(f"Average: {mean_log_f0_rmse:.4f} ± {std_log_f0_rmse:.4f}")

    logging.info("Successfully finished log-F0 RMSE evaluation.")


if __name__ == "__main__":
    main()

