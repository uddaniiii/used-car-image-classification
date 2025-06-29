import numpy as np
import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Geometric Mean Ensemble")
    parser.add_argument("--npy_files", nargs='+', required=True)
    parser.add_argument("--sample_submission", type=str, default='./data/sample_submission.csv')
    parser.add_argument("--output_csv", type=str, default='geo_ensemble_submission.csv')
    return parser.parse_args()

def main():
    args = parse_args()
    probs_list = [np.load(file) for file in args.npy_files]
    for i, prob in enumerate(probs_list):
        print(f"[{i}] shape: {prob.shape}")

    log_probs = [np.log(p + 1e-12) for p in probs_list]
    geo_mean = np.exp(np.mean(log_probs, axis=0))
    ensemble_probs = geo_mean / np.sum(geo_mean, axis=1, keepdims=True)

    submission = pd.read_csv(args.sample_submission, encoding='utf-8-sig')
    submission[submission.columns[1:]] = ensemble_probs
    submission.to_csv(args.output_csv, index=False, encoding='utf-8-sig', float_format="%.16f")
    print(f"✅ 기하 평균 앙상블 완료: {args.output_csv}")

if __name__ == "__main__":
    main()
