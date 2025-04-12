import torch, torchaudio
import argparse
from src.knnvc import hubconf as hc

SAMPLE_RATE = 16000


def create_knn_vc_instance(prematched, device, use_custom_path):
    return hc.knn_vc(prematched=prematched, device=device, use_custom_path=use_custom_path)


def runKnn_VC(knn_vc, src_wav_path, ref_wav_paths, k=4):
    query_seq = knn_vc.get_features(src_wav_path)
    matching_set = knn_vc.get_matching_set(ref_wav_paths)
    return knn_vc.match(query_seq, matching_set, topk=k)


def save_tensor_waveform_to_wav(tensor_wav, output_path, sample_rate=SAMPLE_RATE):
    torchaudio.save(output_path, tensor_wav.unsqueeze(0), sample_rate=sample_rate)


def run_full_pipeline(src_wav_path, ref_wav_paths, output_path, device, prematched=True, k=4, use_custom_path=True):
    knn_vc = create_knn_vc_instance(prematched, device, use_custom_path)
    out_wav = runKnn_VC(knn_vc, src_wav_path, ref_wav_paths, k)
    save_tensor_waveform_to_wav(out_wav, output_path)

def run_full_pipeline_on_existing_knnvc(knn_vc, src_wav_path, ref_wav_paths, output_path, k=4):
    out_wav = runKnn_VC(knn_vc, src_wav_path, ref_wav_paths, k)
    save_tensor_waveform_to_wav(out_wav, output_path)


# #todo: delete this function if not needed. there may be problems related to the prematched
# def run_prematched_pipeline(source_feat_path, target_feat_paths, output_path, device, prematched=True, k=4,
#                             use_custom_path=True):
#     knn_vc = create_knn_vc_instance(prematched, device, use_custom_path)
#     query_seq = torch.load(source_feat_path, map_location=device)
#     matching_set = [torch.load(p, map_location=device) for p in target_feat_paths]
#     out_wav = knn_vc.match(query_seq, matching_set, topk=k)
#     save_tensor_waveform_to_wav(out_wav, output_path)


def main():
    parser = argparse.ArgumentParser(description="Run kNN-VC inference pipeline (wav or prematched mode)")
    parser.add_argument("--src", required=True, help="Path to source audio (.wav) or source feature (.pt)")
    parser.add_argument("--refs", nargs='+', required=True,
                        help="List of reference audio (.wav) or feature (.pt) paths")
    parser.add_argument("--out", required=True, help="Output path for the synthesized .wav file")
    parser.add_argument("--device", default="cuda", help="Device to use for inference (cuda or cpu)")
    parser.add_argument("--k", type=int, default=4, help="Number of nearest neighbors (k)")
    parser.add_argument("--prematched", action="store_true", help="Use vocoder trained with prematched features")
    parser.add_argument("--use_custom_path", action="store_true",
                        help="Load local custom vocoder weights instead of pretrained")

    args = parser.parse_args()

    print("Running full pipeline (with audio input)...")
    run_full_pipeline(
            src_wav_path=args.src,
            ref_wav_paths=args.refs,
            output_path=args.out,
            device=args.device,
            prematched=args.prematched,
            k=args.k,
            use_custom_path=args.use_custom_path
    )


if __name__ == '__main__':
    main()
