import torch, torchaudio
import hubconf as hc


def create_knn_vc_instance():
    # create knn_vc object. use hubconf file
    return hc.knn_vc(prematched=True, pretrained=True)


def runKnn_VC(knn_vc, src_wav_path, ref_wav_paths):
    # Or, if you would like the vocoder trained not using prematched data, set prematched=False.
    query_seq = knn_vc.get_features(src_wav_path)
    matching_set = knn_vc.get_matching_set(ref_wav_paths)
    return knn_vc.match(query_seq, matching_set, topk=4)
    # out_wav is (T,) tensor converted 16kHz output wav using k=4 for kNN.


def save_tensor_waveform_to_wav(tensor_wav, output_path="benchmark_output", sample_rate=16000):
    torchaudio.save(f"{output_path}/output.wav", tensor_wav.unsqueeze(0), sample_rate=16000)


def main():
    src_wav_path = "test_data/resampled/rachel/rachel_prompt1.wav"
    ref_wav_paths = ["test_data/resampled/aviv/aviv_prompt1.wav", "test_data/resampled/aviv/aviv_prompt2.wav"]
    print("=====Create knn vc instance====")
    knn_vc = create_knn_vc_instance()
    print("=====Run knn vc instance====")
    out_wav = runKnn_VC(knn_vc, src_wav_path, ref_wav_paths)
    print("=====Convert and save results=====")
    save_tensor_waveform_to_wav(out_wav)


if __name__ == '__main__':
    main()
