import torch, torchaudio
import hubconf as hc

SAMPLE_RATE=16000

def create_knn_vc_instance(prematched, pretrained=False):
    # create knn_vc object. use hubconf file
    return hc.knn_vc(prematched=prematched, pretrained=pretrained, device='cpu')


def runKnn_VC(knn_vc, src_wav_path, ref_wav_paths, k=4):
    # Or, if you would like the vocoder trained not using prematched data, set prematched=False.
    query_seq = knn_vc.get_features(src_wav_path)
    matching_set = knn_vc.get_matching_set(ref_wav_paths)
    return knn_vc.match(query_seq, matching_set, topk=k)


def save_tensor_waveform_to_wav(tensor_wav, output_path, sample_rate=SAMPLE_RATE):
    torchaudio.save(output_path, tensor_wav.unsqueeze(0), sample_rate=sample_rate)


def run_full_pipline(src_wav_path, ref_wav_paths, output_path, prematched=True, pretrained=False, k=4):
    src_speaker_id = os.path.normpath(src_wav_path).split(os.sep)[-3]
    ref_speaker_id = os.path.normpath(ref_wav_paths[0]).split(os.sep)[-3]
    filename = f"{src_speaker_id}_{ref_speaker_id}.wav"
    output_path = os.path.join(output_path_dir, filename)

    knn_vc = create_knn_vc_instance(prematched)
    out_wav = runKnn_VC(knn_vc, src_wav_path, ref_wav_paths)
    save_tensor_waveform_to_wav(out_wav, output_path)



def main():
    #todo: get args from user
    # src_wav_path = "test_data/resampled/rachel/rachel_prompt3.wav"
    # ref_wav_paths = ["test_data/resampled/aviv/aviv_prompt1.wav", "test_data/resampled/aviv/aviv_prompt2.wav",
    #                  "test_data/resampled/aviv/aviv_prompt4.wav", "test_data/resampled/aviv/aviv_prompt5.wav",
    #                  "test_data/resampled/aviv/aviv_prompt6.wav", "test_data/resampled/aviv/aviv_prompt7.wav"]


if __name__ == '__main__':
    main()
