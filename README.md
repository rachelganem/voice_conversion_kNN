# Voice Conversion With Just Nearest Neighbors
Links:
- Arvix Paper:
- Official Code Repo:

We want to be able to convert source voice to target voice without additional training.
The method includes three steps:
1. Encode source and reference utterances uning WavLM
2. Each source feature is assigned to the mean of the k closest features from the reference.
3. The resulting feature sequence is then vocoded with HiFi-GAN to arrive at the converted waveform output.

In the following code repo we will implement by ourselves the paper above, but we'll try to training the encoder and the decoder with a different dataset.

