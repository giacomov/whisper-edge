from io import BytesIO
import wave
import torch
import os


class VoiceDetector:

    def __init__(self, device='cuda'):

        try:
            model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad:v3.1', model='silero_vad')
        except ValueError:
            # This is expected beacuse of a bug in hubconf.py in the silero vad repo
            
            hub_orig = '.cache/torch/hub/snakers4_silero-vad_v3.1'
            hub_dest = '.cache/torch/hub/snakers4_silero-vad_master'

            os.rename(
                os.path.join(os.path.expanduser(f"~/{hub_orig}")),
                os.path.join(os.path.expanduser(f"~/{hub_dest}")),
            )

            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad:v3.1', 
                model='silero_vad'
            )
        
        (
            self._get_speech_timestamps,
            self._save_audio,
            self._read_audio,
            VADIterator,
            self._collect_chunks
        ) = utils

        self._model = model
        self._device = device
        self._model.to(self._device)

        # Need to warmup because the first inference is very slow
        self._warmup()
    
    def _warmup(self):

        # Warmup model with 0.5 s of random data
        wav = torch.rand(8000, dtype=torch.float32, device=self._device)

        self._get_speech_timestamps(
            wav, 
            self._model, 
            sampling_rate=16000,
            return_seconds=True
        )

    def detect_voice(self, audio_data, sampling_rate):

        wav = torch.from_numpy(audio_data).to(self._device)

        # self._save_audio('only_speech.wav', wav, sampling_rate=sampling_rate)

        speech_timestamps = self._get_speech_timestamps(
            wav, 
            self._model, 
            sampling_rate=sampling_rate,
            return_seconds=False
        )

        if speech_timestamps:
            return speech_timestamps, self._collect_chunks(speech_timestamps, wav)
        else:
            return None, None
