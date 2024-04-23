from absl import app
from absl import flags
from absl import logging

import numpy as np
import sounddevice as sd
import whisper
from detect_voice import VoiceDetector


FLAGS = flags.FLAGS
flags.DEFINE_string('model_name', 'tiny.en',
                    'The version of the OpenAI Whisper model to use.')
flags.DEFINE_string('language', 'en',
                    'The language to use or empty to auto-detect.')
flags.DEFINE_string('input_device', 'plughw:2,0',
                    'The input device used to record audio.')
flags.DEFINE_integer('sample_rate', 16000,
                     'The sample rate of the recorded audio.')
flags.DEFINE_integer('num_channels', 1,
                     'The number of channels of the recorded audio.')
flags.DEFINE_integer('channel_index', 0,
                     'The index of the channel to use for transcription.')
flags.DEFINE_integer('chunk_seconds', 3,
                     'The length in seconds of each recorded chunk of audio.')
flags.DEFINE_string('latency', 'low', 'The latency of the recording stream.')


def record_audio(duration, fs):
    """Record audio for the given duration and sample rate."""
    audio = sd.rec(int(duration * fs), blocking=True)

    # Consider only the indicated channel
    return audio[:, FLAGS.channel_index].copy()


def transcribe(model, audio):
    # Run the Whisper model to transcribe the audio chunk.
    result = whisper.transcribe(model=model, audio=audio)

    # Use the transcribed text.
    text = result['text'].strip()

    return text



def main(argv):
    # Load the Whisper model into memory, downloading first if necessary.
    logging.info(f'Loading Whisper model "{FLAGS.model_name}"...')
    model = whisper.load_model(name=FLAGS.model_name)

    # The first run of the model is slow (buffer init), so run it once empty.
    logging.info('Warming Whisper model up...')
    block_size = FLAGS.chunk_seconds * FLAGS.sample_rate
    whisper.transcribe(model=model,
                       audio=np.zeros(block_size, dtype=np.float32))
    
    logging.info("")
    voice_detector = VoiceDetector()

    # Set recording parameters
    sd.default.device = FLAGS.input_device
    sd.default.channels = FLAGS.num_channels
    sd.default.dtype = np.float32
    sd.default.latency = FLAGS.latency
    sd.default.samplerate = FLAGS.sample_rate

    try:
        while True:
            logging.info('Recording')
            audio = record_audio(duration=FLAGS.chunk_seconds, fs=FLAGS.sample_rate)
            
            logging.info("Detecting voice")
            speech_timestamps, cutout = voice_detector.detect_voice(audio, FLAGS.sample_rate)
            
            if speech_timestamps:
                logging.info(f'Voice detected at {speech_timestamps}, processing command')
                command = transcribe(model, cutout)
                logging.info(f"Received command: '{command}'")
            else:
                logging.info("No voice detected")

    except KeyboardInterrupt:
        print("Interrupted by user")

    # # Stream audio chunks into a queue and process them from there. The
    # # callback is running on a separate thread.
    # logging.info('Starting stream...')
    # audio_queue = queue.Queue()
    # callback = partial(stream_callback, audio_queue=audio_queue)
    # with sd.InputStream(samplerate=FLAGS.sample_rate,
    #                     blocksize=block_size,
    #                     device=FLAGS.input_device,
    #                     channels=FLAGS.num_channels,
    #                     dtype=np.float32,
    #                     latency=FLAGS.latency,
    #                     callback=callback):
    #     while True:
    #         # Process chunks of audio from the queue.
    #         process_audio(audio_queue, model)


if __name__ == '__main__':
    app.run(main)
