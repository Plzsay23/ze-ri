import os
import time
import queue
import wave
import tempfile
import numpy as np
import sounddevice as sd


class AudioInput:
    def __init__(
        self,
        sample_rate=16000,
        channels=6,
        frame_ms=30,
        silence_duration_ms=1000,
        min_speech_ms=300,
        pre_roll_ms=200,
        energy_threshold=0.01,
        output_dir=None,
        device=None,
        use_channel_index=0,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.frame_ms = frame_ms
        self.frame_samples = int(sample_rate * frame_ms / 1000)
        self.silence_frames = max(1, silence_duration_ms // frame_ms)
        self.min_speech_frames = max(1, min_speech_ms // frame_ms)
        self.pre_roll_frames = max(1, pre_roll_ms // frame_ms)
        self.energy_threshold = energy_threshold
        self.device = device
        self.use_channel_index = use_channel_index
        self.q = queue.Queue()

        if output_dir is None:
            output_dir = os.path.join(tempfile.gettempdir(), "stt_segments")
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _callback(self, indata, frames, time_info, status):
        if status:
            pass

        if indata.ndim == 1:
            chunk = indata.copy()
        else:
            ch = min(self.use_channel_index, indata.shape[1] - 1)
            chunk = indata[:, ch].copy()

        self.q.put(chunk)

    def _is_speech(self, chunk: np.ndarray):
        if chunk is None or len(chunk) == 0:
            return False
        rms = np.sqrt(np.mean(np.square(chunk), dtype=np.float64))
        return rms >= self.energy_threshold

    def _save_wav(self, audio: np.ndarray):
        audio = np.clip(audio, -1.0, 1.0)
        pcm16 = (audio * 32767.0).astype(np.int16)

        ts = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(
            self.output_dir,
            f"utt_{ts}_{int(time.time() * 1000) % 1000:03d}.wav",
        )

        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm16.tobytes())

        return path

    def record_utterance(self):
        pre_buffer = []
        speech_buffer = []
        in_speech = False
        silence_count = 0
        speech_frame_count = 0

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=self.frame_samples,
            callback=self._callback,
            device=self.device,
        ):
            while True:
                chunk = self.q.get()

                if len(pre_buffer) >= self.pre_roll_frames:
                    pre_buffer.pop(0)
                pre_buffer.append(chunk)

                speech = self._is_speech(chunk)

                if not in_speech:
                    if speech:
                        in_speech = True
                        speech_buffer = pre_buffer.copy()
                        speech_frame_count = len(speech_buffer)
                        silence_count = 0
                else:
                    speech_buffer.append(chunk)
                    speech_frame_count += 1

                    if speech:
                        silence_count = 0
                    else:
                        silence_count += 1

                    if silence_count >= self.silence_frames:
                        if speech_frame_count >= self.min_speech_frames:
                            audio = np.concatenate(speech_buffer, axis=0)
                            return self._save_wav(audio)

                        in_speech = False
                        speech_buffer = []
                        silence_count = 0
                        speech_frame_count = 0