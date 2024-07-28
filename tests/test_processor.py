import unittest
from datetime import datetime
import numpy as np
from babylon_sts.processor import AudioProcessor
import torch

class TestAudioProcessor(unittest.TestCase):

    def setUp(self):
        # Initialize the AudioProcessor with a test language and model
        self.processor = AudioProcessor(language_to='ua', language_from='en', model_name='tiny', sample_rate=24000)
        self.test_audio_data = self.generate_test_audio_data()

    def generate_test_audio_data(self):
        # Generate some synthetic audio data for testing
        sample_rate = 16000
        duration = 1  # 1 second
        frequency = 440  # 440 Hz - A4 note
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        audio_data = (0.5 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
        audio_data = (audio_data * 32767).astype(np.int16).tobytes()
        return audio_data

    def test_translate_text(self):
        text = "Hello, world!"
        translated_text = self.processor.translate_text(text)
        self.assertIsInstance(translated_text, str)
        self.assertGreater(len(translated_text), 0)

    def test_synthesize_speech(self):
        text = "Привіт, світ!"
        synthesized_audio = self.processor.synthesize_speech(text)
        self.assertIsInstance(synthesized_audio, torch.Tensor)
        self.assertGreater(len(synthesized_audio), 0)

    def test_recognize_speech(self):
        segments = self.processor.recognize_speech(self.test_audio_data)
        self.assertIsInstance(segments, list)
        # As this is synthetic data, the recognized segments might be empty
        # This check ensures no error occurs during recognition

    def test_process_audio(self):
        timestamp = datetime.utcnow()
        final_audio, log_data = self.processor.process_audio(timestamp, self.test_audio_data)
        self.assertIsInstance(final_audio, np.ndarray)
        if log_data is not None:
            self.assertIsInstance(log_data, dict)
            self.assertIn("timestamp", log_data)
            self.assertIn("original_text", log_data)
            self.assertIn("translated_text", log_data)
            self.assertIn("synthesis_delay", log_data)

if __name__ == '__main__':
    unittest.main()
