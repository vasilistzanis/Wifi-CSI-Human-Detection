import unittest
import warnings
import numpy as np

# Core Pipeline & Logic Imports
from data_preprocessing import CSIPipeline
from real_time_inference import _parse_csi_line
from live_data_visualization import parse_csi_frame
from motion_detector import compute_frame_energy

class TestCSIDataParsing(unittest.TestCase):
    """
    SECTION 1: Data extraction and software resilience.
    """
    def test_valid_ht40_parsing(self):
        """Test if real-time and live viewer scripts decode RAW payload identically."""
        dummy_payload = "[" + ",".join(["10", "-5", "4", "3"] * 32) + "]"
        line = f"CSI_DATA,1,MAC,-50,0,-90,1,1,1,1000,128,0,256,0,\"{dummy_payload}\""
        
        rt_frame = _parse_csi_line(line)
        live_frame = parse_csi_frame(line, 64)
        
        self.assertIsNotNone(rt_frame, "Real-time parsing returned None/Failed.")
        np.testing.assert_array_equal(rt_frame, live_frame)

    def test_malformed_data_resilience(self):
        """Test that bad/corrupted data from serial doesn't crash the software."""
        bad_line_1 = "CSI_DATA,1,MAC,-50,0,-90,1,1,1,1000,128" # Missing payload
        bad_line_2 = "CSI_DATA,1,MAC,-50,0,-90,1,1,1,1000,128,0,256,0,\"[10, A, B]\"" # Corrupted integers
        bad_line_3 = "RANDOM_GARBAGE_STRING"
        
        self.assertIsNone(_parse_csi_line(bad_line_1))
        self.assertIsNone(_parse_csi_line(bad_line_2))
        self.assertIsNone(_parse_csi_line(bad_line_3))


class TestCSIMathematics(unittest.TestCase):
    """
    SECTION 2: Strict computational bounds and filter verification.
    """
    def setUp(self):
        warnings.filterwarnings('ignore', category=Warning)
        self.pipeline = CSIPipeline(fs=100.0, background_frames=15, use_diff=True)

    def test_hampel_spike_removal(self):
        """Proves that Hampel filter mathematically erases noise spikes."""
        signal = np.ones((20, 64), dtype=np.float32)
        signal[10, 32] = 10000.0 # Inject massive hardware spike
        
        cleaned = self.pipeline.apply_hampel_filter(signal)
        self.assertLess(cleaned[10, 32], 5000.0, "Hampel filter failed to remove the massive spike!")

    def test_pca_execution(self):
        """Proves PCA dimensionality reduction works without breaking frames."""
        sim_matrix = (np.random.randn(30, 64) + 1j * np.random.randn(30, 64)).astype(np.complex64)
        sim_matrix[:, 0:2] = 0
        
        res = self.pipeline.fit_transform(sim_matrix, use_pca=True, n_components=10, scaler_type='standard')
        
        # 30 frames -> 29 (1 lost to diff)
        # 64 subcarriers -> 10 (reduced by PCA!)
        self.assertEqual(res.shape, (29, 10))

    def test_standard_scaler_moments(self):
        """Proves Z-score mathematically centers data to mean=0 and std=1."""
        sim_matrix = (np.random.randn(50, 64) + 1j * np.random.randn(50, 64)).astype(np.complex64)
        sim_matrix[:, 0:2] = 0
        
        res = self.pipeline.fit_transform(sim_matrix, use_pca=False, scaler_type='standard')
        
        mean = np.mean(res)
        std = np.std(res)
        self.assertTrue(np.abs(mean) < 1e-4, f"Mean expected 0, got {mean}")
        # Standard deviation might vary slightly per subcarrier, but overall should be very close to 1
        self.assertTrue(np.abs(std - 1.0) < 0.1, f"Std expected 1.0, got {std}")


class TestCSIMLCommunication(unittest.TestCase):
    """
    SECTION 3: Integration with Machine Learning limits.
    """
    def test_ml_energy_logic(self):
        """Test if motion_detector energy algorithm interfaces correctly with DSP."""
        pipeline = CSIPipeline(fs=100.0, background_frames=12, use_diff=True)
        sim_matrix = (np.random.randn(30, 64) + 1j * np.random.randn(30, 64)).astype(np.complex64)
        
        # Sequential DSP execution
        amp = pipeline.remove_null_subcarriers(sim_matrix, fit=True)
        bg_sub = pipeline.apply_background_subtraction(amp, fit=True)
        diff_samples = pipeline.apply_temporal_diff(bg_sub)
        
        # Extract energy
        energy = compute_frame_energy(diff_samples)
        
        self.assertEqual(len(energy), diff_samples.shape[0], "Energy bounds failed sync!")
        self.assertTrue(np.all(energy >= 0), "Energy cannot be negative!")


if __name__ == '__main__':
    print("================================================================")
    print("🚀 RUNNING  INTEGRATION TEST")
    print("================================================================")
    unittest.main(verbosity=2)
