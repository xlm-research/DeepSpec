import math
import unittest

import torch

from deepspec.eval.dspark.scheduler import (
    HardwareAwarePrefixScheduler,
    SPSProfile,
    SequentialTemperatureScaler,
    expected_calibration_error,
)


class SequentialTemperatureScalingTest(unittest.TestCase):
    def test_expected_calibration_error_uses_equal_width_bins(self):
        probabilities = torch.tensor([0.1, 0.4, 0.6, 0.9])
        targets = torch.tensor([0, 0, 1, 1])

        ece = expected_calibration_error(
            probabilities,
            targets,
            num_bins=2,
        )

        self.assertAlmostEqual(float(ece.item()), 0.25)

    def test_fit_calibrates_prefixes_sequentially(self):
        raw_probability = 0.8
        raw_logit = math.log(raw_probability / (1.0 - raw_probability))
        confidence_logits = torch.full((4, 2), raw_logit, dtype=torch.float64)
        prefix_targets = torch.tensor(
            [[1, 1], [1, 1], [0, 0], [0, 0]],
            dtype=torch.float64,
        )

        scaler = SequentialTemperatureScaler.fit(
            confidence_logits,
            prefix_targets,
            temperature_grid=[0.5, 1.0, 2.0],
            num_bins=10,
        )

        torch.testing.assert_close(
            scaler.temperatures,
            torch.tensor([2.0, 1.0], dtype=torch.float64),
        )
        calibrated = scaler.calibrate_logits(confidence_logits)
        expected_conditionals = torch.tensor(
            [2.0 / 3.0, 0.8],
            dtype=torch.float64,
        ).expand_as(calibrated)
        torch.testing.assert_close(calibrated, expected_conditionals)
        calibrated_prefix = calibrated.cumprod(dim=1)
        raw_prefix = confidence_logits.sigmoid().cumprod(dim=1)
        calibrated_ece = expected_calibration_error(
            calibrated_prefix[:, 1],
            prefix_targets[:, 1],
            num_bins=10,
        )
        raw_ece = expected_calibration_error(
            raw_prefix[:, 1],
            prefix_targets[:, 1],
            num_bins=10,
        )
        self.assertLess(float(calibrated_ece.item()), float(raw_ece.item()))

    def test_probability_and_logit_calibration_agree(self):
        scaler = SequentialTemperatureScaler(
            temperatures=torch.tensor([0.5, 1.0, 2.0]),
        )
        logits = torch.tensor([[-2.0, 0.0, 2.0], [1.0, -1.0, 0.5]])

        from_logits = scaler.calibrate_logits(logits)
        from_probabilities = scaler.calibrate_probabilities(logits.sigmoid())

        torch.testing.assert_close(from_logits, from_probabilities)

    def test_calibration_applies_temperature_to_logits(self):
        scaler = SequentialTemperatureScaler(
            temperatures=torch.tensor([2.0, 0.5]),
        )
        logits = torch.tensor([[math.log(4.0), -math.log(4.0)]])

        calibrated = scaler.calibrate_logits(logits)

        torch.testing.assert_close(
            calibrated,
            torch.tensor([[2.0 / 3.0, 1.0 / 17.0]]),
        )
        torch.testing.assert_close(
            calibrated.cumprod(dim=1),
            torch.tensor([[2.0 / 3.0, 2.0 / 51.0]]),
        )

    def test_grid_ties_prefer_temperature_closest_to_identity(self):
        logits = torch.zeros((2, 1))
        targets = torch.tensor([[1], [0]])

        scaler = SequentialTemperatureScaler.fit(
            logits,
            targets,
            temperature_grid=[2.0, 0.5, 1.0],
        )

        self.assertEqual(scaler.temperatures.tolist(), [1.0])

    def test_temperature_scaling_preserves_ranking_per_position(self):
        scaler = SequentialTemperatureScaler(
            temperatures=torch.tensor([0.5, 2.0]),
        )
        logits = torch.tensor([[-2.0, 3.0], [0.0, 1.0], [2.0, -1.0]])

        calibrated = scaler.calibrate_logits(logits)

        for position in range(logits.shape[1]):
            self.assertEqual(
                torch.argsort(logits[:, position]).tolist(),
                torch.argsort(calibrated[:, position]).tolist(),
            )

    def test_fit_rejects_non_prefix_labels_and_mask_holes(self):
        logits = torch.zeros((2, 3))
        with self.assertRaisesRegex(ValueError, "non-increasing"):
            SequentialTemperatureScaler.fit(
                logits,
                torch.tensor([[0, 1, 0], [1, 1, 0]]),
                temperature_grid=[1.0],
            )
        with self.assertRaisesRegex(ValueError, "contiguous prefix"):
            SequentialTemperatureScaler.fit(
                logits,
                torch.tensor([[1, 0, 0], [1, 1, 0]]),
                temperature_grid=[1.0],
                valid_mask=torch.tensor(
                    [[True, False, True], [True, True, True]]
                ),
            )

    def test_fit_rejects_invalid_grid_and_missing_position_data(self):
        logits = torch.zeros((2, 2))
        targets = torch.ones((2, 2))
        with self.assertRaisesRegex(ValueError, "duplicates"):
            SequentialTemperatureScaler.fit(
                logits,
                targets,
                temperature_grid=[1.0, 1.0],
            )
        with self.assertRaisesRegex(ValueError, "position 1"):
            SequentialTemperatureScaler.fit(
                logits,
                targets,
                temperature_grid=[1.0],
                valid_mask=torch.tensor([[True, False], [True, False]]),
            )


class SPSProfileTest(unittest.TestCase):
    def test_mapping_and_dense_profiles_use_exact_integer_lookups(self):
        mapping_profile = SPSProfile.from_mapping({3: 8.0, 1: 10.0, 2: 9.0})
        dense_profile = SPSProfile.from_dense([10.0, 9.0, 8.0])

        self.assertEqual(mapping_profile, dense_profile)
        self.assertEqual(mapping_profile.lookup(2), 9.0)
        with self.assertRaisesRegex(ValueError, "missing batch size 4"):
            mapping_profile.lookup(4)

    def test_profile_rejects_invalid_axes_and_rates(self):
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            SPSProfile((1, 1), (10.0, 9.0))
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            SPSProfile((1,), (0.0,))
        with self.assertRaisesRegex(ValueError, "positive integer"):
            SPSProfile((0,), (1.0,))


class HardwareAwarePrefixSchedulerTest(unittest.TestCase):
    def test_algorithm_one_selects_best_point_on_greedy_path(self):
        probabilities = torch.tensor([[0.9, 0.9], [0.8, 0.5]])
        scheduler = HardwareAwarePrefixScheduler(
            {2: 10.0, 3: 9.0, 4: 8.0, 5: 7.0, 6: 5.0}
        )

        result = scheduler.schedule(probabilities)

        self.assertEqual(result.prefix_lengths.tolist(), [2, 1])
        torch.testing.assert_close(
            result.prefix_survival_probabilities,
            torch.tensor([[0.9, 0.81], [0.8, 0.4]], dtype=torch.float64),
        )
        self.assertEqual(result.admitted_draft_tokens, 3)
        self.assertEqual(result.target_batch_size, 5)
        self.assertAlmostEqual(result.expected_accepted_tokens, 4.51, places=6)
        self.assertAlmostEqual(result.expected_throughput, 4.51 * 7.0, places=6)
        self.assertEqual(
            scheduler.select_prefix_lengths(probabilities).tolist(),
            [2, 1],
        )

    def test_equal_confidence_ties_admit_shallow_positions_first(self):
        probabilities = torch.ones((2, 2))
        scheduler = HardwareAwarePrefixScheduler(
            {2: 10.0, 3: 9.0, 4: 8.0, 5: 5.0, 6: 4.0}
        )

        result = scheduler.schedule(probabilities)

        self.assertEqual(result.prefix_lengths.tolist(), [1, 1])
        self.assertEqual(result.target_batch_size, 4)

    def test_first_non_improvement_stops_before_later_recovery(self):
        probabilities = torch.tensor([[0.9, 0.9]])
        scheduler = HardwareAwarePrefixScheduler({1: 10.0, 2: 5.0, 3: 100.0})

        result = scheduler.schedule(probabilities)

        self.assertEqual(result.prefix_lengths.tolist(), [0])
        self.assertEqual(result.target_batch_size, 1)
        self.assertEqual(result.expected_accepted_tokens, 1.0)
        self.assertEqual(result.expected_throughput, 10.0)

    def test_suffix_confidence_cannot_change_decision_after_first_decline(self):
        scheduler = HardwareAwarePrefixScheduler({1: 1.0, 2: 0.5, 3: 0.45})

        high_suffix = scheduler.schedule(torch.tensor([[0.8, 0.9]]))
        zero_suffix = scheduler.schedule(torch.tensor([[0.8, 0.0]]))

        self.assertEqual(high_suffix.prefix_lengths.tolist(), [0])
        self.assertEqual(zero_suffix.prefix_lengths.tolist(), [0])

    def test_equal_throughput_does_not_replace_baseline(self):
        scheduler = HardwareAwarePrefixScheduler({1: 10.0, 2: 5.0})

        result = scheduler.schedule(torch.ones((1, 1)))

        self.assertEqual(result.prefix_lengths.tolist(), [0])
        self.assertEqual(result.expected_throughput, 10.0)

    def test_zero_survival_requires_only_baseline_profile(self):
        scheduler = HardwareAwarePrefixScheduler({2: 7.0})

        result = scheduler.schedule(torch.zeros((2, 3)))

        self.assertEqual(result.prefix_lengths.tolist(), [0, 0])
        self.assertEqual(result.target_batch_size, 2)
        self.assertEqual(result.expected_accepted_tokens, 2.0)
        self.assertEqual(result.expected_throughput, 14.0)

    def test_empty_block_returns_target_only_schedule(self):
        scheduler = HardwareAwarePrefixScheduler({2: 7.0})

        result = scheduler.schedule(torch.empty((2, 0)))

        self.assertEqual(result.prefix_lengths.tolist(), [0, 0])
        self.assertEqual(tuple(result.prefix_survival_probabilities.shape), (2, 0))

    def test_profile_must_cover_every_candidate_batch_size(self):
        scheduler = HardwareAwarePrefixScheduler({1: 10.0, 2: 9.0})

        with self.assertRaisesRegex(ValueError, "missing 3"):
            scheduler.schedule(torch.ones((1, 2)))

    def test_invalid_confidence_probabilities_are_rejected(self):
        scheduler = HardwareAwarePrefixScheduler({1: 10.0})
        for probabilities in (
            torch.tensor([0.5]),
            torch.empty((0, 2)),
            torch.tensor([[1.1]]),
            torch.tensor([[float("nan")]]),
        ):
            with self.subTest(shape=tuple(probabilities.shape)):
                with self.assertRaises(ValueError):
                    scheduler.schedule(probabilities)


if __name__ == "__main__":
    unittest.main()
