# OmniResponse reference metric evaluation

Status: complete
Samples per seed: 1; seeds: [137]

FVD-32; 25 fps, stride 1, deterministic paired window.
FD is supplied-feature distance, not Inception FID. UTMOS/SyncNet use listener audio.
Full-set scores are null for incomplete coverage; supported subsets and exact IDs remain in JSON.

## Seed 137

```json
{
  "fvd": {
    "full_value": null,
    "supported_subset_value": null,
    "coverage": {
      "planned": 1,
      "eligible": 1,
      "succeeded": 1,
      "inapplicable": 0,
      "failed": 0,
      "succeeded_ids": [
        {
          "sample_id": "naturalistic_V03_S0876_I00000149_local_f808d9650926",
          "sample_seed": 137
        }
      ],
      "inapplicable_ids": [],
      "failed_ids": []
    },
    "reason": "need_at_least_two_supported_samples"
  },
  "utmos": {
    "gt": {
      "full_value": 1.0078125,
      "supported_subset_value": 1.0078125,
      "coverage": {
        "planned": 1,
        "eligible": 1,
        "succeeded": 1,
        "inapplicable": 0,
        "failed": 0,
        "succeeded_ids": [
          {
            "sample_id": "naturalistic_V03_S0876_I00000149_local_f808d9650926",
            "sample_seed": 137
          }
        ],
        "inapplicable_ids": [],
        "failed_ids": []
      }
    },
    "generated": {
      "full_value": 0.61767578125,
      "supported_subset_value": 0.61767578125,
      "coverage": {
        "planned": 1,
        "eligible": 1,
        "succeeded": 1,
        "inapplicable": 0,
        "failed": 0,
        "succeeded_ids": [
          {
            "sample_id": "naturalistic_V03_S0876_I00000149_local_f808d9650926",
            "sample_seed": 137
          }
        ],
        "inapplicable_ids": [],
        "failed_ids": []
      }
    }
  },
  "sync": {
    "gt": {
      "full_value": {
        "lse_d": 4.750168323516846,
        "lse_c": 0.48270368576049805
      },
      "supported_subset_value": {
        "lse_d": 4.750168323516846,
        "lse_c": 0.48270368576049805
      },
      "coverage": {
        "planned": 1,
        "eligible": 1,
        "succeeded": 1,
        "inapplicable": 0,
        "failed": 0,
        "succeeded_ids": [
          {
            "sample_id": "naturalistic_V03_S0876_I00000149_local_f808d9650926",
            "sample_seed": 137
          }
        ],
        "inapplicable_ids": [],
        "failed_ids": []
      },
      "unit": "equal weight per supported face track",
      "tracks": 1,
      "low_support_tracks": 0,
      "clip_equal_mean": {
        "lse_d": 4.750168323516846,
        "lse_c": 0.48270368576049805
      }
    },
    "generated": {
      "full_value": {
        "lse_d": 6.915524482727051,
        "lse_c": 0.16041326522827148
      },
      "supported_subset_value": {
        "lse_d": 6.915524482727051,
        "lse_c": 0.16041326522827148
      },
      "coverage": {
        "planned": 1,
        "eligible": 1,
        "succeeded": 1,
        "inapplicable": 0,
        "failed": 0,
        "succeeded_ids": [
          {
            "sample_id": "naturalistic_V03_S0876_I00000149_local_f808d9650926",
            "sample_seed": 137
          }
        ],
        "inapplicable_ids": [],
        "failed_ids": []
      },
      "unit": "equal weight per supported face track",
      "tracks": 1,
      "low_support_tracks": 0,
      "clip_equal_mean": {
        "lse_d": 6.915524482727051,
        "lse_c": 0.16041326522827148
      }
    }
  }
}
```
