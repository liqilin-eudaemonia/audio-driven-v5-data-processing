# OmniResponse reference metric evaluation

Status: complete
Samples per seed: 24; seeds: [137, 251]

FVD-32; 25 fps, stride 1, deterministic paired window.
FD is supplied-feature distance, not Inception FID. UTMOS/SyncNet use listener audio.
Full-set scores are null for incomplete coverage; supported subsets and exact IDs remain in JSON.

## Seed 137

```json
{
  "fvd": {
    "full_value": 513.8122074825878,
    "supported_subset_value": 513.8122074825878,
    "coverage": {
      "planned": 24,
      "eligible": 24,
      "succeeded": 24,
      "inapplicable": 0,
      "failed": 0,
      "succeeded_ids": [
        {
          "sample_id": "naturalistic_V03_S0208_I00000479_local_c9ab05da27c1",
          "sample_seed": 137
        },
        {
          "sample_id": "naturalistic_V03_S0766_I00000125_local_0aadd062235e",
          "sample_seed": 137
        },
        {
          "sample_id": "naturalistic_V03_S0766_I00000138_local_c74cd9968a0d",
          "sample_seed": 137
        },
        {
          "sample_id": "naturalistic_V03_S0921_I00000314_local_a5b50e764dc3",
          "sample_seed": 137
        },
        {
          "sample_id": "naturalistic_V03_S0921_I00000314_local_b69cf369a34c",
          "sample_seed": 137
        },
        {
          "sample_id": "naturalistic_V03_S0921_I00000317_local_dc937fcbdd9a",
          "sample_seed": 137
        },
        {
          "sample_id": "naturalistic_V03_S0921_I00000323_local_d1b4c7b0ed42",
          "sample_seed": 137
        },
        {
          "sample_id": "naturalistic_V03_S1493_I00000479_local_79a47ef19212",
          "sample_seed": 137
        },
        {
          "sample_id": "naturalistic_V03_S1493_I00000479_local_dd7b63f3aabf",
          "sample_seed": 137
        },
        {
          "sample_id": "naturalistic_V03_S1493_I00000479_local_e294647d2c81",
          "sample_seed": 137
        },
        {
          "sample_id": "naturalistic_V03_S1493_I00000480_local_2b8078b726ef",
          "sample_seed": 137
        },
        {
          "sample_id": "naturalistic_V03_S1493_I00000480_local_ba61764a5a65",
          "sample_seed": 137
        },
        {
          "sample_id": "naturalistic_V03_S1493_I00000481_local_135b540c056d",
          "sample_seed": 137
        },
        {
          "sample_id": "naturalistic_V03_S1493_I00000484_local_304b29c7cd1b",
          "sample_seed": 137
        },
        {
          "sample_id": "naturalistic_V03_S1493_I00000495_local_37d369c272eb",
          "sample_seed": 137
        },
        {
          "sample_id": "naturalistic_V03_S1493_I00000495_local_5fa0cac62f51",
          "sample_seed": 137
        },
        {
          "sample_id": "naturalistic_V03_S1493_I00000560_local_10a4cebbda73",
          "sample_seed": 137
        },
        {
          "sample_id": "naturalistic_V03_S1493_I00000560_local_5753094a1438",
          "sample_seed": 137
        },
        {
          "sample_id": "naturalistic_V03_S1498_I00000015_local_49029f1fd2a7",
          "sample_seed": 137
        },
        {
          "sample_id": "naturalistic_V03_S1498_I00000018_local_714e74cdf6fd",
          "sample_seed": 137
        },
        {
          "sample_id": "naturalistic_V03_S1498_I00000019_local_a9a1bb029158",
          "sample_seed": 137
        },
        {
          "sample_id": "naturalistic_V03_S1731_I00000363_local_0e536e54fb3b",
          "sample_seed": 137
        },
        {
          "sample_id": "naturalistic_V03_S1731_I00000363_local_6c8c62fce166",
          "sample_seed": 137
        },
        {
          "sample_id": "naturalistic_V03_S1731_I00000364_local_32855d9c1987",
          "sample_seed": 137
        }
      ],
      "inapplicable_ids": [],
      "failed_ids": []
    },
    "reason": null,
    "raw_reference_value": 513.8122074825878,
    "negative_roundoff_clamped": false,
    "roundoff_tolerance": 0.0016556658620650066,
    "name": "FVD-32"
  },
  "utmos": {
    "gt": {
      "full_value": null,
      "supported_subset_value": 1.0605817522321428,
      "coverage": {
        "planned": 24,
        "eligible": 14,
        "succeeded": 14,
        "inapplicable": 10,
        "failed": 0,
        "succeeded_ids": [
          {
            "sample_id": "naturalistic_V03_S0766_I00000125_local_0aadd062235e",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S0921_I00000317_local_dc937fcbdd9a",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S0921_I00000323_local_d1b4c7b0ed42",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000479_local_79a47ef19212",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000479_local_dd7b63f3aabf",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000479_local_e294647d2c81",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000481_local_135b540c056d",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000495_local_37d369c272eb",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1498_I00000015_local_49029f1fd2a7",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1498_I00000018_local_714e74cdf6fd",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1498_I00000019_local_a9a1bb029158",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1731_I00000363_local_0e536e54fb3b",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1731_I00000363_local_6c8c62fce166",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1731_I00000364_local_32855d9c1987",
            "sample_seed": 137
          }
        ],
        "inapplicable_ids": [
          {
            "sample_id": "naturalistic_V03_S0208_I00000479_local_c9ab05da27c1",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S0766_I00000138_local_c74cd9968a0d",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S0921_I00000314_local_a5b50e764dc3",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S0921_I00000314_local_b69cf369a34c",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000480_local_2b8078b726ef",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000480_local_ba61764a5a65",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000484_local_304b29c7cd1b",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000495_local_5fa0cac62f51",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000560_local_10a4cebbda73",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000560_local_5753094a1438",
            "sample_seed": 137
          }
        ],
        "failed_ids": []
      }
    },
    "generated": {
      "full_value": 2.0040690104166665,
      "supported_subset_value": 2.0040690104166665,
      "coverage": {
        "planned": 24,
        "eligible": 24,
        "succeeded": 24,
        "inapplicable": 0,
        "failed": 0,
        "succeeded_ids": [
          {
            "sample_id": "naturalistic_V03_S0208_I00000479_local_c9ab05da27c1",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S0766_I00000125_local_0aadd062235e",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S0766_I00000138_local_c74cd9968a0d",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S0921_I00000314_local_a5b50e764dc3",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S0921_I00000314_local_b69cf369a34c",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S0921_I00000317_local_dc937fcbdd9a",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S0921_I00000323_local_d1b4c7b0ed42",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000479_local_79a47ef19212",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000479_local_dd7b63f3aabf",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000479_local_e294647d2c81",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000480_local_2b8078b726ef",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000480_local_ba61764a5a65",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000481_local_135b540c056d",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000484_local_304b29c7cd1b",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000495_local_37d369c272eb",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000495_local_5fa0cac62f51",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000560_local_10a4cebbda73",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000560_local_5753094a1438",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1498_I00000015_local_49029f1fd2a7",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1498_I00000018_local_714e74cdf6fd",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1498_I00000019_local_a9a1bb029158",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1731_I00000363_local_0e536e54fb3b",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1731_I00000363_local_6c8c62fce166",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1731_I00000364_local_32855d9c1987",
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
      "full_value": null,
      "supported_subset_value": {
        "lse_d": 6.835245166506086,
        "lse_c": 0.9500055994306292
      },
      "coverage": {
        "planned": 24,
        "eligible": 14,
        "succeeded": 14,
        "inapplicable": 10,
        "failed": 0,
        "succeeded_ids": [
          {
            "sample_id": "naturalistic_V03_S0766_I00000125_local_0aadd062235e",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S0921_I00000317_local_dc937fcbdd9a",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S0921_I00000323_local_d1b4c7b0ed42",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000479_local_79a47ef19212",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000479_local_dd7b63f3aabf",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000479_local_e294647d2c81",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000481_local_135b540c056d",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000495_local_37d369c272eb",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1498_I00000015_local_49029f1fd2a7",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1498_I00000018_local_714e74cdf6fd",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1498_I00000019_local_a9a1bb029158",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1731_I00000363_local_0e536e54fb3b",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1731_I00000363_local_6c8c62fce166",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1731_I00000364_local_32855d9c1987",
            "sample_seed": 137
          }
        ],
        "inapplicable_ids": [
          {
            "sample_id": "naturalistic_V03_S0208_I00000479_local_c9ab05da27c1",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S0766_I00000138_local_c74cd9968a0d",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S0921_I00000314_local_a5b50e764dc3",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S0921_I00000314_local_b69cf369a34c",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000480_local_2b8078b726ef",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000480_local_ba61764a5a65",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000484_local_304b29c7cd1b",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000495_local_5fa0cac62f51",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000560_local_10a4cebbda73",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000560_local_5753094a1438",
            "sample_seed": 137
          }
        ],
        "failed_ids": []
      },
      "unit": "equal weight per supported face track",
      "tracks": 14,
      "low_support_tracks": 0,
      "clip_equal_mean": {
        "lse_d": 6.835245166506086,
        "lse_c": 0.9500055994306292
      }
    },
    "generated": {
      "full_value": {
        "lse_d": 8.360301653544107,
        "lse_c": 0.5055761535962423
      },
      "supported_subset_value": {
        "lse_d": 8.360301653544107,
        "lse_c": 0.5055761535962423
      },
      "coverage": {
        "planned": 24,
        "eligible": 24,
        "succeeded": 24,
        "inapplicable": 0,
        "failed": 0,
        "succeeded_ids": [
          {
            "sample_id": "naturalistic_V03_S0208_I00000479_local_c9ab05da27c1",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S0766_I00000125_local_0aadd062235e",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S0766_I00000138_local_c74cd9968a0d",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S0921_I00000314_local_a5b50e764dc3",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S0921_I00000314_local_b69cf369a34c",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S0921_I00000317_local_dc937fcbdd9a",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S0921_I00000323_local_d1b4c7b0ed42",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000479_local_79a47ef19212",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000479_local_dd7b63f3aabf",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000479_local_e294647d2c81",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000480_local_2b8078b726ef",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000480_local_ba61764a5a65",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000481_local_135b540c056d",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000484_local_304b29c7cd1b",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000495_local_37d369c272eb",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000495_local_5fa0cac62f51",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000560_local_10a4cebbda73",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000560_local_5753094a1438",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1498_I00000015_local_49029f1fd2a7",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1498_I00000018_local_714e74cdf6fd",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1498_I00000019_local_a9a1bb029158",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1731_I00000363_local_0e536e54fb3b",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1731_I00000363_local_6c8c62fce166",
            "sample_seed": 137
          },
          {
            "sample_id": "naturalistic_V03_S1731_I00000364_local_32855d9c1987",
            "sample_seed": 137
          }
        ],
        "inapplicable_ids": [],
        "failed_ids": []
      },
      "unit": "equal weight per supported face track",
      "tracks": 24,
      "low_support_tracks": 0,
      "clip_equal_mean": {
        "lse_d": 8.360301653544107,
        "lse_c": 0.5055761535962423
      }
    }
  }
}
```

## Seed 251

```json
{
  "fvd": {
    "full_value": 516.9137548789153,
    "supported_subset_value": 516.9137548789153,
    "coverage": {
      "planned": 24,
      "eligible": 24,
      "succeeded": 24,
      "inapplicable": 0,
      "failed": 0,
      "succeeded_ids": [
        {
          "sample_id": "naturalistic_V03_S0208_I00000479_local_c9ab05da27c1",
          "sample_seed": 251
        },
        {
          "sample_id": "naturalistic_V03_S0766_I00000125_local_0aadd062235e",
          "sample_seed": 251
        },
        {
          "sample_id": "naturalistic_V03_S0766_I00000138_local_c74cd9968a0d",
          "sample_seed": 251
        },
        {
          "sample_id": "naturalistic_V03_S0921_I00000314_local_a5b50e764dc3",
          "sample_seed": 251
        },
        {
          "sample_id": "naturalistic_V03_S0921_I00000314_local_b69cf369a34c",
          "sample_seed": 251
        },
        {
          "sample_id": "naturalistic_V03_S0921_I00000317_local_dc937fcbdd9a",
          "sample_seed": 251
        },
        {
          "sample_id": "naturalistic_V03_S0921_I00000323_local_d1b4c7b0ed42",
          "sample_seed": 251
        },
        {
          "sample_id": "naturalistic_V03_S1493_I00000479_local_79a47ef19212",
          "sample_seed": 251
        },
        {
          "sample_id": "naturalistic_V03_S1493_I00000479_local_dd7b63f3aabf",
          "sample_seed": 251
        },
        {
          "sample_id": "naturalistic_V03_S1493_I00000479_local_e294647d2c81",
          "sample_seed": 251
        },
        {
          "sample_id": "naturalistic_V03_S1493_I00000480_local_2b8078b726ef",
          "sample_seed": 251
        },
        {
          "sample_id": "naturalistic_V03_S1493_I00000480_local_ba61764a5a65",
          "sample_seed": 251
        },
        {
          "sample_id": "naturalistic_V03_S1493_I00000481_local_135b540c056d",
          "sample_seed": 251
        },
        {
          "sample_id": "naturalistic_V03_S1493_I00000484_local_304b29c7cd1b",
          "sample_seed": 251
        },
        {
          "sample_id": "naturalistic_V03_S1493_I00000495_local_37d369c272eb",
          "sample_seed": 251
        },
        {
          "sample_id": "naturalistic_V03_S1493_I00000495_local_5fa0cac62f51",
          "sample_seed": 251
        },
        {
          "sample_id": "naturalistic_V03_S1493_I00000560_local_10a4cebbda73",
          "sample_seed": 251
        },
        {
          "sample_id": "naturalistic_V03_S1493_I00000560_local_5753094a1438",
          "sample_seed": 251
        },
        {
          "sample_id": "naturalistic_V03_S1498_I00000015_local_49029f1fd2a7",
          "sample_seed": 251
        },
        {
          "sample_id": "naturalistic_V03_S1498_I00000018_local_714e74cdf6fd",
          "sample_seed": 251
        },
        {
          "sample_id": "naturalistic_V03_S1498_I00000019_local_a9a1bb029158",
          "sample_seed": 251
        },
        {
          "sample_id": "naturalistic_V03_S1731_I00000363_local_0e536e54fb3b",
          "sample_seed": 251
        },
        {
          "sample_id": "naturalistic_V03_S1731_I00000363_local_6c8c62fce166",
          "sample_seed": 251
        },
        {
          "sample_id": "naturalistic_V03_S1731_I00000364_local_32855d9c1987",
          "sample_seed": 251
        }
      ],
      "inapplicable_ids": [],
      "failed_ids": []
    },
    "reason": null,
    "raw_reference_value": 516.9137548789153,
    "negative_roundoff_clamped": false,
    "roundoff_tolerance": 0.001665779774091496,
    "name": "FVD-32"
  },
  "utmos": {
    "gt": {
      "full_value": null,
      "supported_subset_value": 1.0605817522321428,
      "coverage": {
        "planned": 24,
        "eligible": 14,
        "succeeded": 14,
        "inapplicable": 10,
        "failed": 0,
        "succeeded_ids": [
          {
            "sample_id": "naturalistic_V03_S0766_I00000125_local_0aadd062235e",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S0921_I00000317_local_dc937fcbdd9a",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S0921_I00000323_local_d1b4c7b0ed42",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000479_local_79a47ef19212",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000479_local_dd7b63f3aabf",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000479_local_e294647d2c81",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000481_local_135b540c056d",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000495_local_37d369c272eb",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1498_I00000015_local_49029f1fd2a7",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1498_I00000018_local_714e74cdf6fd",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1498_I00000019_local_a9a1bb029158",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1731_I00000363_local_0e536e54fb3b",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1731_I00000363_local_6c8c62fce166",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1731_I00000364_local_32855d9c1987",
            "sample_seed": 251
          }
        ],
        "inapplicable_ids": [
          {
            "sample_id": "naturalistic_V03_S0208_I00000479_local_c9ab05da27c1",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S0766_I00000138_local_c74cd9968a0d",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S0921_I00000314_local_a5b50e764dc3",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S0921_I00000314_local_b69cf369a34c",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000480_local_2b8078b726ef",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000480_local_ba61764a5a65",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000484_local_304b29c7cd1b",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000495_local_5fa0cac62f51",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000560_local_10a4cebbda73",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000560_local_5753094a1438",
            "sample_seed": 251
          }
        ],
        "failed_ids": []
      }
    },
    "generated": {
      "full_value": 2.1775716145833335,
      "supported_subset_value": 2.1775716145833335,
      "coverage": {
        "planned": 24,
        "eligible": 24,
        "succeeded": 24,
        "inapplicable": 0,
        "failed": 0,
        "succeeded_ids": [
          {
            "sample_id": "naturalistic_V03_S0208_I00000479_local_c9ab05da27c1",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S0766_I00000125_local_0aadd062235e",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S0766_I00000138_local_c74cd9968a0d",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S0921_I00000314_local_a5b50e764dc3",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S0921_I00000314_local_b69cf369a34c",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S0921_I00000317_local_dc937fcbdd9a",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S0921_I00000323_local_d1b4c7b0ed42",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000479_local_79a47ef19212",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000479_local_dd7b63f3aabf",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000479_local_e294647d2c81",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000480_local_2b8078b726ef",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000480_local_ba61764a5a65",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000481_local_135b540c056d",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000484_local_304b29c7cd1b",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000495_local_37d369c272eb",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000495_local_5fa0cac62f51",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000560_local_10a4cebbda73",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000560_local_5753094a1438",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1498_I00000015_local_49029f1fd2a7",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1498_I00000018_local_714e74cdf6fd",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1498_I00000019_local_a9a1bb029158",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1731_I00000363_local_0e536e54fb3b",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1731_I00000363_local_6c8c62fce166",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1731_I00000364_local_32855d9c1987",
            "sample_seed": 251
          }
        ],
        "inapplicable_ids": [],
        "failed_ids": []
      }
    }
  },
  "sync": {
    "gt": {
      "full_value": null,
      "supported_subset_value": {
        "lse_d": 6.835245166506086,
        "lse_c": 0.9500055994306292
      },
      "coverage": {
        "planned": 24,
        "eligible": 14,
        "succeeded": 14,
        "inapplicable": 10,
        "failed": 0,
        "succeeded_ids": [
          {
            "sample_id": "naturalistic_V03_S0766_I00000125_local_0aadd062235e",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S0921_I00000317_local_dc937fcbdd9a",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S0921_I00000323_local_d1b4c7b0ed42",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000479_local_79a47ef19212",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000479_local_dd7b63f3aabf",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000479_local_e294647d2c81",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000481_local_135b540c056d",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000495_local_37d369c272eb",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1498_I00000015_local_49029f1fd2a7",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1498_I00000018_local_714e74cdf6fd",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1498_I00000019_local_a9a1bb029158",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1731_I00000363_local_0e536e54fb3b",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1731_I00000363_local_6c8c62fce166",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1731_I00000364_local_32855d9c1987",
            "sample_seed": 251
          }
        ],
        "inapplicable_ids": [
          {
            "sample_id": "naturalistic_V03_S0208_I00000479_local_c9ab05da27c1",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S0766_I00000138_local_c74cd9968a0d",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S0921_I00000314_local_a5b50e764dc3",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S0921_I00000314_local_b69cf369a34c",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000480_local_2b8078b726ef",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000480_local_ba61764a5a65",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000484_local_304b29c7cd1b",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000495_local_5fa0cac62f51",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000560_local_10a4cebbda73",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000560_local_5753094a1438",
            "sample_seed": 251
          }
        ],
        "failed_ids": []
      },
      "unit": "equal weight per supported face track",
      "tracks": 14,
      "low_support_tracks": 0,
      "clip_equal_mean": {
        "lse_d": 6.835245166506086,
        "lse_c": 0.9500055994306292
      }
    },
    "generated": {
      "full_value": {
        "lse_d": 7.600306302309036,
        "lse_c": 0.605687826871872
      },
      "supported_subset_value": {
        "lse_d": 7.600306302309036,
        "lse_c": 0.605687826871872
      },
      "coverage": {
        "planned": 24,
        "eligible": 24,
        "succeeded": 24,
        "inapplicable": 0,
        "failed": 0,
        "succeeded_ids": [
          {
            "sample_id": "naturalistic_V03_S0208_I00000479_local_c9ab05da27c1",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S0766_I00000125_local_0aadd062235e",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S0766_I00000138_local_c74cd9968a0d",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S0921_I00000314_local_a5b50e764dc3",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S0921_I00000314_local_b69cf369a34c",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S0921_I00000317_local_dc937fcbdd9a",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S0921_I00000323_local_d1b4c7b0ed42",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000479_local_79a47ef19212",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000479_local_dd7b63f3aabf",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000479_local_e294647d2c81",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000480_local_2b8078b726ef",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000480_local_ba61764a5a65",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000481_local_135b540c056d",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000484_local_304b29c7cd1b",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000495_local_37d369c272eb",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000495_local_5fa0cac62f51",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000560_local_10a4cebbda73",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1493_I00000560_local_5753094a1438",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1498_I00000015_local_49029f1fd2a7",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1498_I00000018_local_714e74cdf6fd",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1498_I00000019_local_a9a1bb029158",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1731_I00000363_local_0e536e54fb3b",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1731_I00000363_local_6c8c62fce166",
            "sample_seed": 251
          },
          {
            "sample_id": "naturalistic_V03_S1731_I00000364_local_32855d9c1987",
            "sample_seed": 251
          }
        ],
        "inapplicable_ids": [],
        "failed_ids": []
      },
      "unit": "equal weight per supported face track",
      "tracks": 24,
      "low_support_tracks": 0,
      "clip_equal_mean": {
        "lse_d": 7.600306302309036,
        "lse_c": 0.605687826871872
      }
    }
  }
}
```
