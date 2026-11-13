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
    "full_value": 651.5832093273501,
    "supported_subset_value": 651.5832093273501,
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
    "raw_reference_value": 651.5832093273501,
    "negative_roundoff_clamped": false,
    "roundoff_tolerance": 0.0018968532011941603,
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
      "full_value": 2.1469090779622397,
      "supported_subset_value": 2.1469090779622397,
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
        "lse_d": 9.111768360137939,
        "lse_c": 0.6382275390625
      },
      "supported_subset_value": {
        "lse_d": 9.111768360137939,
        "lse_c": 0.6382275390625
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
      "tracks": 25,
      "low_support_tracks": 0,
      "clip_equal_mean": {
        "lse_d": 9.074608107407888,
        "lse_c": 0.6159089207649231
      }
    }
  }
}
```

## Seed 251

```json
{
  "fvd": {
    "full_value": 563.2036371510985,
    "supported_subset_value": 563.2036371510985,
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
    "raw_reference_value": 563.2036371510985,
    "negative_roundoff_clamped": false,
    "roundoff_tolerance": 0.001739185720282336,
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
      "full_value": 2.2350870768229165,
      "supported_subset_value": 2.2350870768229165,
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
        "lse_d": 8.321465452512106,
        "lse_c": 0.5831427971522013
      },
      "supported_subset_value": {
        "lse_d": 8.321465452512106,
        "lse_c": 0.5831427971522013
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
        "lse_d": 8.321465452512106,
        "lse_c": 0.5831427971522013
      }
    }
  }
}
```
