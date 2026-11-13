"""Apply the documented resource budget to the unchanged separation helper."""
import math
import os
from pathlib import Path
import sys

import parallel as source

WORK = Path(__file__).resolve().parent


def main():
    name = sys.argv[1]
    assert name in [f'worker_{i}' for i in range(4)]
    protocol = source.read(WORK/'RESOURCE_SCHEDULER_V2_PROTOCOL.json')
    assert protocol['resource_executor_sha256'] == source.sha(__file__)
    module = source.helper()
    module.WORK = WORK/'execution'/name
    original_guard = module.guard

    def budget_guard(fraction, original_floor):
        assert fraction == .05 and original_floor == 32768
        _, total, _ = source.gpu()
        required = math.ceil(total*fraction) + 1024 + 16384
        result = original_guard(fraction, required)
        result.update(resource_executor_sha256=source.sha(__file__),
            scheduler_protocol_sha256=source.sha(WORK/'RESOURCE_SCHEDULER_V2_PROTOCOL.json'),
            previous_redundant_floor_mib=original_floor, context_budget_mib=1024)
        return result

    module.guard = budget_guard
    module.run('all')


if __name__ == '__main__':
    main()
