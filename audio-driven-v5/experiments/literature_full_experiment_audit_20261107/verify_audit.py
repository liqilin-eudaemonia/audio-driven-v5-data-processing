"""Verify inventory/ledger/link consistency only, never research completeness."""
import hashlib
import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent
PROJECT = BASE.parents[1]


def read(name):
    return json.loads((BASE / name).read_text())


def main():
    inventory = read('INVENTORY.json')
    papers = read('PAPER_CATALOG.json')
    occurrences = read('METHOD_OCCURRENCES.json')
    registry = read('METHOD_REGISTRY.json')
    resources = read('RESOURCE_MATRIX.json')
    counts = read('METHOD_COUNTS.json')
    provenance = read('ROW_PROVENANCE.json')
    census = read('TABLE_CENSUS.json')
    actual_pdf = {str(p.relative_to(PROJECT)) for p in (PROJECT / 'writing').rglob('*.pdf')}
    assert actual_pdf == {r['pdf'] for r in inventory}, 'PDF inventory changed'
    for row in inventory:
        digest = hashlib.sha256()
        with (PROJECT / row['pdf']).open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        assert digest.hexdigest() == row['sha256'], row['pdf']
    research = {r['sha256'] for r in inventory if not r['reading_status'].startswith('excluded')}
    assert len(research) == len(papers) == 21
    for rows in [registry, resources]:
        assert len(rows) == len({r['label'] for r in rows})
        assert {r['label'] for r in rows} == set(occurrences)
        assert all(r['resource_status'] != 'pending_per_method_resource_audit' for r in rows)
    for paper in papers:
        assert all(paper[k] for k in ['task_definition', 'inputs', 'outputs', 'training', 'tables', 'caveats'])
        for table in paper['tables'] + paper.get('comparisons_outside_tables', []):
            assert all(table[k] for k in ['dataset', 'methods', 'metrics', 'provenance', 'pdf_pages'])
            for method in table['methods']:
                assert any(o['paper'] == paper['name'] and o['table'] == table['table']
                           for o in occurrences[method])
    expected_rows = [(p['name'], t['table'], m) for p in papers
                     for t in p['tables'] + p.get('comparisons_outside_tables', []) for m in t['methods']]
    actual_rows = [(r['paper'], r['location'], r['method']) for r in provenance]
    assert sorted(expected_rows) == sorted(actual_rows)
    assert all(r['classification'] and r['explanation'] and r['table_context_evidence'] for r in provenance)
    assert len({r['method'] for r in provenance}) == len(resources)
    assert census['reviewed_numbered_table_instances'] == census['supplied_numbered_table_instances'] + census['additional_external_table_instances']
    local_links = 0
    for name in ['REPORT.md', 'RESOURCE_MATRIX.md', 'PROTOCOL_MAP.md', 'EXPERIMENT_RECOMMENDATION.md', 'PAPER_DETAILS.md', 'TABLE_CENSUS.md', 'SOURCE_GUIDE.md', 'COMPLETION_AUDIT.md']:
        for href in re.findall(r'\]\(([^)]+)\)', (BASE / name).read_text()):
            if href.startswith(('https://', 'http://', '#')):
                continue
            path = (BASE / href.split('#', 1)[0]).resolve()
            assert path.exists(), (name, href)
            local_links += 1
    assert counts['raw_labels'] == len(occurrences) == len(resources)
    assert counts['resource_decisions'] == len(resources)
    result = dict(passed_for_scope=True,
                  scope='PDF inventory fingerprints, ledger/resource label coverage and relative file links only',
                  unique_research_pdf_catalog_entries=len(papers),
                  table_groups=sum(len(p['tables']) for p in papers), raw_labels=len(resources),
                  non_table_comparison_groups=sum(len(p.get('comparisons_outside_tables', [])) for p in papers),
                  per_method_provenance_records=len(provenance),
                  reviewed_numbered_table_instances=census['reviewed_numbered_table_instances'],
                  all_current_inventory_fingerprints_match=True,
                  source_files_changed_since_initial_inventory=[r['pdf'] for r in inventory if r.get('previous_versions')],
                  resource_label_coverage_complete=True,
                  local_links_checked=local_links,
                  does_not_verify=['fullpaper reading completeness', 'all external supplementary tables',
                                   'semantic correctness of every provenance claim', 'checkpoint payload validity',
                                   'original published protocol comparability'], structural_check_alone_proves_full_goal=False)
    (BASE / 'COVERAGE_CHECK.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
