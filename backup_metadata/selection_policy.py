import pathlib, re

CODE = set('.py .pyi .sh .bash .zsh .fish .js .jsx .ts .tsx .c .cc .cpp .h .hpp .cu .cuh .rs .go .java .r .R .sql .ipynb .patch .diff .bat .cmd .ps1 .mk .m .mm .lua .kt .swift .cs .gradle .groovy .pl .rb .php .jl .f90 .f95'.lower().split())
CODE.update('.pyx .pxd .pxi .cxx .hxx .hh .slurm .sbatch .srun .proto .in .m4 .ac .am .asm .s .cl .glsl .vert .frag .vue .svelte .def'.split())
DOC = set('.md .rst .txt .tex .bib .pdf .doc .docx .ppt .pptx .odt .html .htm .css .scss .svg'.split())
DOC.update('.sty .cls .bst .bbl .bbx .cbx .typ .rmd .qmd .org .adoc .rtf .xls .xlsx .ods .odp .pages .numbers .eps .ps .drawio .vsdx .latex .ltx .dtx .ins'.split())
CONFIG = set('.yaml .yml .toml .ini .cfg .conf .xml .properties .service .desktop .cmake'.split())
CONFIG.update('.json5 .jsonc .j2 .jinja .jinja2 .tmpl .tpl .props .csproj .sln .vcxproj .filters .pro .pri .qrc'.split())
ASSETS = set('.png .jpg .jpeg .gif .webp .ico .woff .woff2 .ttf .otf'.split())
SKIP = {'.git', '__pycache__', '.pytest_cache', 'node_modules', '.mypy_cache', '.ruff_cache', '.eval_deps', '.eval_models', '.envs'}
VENV_ROOTS = {
    'audio-driven-v5/experiments/listen_j_omni_20261030/nod_cpu_share/runtime_env',
    'audio-driven-v5/experiments/listen_j_omni_20261030/visual_quality_rollout/runtime_env',
    'audio-driven-v5/experiments/listen_j_reaction_vocal_omni_20261103/reaction_runtime_env',
}
DOCUMENT_DIRS = {'writing', 'docs', 'feishu_docs', 'related', 'releases'}
SOURCE_DIRS = DOCUMENT_DIRS | {'src','scripts','tests','configs','examples','external_sources'}
BULK = {'cache','caches','media','audio','video','videos','normalized_audio','normalized_pairs','audit','summaries','results','render_results','snapshots','features','latents','locks','claims','pairs','samples','events','logs','tensorboard'}
LIST_NAME = re.compile(r'(manifest|(?:^|[_\-.])(?:index|indices|split|splits|catalog|inventory|dataset|selection|selected|train|training|val|valid|validation|test|pairs|metadata|meta|contract)(?:$|[_\-.]))', re.I)
SAMPLE_DIR = re.compile(r'^(?:naturalistic|improvised)_V\d+_S\d+_I\d+(?:_|$)')
DATA_CACHE_DIRS = {'cache','caches','semantic_cache','media','audio','video','videos','normalized_audio','normalized_pairs','features','latents','detections','references','locks','claims','frames','tensorboard'}

def excluded_directory(parts):
    joined='/'.join(parts)
    for root in VENV_ROOTS:
        if joined==root or joined.startswith(root+'/'):return root.count('/'),'environment_cache_or_git_internal'
    for i,p in enumerate(parts):
        if p in SKIP or p.startswith('.venv'): return i,'environment_cache_or_git_internal'
        if SAMPLE_DIR.match(p): return i,'per_sample_generated_data_directory'
        if 'captions' in parts[:i] and re.fullmatch(r'[0-9a-f]{16,64}',p) and any(re.fullmatch(r'[0-9a-f]{16,64}',s) for s in parts[parts.index('captions')+1:i]): return i,'per_sample_generated_caption_directory'
        if i>=3 and len(parts)>1 and parts[1] in {'experiments','data_corpora','data_previews','evaluation','compare'} and not any(s in {'src','scripts','source','sources','runtime','code','frozen_source','configs'} for s in parts[2:i]):
            if p in DATA_CACHE_DIRS or p.endswith('_cache'): return i,'generated_cache_media_or_feature_directory'
    return None,None

def select(row):
    parts=row['path'].split('/'); name=parts[-1]; ext=('.'+name.rsplit('.',1)[-1].lower()) if '.' in name else ''
    for compression in ('.gz','.bz2','.xz','.zst'):
        if name.lower().endswith(compression):
            inner=name[:-len(compression)].rsplit('.',1)[-1].lower()
            if inner in {'json','jsonl','csv','tsv','ndjson'}:ext='.'+inner
    excluded_index,excluded_reason=excluded_directory(parts[:-1])
    if excluded_index is not None:return False,excluded_reason
    if any(x in SKIP or x.startswith('.venv') for x in parts[1:]): return False,'environment_cache_or_git_internal'
    if name == '.env' or name.startswith('.env.') or name in {'id_rsa','id_ed25519','id_github','credentials','credentials.json'} or ext in {'.pem','.key','.p12','.pfx'}: return False,'credential_file'
    if row['type']=='dir': return False,'directory'
    if row['type']=='symlink': return False,'symlink_recorded_without_target_copy'
    if row['type']!='file': return False,'special_file'
    if ('naturalistic_' in row['path'] or 'improvised_' in row['path']) and any(SAMPLE_DIR.match(part) for part in parts[1:-1]): return False,'per_sample_generated_data_directory'
    if ext in {'.json','.jsonl','.csv','.tsv','.ndjson','.parquet','.arrow'} and row['size'] > 100_000_000: return False,'oversized_manifest_over_100MB_user_approved'
    top=parts[1] if len(parts)>2 else ''
    if ext in CODE|CONFIG or name.lower() in {'dockerfile','makefile','cmakelists.txt','license','notice','copying','.gitignore','.gitattributes','.gitmodules','requirements','procfile','uv.lock','poetry.lock','pdm.lock','.editorconfig','.coveragerc','.flake8','.python-version','.dockerignore','.latexmkrc','configure','compile','install-sh','config.guess','config.sub'}: return True,'code_or_configuration'
    if top in DOCUMENT_DIRS and name.lower().endswith(('.zip','.tar.gz','.tgz','.tar.xz')) and any(w in name.lower() for w in ('source','code','author-kit','cvpr','neurips','arxiv','arxiv')): return True,'document_or_source_archive'
    if not ext and top in SOURCE_DIRS and row['size'] <= 1024**2: return True,'source_directory_extensionless_file'
    if ext in DOC:
        if ext == '.txt' and any(x in BULK for x in parts[2:-1]) and not any(t in name.lower() for t in ['readme','requirements','license']): return False,'generated_text_output'
        return True,'documentation'
    if ext in ASSETS and (top in DOCUMENT_DIRS or 'figures' in parts or 'assets' in parts): return True,'documentation_asset'
    if ext in {'.json','.jsonl','.csv','.tsv','.ndjson','.parquet','.arrow'}:
        if parts[0]=='audio-driven-v5-data-processing': return True,'existing_backup_metadata'
        if top in SOURCE_DIRS or top=='data_reviews': return True,'source_or_review_metadata'
        if 'configs' in parts or re.search(r'(config|args|hparams|settings|parameters|params|spec|schema|contract|prompt|recipe)',name,re.I): return True,'configuration'
        if any(x in BULK for x in parts[2:-1]): return False,'generated_per_sample_metadata'
        if len(parts)<=4 or LIST_NAME.search(name) or any(x in {'manifests','indices','indexes','splits'} for x in parts): return True,'dataset_manifest_or_configuration'
        if name.lower() in {'config.json','args.json','hparams.json','run_config.json','settings.json','tokenizer_config.json','generation_config.json','preprocessor_config.json'}: return True,'configuration'
        return False,'generated_metadata'
    return False,'large_data_binary_or_generated_output'
