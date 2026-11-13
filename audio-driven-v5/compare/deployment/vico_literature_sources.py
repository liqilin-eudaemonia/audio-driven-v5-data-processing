"""Independently verify every cited numeric row in official publication PDFs."""
import re
import sys

import pymupdf

from vico_literature_protocol import WORK,read,save,sha


def write_bibliography():
    folder=WORK/'paper_tables';folder.mkdir(exist_ok=True)
    text='''% Result-source references verified against official proceedings pages.
% Also cite each baseline's original method paper when discussing that method.
@inproceedings{tran2024dim,
  author = {Tran, Minh and Chang, Di and Siniukov, Maksim and Soleymani, Mohammad},
  title = {{DIM}: Dyadic Interaction Modeling for Social Behavior Generation},
  booktitle = {Computer Vision -- ECCV 2024},
  year = {2024},
  pages = {484--503},
  doi = {10.1007/978-3-031-72913-3_27},
  url = {https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/5491_ECCV_2024_paper.php}
}
@inproceedings{zhu2025infp,
  author = {Zhu, Yongming and Zhang, Longhao and Rong, Zhengkun and Hu, Tianshu and Liang, Shuang and Ge, Zhipeng},
  title = {{INFP}: Audio-Driven Interactive Head Generation in Dyadic Conversations},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year = {2025},
  pages = {10667--10677},
  url = {https://openaccess.thecvf.com/content/CVPR2025/html/Zhu_INFP_Audio-Driven_Interactive_Head_Generation_in_Dyadic_Conversations_CVPR_2025_paper.html}
}
@inproceedings{ki2026avatarforcing,
  author = {Ki, Taekyung and Jang, Sangwon and Jo, Jaehyeong and Yoon, Jaehong and Hwang, Sung Ju},
  title = {Avatar Forcing: Real-Time Interactive Head Avatar Generation for Natural Conversation},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year = {2026},
  pages = {18074--18084},
  url = {https://openaccess.thecvf.com/content/CVPR2026/html/Ki_Avatar_Forcing_Real-Time_Interactive_Head_Avatar_Generation_for_Natural_Conversation_CVPR_2026_paper.html}
}
'''
    (folder/'result_sources.bib').write_text(text)


def verify_sources():
    sources={
        'DIM Table1':('dim_eccv2024_published',10,['FD','P_FD','MSE','SID','Var','rPCC']),
        'INFP Table3':('infp_cvpr2025_published',8,['FD','MSE','SID','Var']),
        'AvatarForcing Table4':('avatarforcing_cvpr2026_published',6,['FD','rPCC','SID','Var'])}
    versions=read(WORK/'web/PUBLISHED_PDFS.json')
    rows=read(WORK/'LITERATURE_ROWS.json')['rows'];cells=[];proofs=[]
    for r in rows:
        key,page,order=sources[r['original_result_source']];meta=versions[key]
        assert sha(meta['path'])==meta['sha256']
        d=pymupdf.open(meta['path']);pdf_page=d[page-1]
        # Avatar Forcing Table3 and Table4 are side by side. Extract the right
        # column so the HDTF table's numbers cannot contaminate the ViCo row.
        clip=pymupdf.Rect(pdf_page.rect.width/2,0,pdf_page.rect.width,pdf_page.rect.height) if key.startswith('avatarforcing') else None
        text=pdf_page.get_text(sort=True,clip=clip)
        expected=[r['metrics'][k][g] for k in order for g in ['exp','pose'] if r['metrics'][k][g] is not None]
        matches=[]
        for line in text.splitlines():
            values=[float(x) for x in re.findall(r'(?<![\d.])\d+\.\d+(?![\d.])',line)]
            if values==expected:matches.append(line.strip())
        assert len(matches)==1,(r['method'],expected,matches)
        proofs.append(dict(method=r['method'],original_result_source=r['original_result_source'],
            published_pdf=meta['path'],published_pdf_sha256=meta['sha256'],pdf_page=page,
            exact_numeric_row_matched=True,matched_row=matches[0]))
        for metric,groups in r['metrics'].items():
            for group,value in groups.items():
                cells.append(dict(method=r['method'],metric=metric,group=group,value=value,
                    reported=value is not None,result_origin=r['original_result_source'] if value is not None else None,
                    paper_url=meta['url'] if value is not None else None,pdf_page=page if value is not None else None))
    save(WORK/'LITERATURE_CELL_PROVENANCE.json',dict(passed=True,rows=len(rows),
        reported_cells=sum(c['reported'] for c in cells),cells=cells,row_proofs=proofs,
        meaning='Complete numeric rows independently matched against official publication PDFs; source column order preserved',
        version_note='DIM official ECCV Table1 is PDF page10; local preprint Table1 is PDF page11. Cited values match.'))
    return len(proofs)


if __name__=='__main__':
    print('Verified published numeric rows:',verify_sources());write_bibliography()
