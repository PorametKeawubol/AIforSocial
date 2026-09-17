"""Evidence-first police manual graph. No generative model calls."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parent
TERMS = {
    'DocumentType': ['บัตรประจำตัวประชาชน', 'สำเนาทะเบียนบ้าน', 'หนังสือเดินทาง',
                     'หนังสือมอบอำนาจ', 'หนังสือรับรองบริษัท', 'ใบทะเบียนรถยนต์',
                     'ใบเสร็จรับเงิน', 'สัญญาซื้อขาย', 'ภาพถ่ายคนหาย'],
    'Organization': ['สำนักงานตำรวจแห่งชาติ', 'สถานีตำรวจ', 'สำนักงานทะเบียน',
                     'มูลนิธิกระจกเงา', 'กรมการขนส่งทางบก'],
    'PersonRole': ['พนักงานสอบสวน', 'ผู้ต้องหา', 'ผู้เสียหาย', 'ผู้ถูกจับ'],
    'Topic': ['แจ้งความคนหาย', 'แจ้งความรถหรือเรือหาย', 'การประกันตัว',
              'อาวุธปืน', 'การป้องกันอาชญากรรม'],
}
LABELS = {'Document', 'Page', 'Section', 'Passage', 'Entity', 'OntologyClass'}
RELATIONS = {'HAS_PAGE', 'HAS_SECTION', 'HAS_PASSAGE', 'ON_PAGE', 'MENTIONS',
             'INSTANCE_OF', 'HAS_GUIDANCE', 'CONTACT_VIA'}

def uid(*parts):
    return hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()

def clean(text):
    # A known extraction artifact; preserve raw text separately for audit.
    return re.sub(r'[ \t]+', ' ', text.replace('�ำ', 'ำ')).strip()

def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def build(pdf, out):
    out.mkdir(parents=True, exist_ok=True)
    raw = subprocess.check_output(['pdftotext', '-layout', str(pdf), '-']).decode('utf-8')
    pages = raw.split('\f')
    if not pages[-1].strip():
        pages.pop()
    docid = uid(pdf.read_bytes().hex())
    nodes, edges, page_rows = {}, {}, []
    def node(label, key, **props):
        ident = uid(docid, label, key)
        nodes[ident] = {'id': ident, 'label': label, 'props': {'document_id': docid, **props}}
        return ident
    def edge(source, kind, target, **props):
        ident = uid(source, kind, target)
        edges[ident] = {'id': ident, 'source': source, 'type': kind, 'target': target, 'props': props}
    doc = node('Document', docid, name=pdf.name, sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),
               edition='กรกฎาคม 2558', scope='Historical source; not verified current guidance')
    for typ in TERMS:
        node('OntologyClass', typ, name=typ)
    current = None
    last_section = (1, 0)
    body = False
    for number, original in enumerate(pages, 1):
        text = clean(original)
        printed = re.search(r'(?:^|\n)\s*(\d+)\s+คู่มือประชาชน|กลุ่มงานการมีส่วนร่วมของประชาชน\s+(\d+)\s*$', text)
        printed = next((x for x in printed.groups() if x), '') if printed else ''
        page = node('Page', number, name=f'PDF {number}', pdf_page=number,
                    printed_page=printed, raw_text=original, text=text,
                    needs_text_review=('�' in text or any('\uf700' <= c <= '\uf7ff' for c in text)))
        edge(doc, 'HAS_PAGE', page)
        page_rows.append({'pdf_page': number, 'printed_page': printed, 'raw_text': original, 'text': text})
        # Detect the body from printed page numbering, excluding the table of contents.
        if printed == '1':
            body = True
        if not body:
            continue
        lines = text.splitlines()
        groups, buffer = [], []
        for line in lines:
            line = line.strip()
            if not line or 'กลุ่มงานการมีส่วนร่วมของประชาชน' in line or re.match(r'^\d*\s*คู่มือประชาชน', line):
                continue
            heading = re.match(r'^([1-7])\.(\d+)\s', line) if printed and int(printed) >= 8 else None
            sequence = tuple(map(int, heading.groups())) if heading else None
            is_next = sequence and (sequence == (last_section[0], last_section[1] + 1)
                                    or sequence == (last_section[0] + 1, 1))
            if is_next:
                if buffer:
                    groups.append((current, '\n'.join(buffer)))
                    buffer = []
                section_number = line.split()[0]
                last_section = sequence
                current = node('Section', section_number, name=line, section_number=section_number,
                               title_needs_review=True)
                edge(doc, 'HAS_SECTION', current)
            if re.match(r'^\(?\d+\)|^- ', line) and buffer:
                groups.append((current, '\n'.join(buffer)))
                buffer = []
            buffer.append(line)
        if buffer:
            groups.append((current, '\n'.join(buffer)))
        for idx, (section, evidence) in enumerate(groups):
            passage = node('Passage', f'{number}:{idx}', name=evidence[:85], text=evidence,
                           pdf_page=number, printed_page=printed, extraction='layout_rules_v1',
                           needs_review=True)
            edge(page, 'HAS_PASSAGE', passage)
            if section:
                edge(section, 'HAS_PASSAGE', passage)
                edge(section, 'ON_PAGE', page)
            for typ, terms in TERMS.items():
                for term in terms:
                    if term in evidence:
                        entity = node('Entity', f'{typ}:{term}', name=term, entity_type=typ)
                        edge(entity, 'INSTANCE_OF', uid(docid, 'OntologyClass', typ))
                        edge(passage, 'MENTIONS', entity, evidence=evidence, pdf_page=number,
                             method='exact_dictionary', status='mention_only')
            # Narrow source-specific rules. Preserve modality in full guidance text.
            section_num = nodes[section]['props']['section_number'] if section else ''
            if section_num == '1.4' and 'ไม่จำเป็นต้องรอเวลาให้ครบ 24 ชั่วโมง' in evidence:
                edge(section, 'HAS_GUIDANCE', passage, evidence=evidence, pdf_page=number,
                     method='missing_person_wait_rule', status='needs_review')
            if section_num == '1.5' and 'แจ้งทาง 191 หรือ แจ้ง 1192' in evidence:
                for phone in ['191', '1192']:
                    entity = node('Entity', f'Contact:{phone}', name=phone, entity_type='Contact')
                    edge(section, 'CONTACT_VIA', entity, evidence=evidence, pdf_page=number,
                         alternative_group='191_or_1192', method='vehicle_contact_rule', status='needs_review')
    graph = {'document_id': docid, 'nodes': list(nodes.values()), 'edges': list(edges.values())}
    validate(graph)
    dump(out / 'pages.json', page_rows)
    dump(out / 'graph.json', graph)
    report = {'pages': len(pages), 'nodes': dict(Counter(n['label'] for n in nodes.values())),
              'relationships': dict(Counter(e['type'] for e in edges.values())),
              'pages_needing_text_review': [n['props']['pdf_page'] for n in nodes.values()
                    if n['label'] == 'Page' and n['props']['needs_text_review']],
              'embedding_run': False, 'neo4j_imported': False,
              'limitations': ['Dictionary coverage is partial', 'Passages may continue across pages',
                 'Titles may wrap across lines', 'Semantic rules require review; no current legal verification']}
    dump(out / 'validation.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))

def validate(graph):
    ids = {n['id'] for n in graph['nodes']}
    assert len(ids) == len(graph['nodes']), 'Duplicate node IDs'
    assert len({e['id'] for e in graph['edges']}) == len(graph['edges']), 'Duplicate edges'
    for n in graph['nodes']:
        assert n['label'] in LABELS
    for e in graph['edges']:
        assert e['source'] in ids and e['target'] in ids
        assert e['type'] in RELATIONS
        if e['type'] in {'MENTIONS', 'HAS_GUIDANCE', 'CONTACT_VIA'}:
            assert e['props']['evidence'] and e['props']['pdf_page'] > 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf', type=Path, default=ROOT.parent / 'คู่มือประชาชน สำนักงานตำรวจแห่งชาติ.pdf')
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts')
    args = parser.parse_args()
    build(args.pdf, args.out)
