"""Small cross-engine fixture, comparing observable mechanics under Gen5 PRNG.

Failure reports a mismatch rather than allowing silent simulator substitution.
"""
import json
from pathlib import Path
import subprocess
import time
import pikapro
root=Path(__file__).resolve().parents[1]
out=root/'runs'/f'conformance-{time.time_ns()}';out.mkdir(parents=True)
reports=[]
for name in ('balance','native-sample'):
    team=(root/'teams'/f'{name}.txt').read_text()
    b=pikapro.Battle('championsvgc',[team,team],seed=[1,2,3,4]);choices=[]
    for _ in range(1000):
        if b.over:break
        pending=[(s,b.legal_actions(s)[0]) for s in range(2) if b.request_kind(s)!='wait']
        for side,index in pending:
            text=str(b.legal_choices(side)[0])
            # Choice's repr wraps its Showdown choice text.
            text=text.removeprefix('Choice("').removesuffix('")')
            choices.append([side,text]);b.submit(side,index)
    if not b.over:raise RuntimeError('Fixture did not finish')
    source=out/f'{name}-input.json';target=out/f'{name}-showdown.json'
    source.write_text(json.dumps({'team':team,'choices':choices}))
    subprocess.run(['node','scripts/check_native.cjs',str(source),str(target)],cwd=root,check=True)
    # Compare mechanics-bearing protocol events; UI and timestamp messages differ.
    kinds={'switch','move','-damage','-heal','faint','-status','-boost','-unboost','-weather','win','tie'}
    def select(lines):return [x for x in lines if len(x.split('|'))>1 and x.split('|')[1] in kinds]
    # Reference log includes split/private lines; spectator perspective avoids exact HP.
    log=json.loads(target.read_text())['log'];public=[];i=0
    while i<len(log):
        if log[i].startswith('|split|'):
            public.append(log[i+2]);i+=3
        else:public.append(log[i]);i+=1
    actual=select(b.log('spectator'));expected=select(public)
    (out/f'{name}-native.json').write_text(json.dumps(actual,indent=2))
    same=actual==expected
    reports.append({'team':name,'matching':same,'native_events':len(actual),'reference_events':len(expected)})
(out/'report.json').write_text(json.dumps(reports,indent=2));print(out);print(json.dumps(reports,indent=2))

if not all(r['matching'] for r in reports):
    raise SystemExit('Native/reference mismatch; inspect saved fixtures')
