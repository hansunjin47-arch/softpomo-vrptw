"""Offline prompt revision. Preserve v5 evidence/schema; never call an API."""
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'ontology_v5_action_semantics_v1'
SOURCE = ROOT/'result/rc101_ontology_v5_train20_b128_seed43/requests'
OUT = ROOT/'result/rc101_ontology_v5_action_semantics_v1_train20'

INSTRUCTIONS = '''Assign static action-preference scores to reference-group customers and depot 0.
Minimize J=0.04*K+T/5760+0.0075*Lt: K is vehicles, T is driving minutes including returns,
and Lt is total customer and depot lateness. This is a weighted sum, not lateness-first.
Capacity is hard, closing times are soft, and all customers must be served; unserved demand has a large penalty.

SCORE MEANING AND EXACT DECODER CONTRACT
A score in [0,1] specifies a static selection preference, not on-time probability, confidence in a diagnosis,
or suitability for assignment to a vehicle. Higher effective bias favors choosing that action now whenever active.
RL still decides using current position, time, load and remaining customers. You receive no live rollout state.
The actual first customer's reference group stays active until depot return. There is no bias before that first choice.
Other-group customers remain feasible but have zero bias; they do not carry their original group's score into this route.
For active group m, let d be its depot score and c_i a member customer's score. With beta=5:
  depot: b_0 = 5*d;
  member customer with c_i < d: b_i = -5;
  member customer with c_i >= d: b_i = 5*c_i;
  nonmember customer: b_i = 0.
The strict comparison matters: equality does not trigger the negative transformation.
For example d=0.30,c_i=0.29 gives -5; d=0.30,c_i=0.30 gives +1.5.
Among feasible actions, final logits are base logits plus b_i; odds are multiplied by exp(b_i-b_j).
Thus raw-score odds exp(5*(c_i-c_j)) are NOT generally valid across the negative threshold.
Masks still enforce visited/capacity constraints. A negative bias is finite, not a hard exclusion.
Depot preference is soft, but raising its score also suppresses EVERY member whose score falls below it.
Do not treat the depot score as an independent probability or a harmless cutoff.
All-zero customer AND depot scores disable additive guidance. Equal positive member scores still favor members
over outsiders. A score of 0.5 is not neutral. A zero customer score below a positive depot score becomes -5.
Rain bias is available after route start. Accident bias activates only after the current vehicle clock reaches
disclosure, and remains available afterwards. Each new vehicle starts at time zero.

INTERPRETING REFERENCE EVIDENCE
Use exposure, inherited delay, waiting absorption, deadline margins and existing versus added lateness jointly.
Do not lower a score merely because the reference visit is late, risky, or appears unsuitable for its group:
lower preference can postpone service; it does NOT request reassignment or guarantee another vehicle will serve it earlier.
Conversely, lateness or exposure does NOT mandate a higher score: earlier selection can harm other customers,
change exposure or increase vehicle use. There is no universal mapping from a diagnosis to a score.
The reference route is evidence, not a target that must be reconstructed. Preserving its group or order is not itself a benefit.
Distinguish evidence about a visit on this reference schedule from a customer-wide priority valid under changed schedules.
Static scores cannot encode time-conditional actions, select a receiving route, or guarantee a repair.
When evidence does not support a direction, avoid inventing a strong preference. Consider both within-group
score differences and the effective comparison with zero-biased outsiders and depot, including the negative threshold.
Reference slack alone does not certify a safe move. Waiting absorption holds at that reference visit only.
No alternative-route search or measured edit improvement is supplied; do not claim one.
Reference schedules are exact offline simulator feedback rounded for display, not public-only observations.
Travel ratios can reveal hidden multipliers; excluding the multiplier field does not remove that information.

OUTPUT
Return ONLY the supplied confidence JSON schema with {cluster,customer,score} entries.
Use one-based cluster IDs, every listed group customer and depot 0 exactly once per group.
No explanations, new assignments, route edits, or extra pairs.
'''


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    paths = sorted(SOURCE.glob('*/request.json'))
    assert len(paths)==20
    prepared, provenance = {}, {}
    for path in paths:
        raw = path.read_bytes()
        old = json.loads(raw)
        assert len(old['messages'])==1
        content = old['messages'][0]['content']
        payload_text = content[content.index('{"ontology":'):]
        json.loads(payload_text)
        new = copy.deepcopy(old)
        new['messages'][0]['content'] = INSTRUCTIONS+'\n'+payload_text
        assert new['response_format']==old['response_format']
        assert new['model']==old['model']
        assert new['messages'][0]['content'].endswith(payload_text)
        name = path.parent.name
        prepared[name] = new
        provenance[name] = dict(source=str(path),source_sha256=digest(raw),
            payload_sha256=digest(payload_text.encode()),
            old_characters=len(content),new_characters=len(new['messages'][0]['content']))
    OUT.mkdir(exist_ok=False)
    for name, request in prepared.items():
        folder=OUT/'requests'/name
        folder.mkdir(parents=True)
        dest=folder/'request.json'
        dest.write_text(json.dumps(request,indent=2),encoding='utf-8')
        provenance[name]['output_sha256']=digest(dest.read_bytes())
        assert digest(Path(provenance[name]['source']).read_bytes())==provenance[name]['source_sha256']
    (OUT/'instructions.txt').write_text(INSTRUCTIONS,encoding='utf-8')
    (OUT/'protocol.json').write_text(json.dumps(dict(version=VERSION,status='draft_prepared_not_called',
        api_calls=0,training_started=False,ontology_schema='unchanged v5',
        changed='Instruction text only; same facts, model, output schema and reference routes.',
        source_script_sha256=digest(Path(__file__).read_bytes()),requests=provenance),indent=2),encoding='utf-8')
    print(f'PASS: {len(prepared)} draft requests; exact payload/schema preservation; source hashes unchanged; API calls=0')


if __name__=='__main__':
    main()
