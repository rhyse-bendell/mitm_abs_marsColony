import sys, tempfile, random, json
sys.path.insert(0, r"D:\Post-doc Work\mitm_abs_marsColony")
from modules.simulation import SimulationState
from modules.agent import Agent
from modules.brain_provider import RuleBrain
from collections import Counter
seed=int(sys.argv[1]); dur=int(sys.argv[2]); verbose=len(sys.argv)>3
cap={}
o1=RuleBrain._compute_mode_scores
def cm(self, features, legal_types, control_state, mp):
    r=o1(self, features, legal_types, control_state, mp)
    cap["modes"]={k:round(v,2) for k,v in r[0].items() if k in ("VALIDATE","REPAIR","ACQUIRE_DIK","CONSTRUCT","LOGISTICS")}
    cap["feat"]={k:round(features.get(k,0),2) for k in ("observed_repair_need","repair_executable","repair_knowledge_gap","repair_pressure","validation_work_pending")}
    return r
RuleBrain._compute_mode_scores=cm
o2=RuleBrain._score_actions_for_mode
def sc(self, *, sorted_affordances, selected_mode, features, mechanism_profile):
    r=o2(self, sorted_affordances=sorted_affordances, selected_mode=selected_mode, features=features, mechanism_profile=mechanism_profile)
    cap["mode"]=selected_mode
    cap["aff"]=[(a.get("action_type")[:3], a.get("target_id")[-14:], round(r[f"{i}:{a.get('action_type')}"],2), a.get("repair_ready"), a.get("validation_failed")) for i,a in enumerate(sorted_affordances) if a.get("action_type") in ("validate_construction","repair_or_correct_construction")]
    return r
RuleBrain._score_actions_for_mode=sc
rows=[]
o3=Agent._select_decision_through_pilot_adapter
def sel(self, sim_state, context, trigger_reason, runtime=None):
    cap.clear()
    d=o3(self, sim_state, context, trigger_reason, runtime)
    rows.append(dict(t=round(sim_state.time,1), agent=self.name, action=d.selected_action.value, target=d.target_id, **{k:v for k,v in cap.items()}))
    return d
Agent._select_decision_through_pilot_adapter=sel
random.seed(seed)
with tempfile.TemporaryDirectory() as tmp:
    st=SimulationState(phases=None, project_root=tmp, flash_mode=True, experiment_name="dg", seed=seed, experimental_settings={"constructs":{"teamwork_potential":0.5,"taskwork_potential":0.5},"profile_mode":None} if False else {"constructs":{"teamwork_potential":0.5,"taskwork_potential":0.5}})
    ev=[]
    names={"construction_build_error_introduced","construction_validation_attempted","construction_validated_incorrect","construction_validated_correct","construction_repair_episode","construction_repair_blocked"}
    st.logger.register_event_listener(lambda e: ev.append((round(st.time,1), e["event_type"], e["payload_data"].get("project_id"), e["payload_data"].get("agent"), e["payload_data"].get("cause"))) if e["event_type"] in names else None)
    snap={}
    for i in range(dur):
        st.update(1.0)
    c=Counter(e[1] for e in ev); print("EVENTS", dict(c))
    first=next((e for e in ev if e[1]=="construction_validated_incorrect"), None)
    print("first failed validation", first)
    if first:
        pid=first[2]
        print("PROJECT TIMELINE", pid)
        for e in ev:
            if e[2]==pid: print("  ", e)
        t0=first[0]
        print("DECISIONS after first failure (all agents), first 14 involving validate/repair/inspect/consult:")
        n=0
        for r in rows:
            if r["t"]>=t0 and r["action"] in ("validate_construction","repair_or_correct_construction","inspect_information_source","consult_team_artifact"):
                print("  ", r["t"], r["agent"], r["action"], (r["target"] or "")[-16:], "mode",r.get("mode"), r.get("modes"), r.get("feat"), r.get("aff")); n+=1
                if n>=14: break
    acts=Counter((r["agent"],r["action"],r.get("mode")) for r in rows if first and r["t"]>=first[0])
    print("ACTION COUNTS after first failure:", sorted(acts.items(), key=lambda kv:-kv[1])[:14])
    ready=[r for r in rows if any(a[3] is True for a in r.get("aff",[]))]
    print("DECISIONS with repair_ready:", len(ready), Counter((r["action"],r.get("mode")) for r in ready).most_common(6))
    for r in ready[:4]: print("   READY", r["t"], r["agent"], r["action"], r.get("mode"), r.get("modes"), r.get("aff"))
    print("BAD LEFT:", [(p, st.environment.construction.observable_status(q), q.get("correct")) for p,q in st.environment.construction.projects.items() if q.get("correct",True) is False][:8])
    tot=Counter()
    for a in st.agents:
        for k in ("validation_attempts","validation_failures","revalidations","repair_selections","repair_decision_episodes","repair_applications","repair_successes","repair_blocked"): tot[k]+=a.experimental_metrics[k]
    print("TOTALS", dict(tot), "validated", sum(1 for q in st.environment.construction.projects.values() if q.get("validated_complete")))
