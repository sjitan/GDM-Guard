import os, sys, json
root=os.path.dirname(os.path.abspath(__file__)); root=os.path.abspath(os.path.join(root,".."))
if root not in sys.path: sys.path.insert(0, root)
from agent.m_stack import build
if __name__=="__main__":
    intake="data/intake_demo.json"
    selfie="sessions/vis_metrics.json"
    r=build(intake, selfie)
    print(json.dumps(r,indent=2))
