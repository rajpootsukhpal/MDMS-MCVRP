import numpy as np
import random
from enum import Enum

random.seed(1)
np.random.seed(1)

# =========================
# SEASONS
# =========================
class Season(Enum):
    SPRING=1; SUMMER=2; AUTUMN=3; WINTER=4

def get_season(t,T):
    r=t/T
    if r<=0.25: return Season.SPRING
    if r<=0.55: return Season.SUMMER
    if r<=0.8:  return Season.AUTUMN
    return Season.WINTER

ANTS={
Season.SPRING:40,
Season.SUMMER:25,
Season.AUTUMN:15,
Season.WINTER:8,
}

# =========================
# ACO PARAMETERS
# =========================
alpha=1.5
beta=3
rho=0.2
Qpher=100

D1 = None
D2 = None

tau1=np.ones_like(D1,dtype=float)
tau2=np.ones_like(D2,dtype=float)

# =========================
# SOLUTION CLASS
# =========================
class Sol:
    def __init__(self):
        self.E1={}
        self.E2={}
        self.cost=1e18

# =========================
# PROBABILITY
# =========================
def select_next(i, cand, tau, dist):

    if len(cand) == 1:
        return cand[0]

    probs = []

    for j in cand:
        pher = tau[i, j]
        heuristic = 1 / (dist[i, j] + 1e-6)

        val = (pher ** alpha) * (heuristic ** beta)
        probs.append(val)

    probs = np.array(probs, dtype=float)

    s = probs.sum()

    # ---------- FIX ----------
    if s <= 0 or not np.isfinite(s):
        # fallback → random choice
        return random.choice(cand)

    probs /= s
    return random.choices(cand, weights=probs)[0]


# =========================
# BUILD ECHELON 1
# =========================
def build_e1(sol):
    sats=VS.copy()
    random.shuffle(sats)
    groups=np.array_split(sats,len(VK1))

    for k,g in zip(VK1,groups):
        d=random.choice(VD)
        sol.E1[k]=[d]+list(g)+[d]

# =========================
# BUILD ECHELON 2
# =========================
def build_e2(sol):
    customers=VC.copy()
    random.shuffle(customers)
    chunks=np.array_split(customers,len(VK2))

    for k,ch in zip(VK2,chunks):
        s=random.choice(VS)
        route=[s]
        cur=s
        rem=list(ch)

        while rem:
            nxt=select_next(cur-3,[c-3 for c in rem],tau2,D2)
            real=nxt+3
            route.append(real)
            rem.remove(real)
            cur=real

        route.append(s)
        sol.E2[k]=route

# =========================
# FEASIBILITY
# =========================
def feasible(sol):

    # satellite load from E2
    sat_load={s:np.zeros(len(VP)) for s in VS}

    for r in sol.E2.values():
        s=r[0]
        for c in r[1:-1]:
            sat_load[s]+=Q2[c-6]

    # check Q1
    for i,s in enumerate(VS):
        if any(sat_load[s] > Q1[i]):
            return False

    # check customer coverage
    served=[]
    for r in sol.E2.values():
        served+=r[1:-1]

    if sorted(served)!=sorted(VC):
        return False

    return True

# =========================
# REPAIR OPERATORS
# =========================

# --- 1 COVERAGE ---
def repair_customer(sol):
    served=set()
    for r in sol.E2.values():
        for c in r[1:-1]:
            served.add(c)

    missing=[c for c in VC if c not in served]

    for c in missing:
        k=random.choice(VK2)
        sol.E2[k].insert(-1,c)

# --- 2 SAT LOAD ---
def repair_satellite(sol):
    for _ in range(5):
        sat_load={s:np.zeros(len(VP)) for s in VS}

        for r in sol.E2.values():
            s=r[0]
            for c in r[1:-1]:
                sat_load[s]+=Q2[c-6]

        for s in VS:
            idx=VS.index(s)
            if any(sat_load[s]>Q1[idx]):
                for k in VK2:
                    r=sol.E2[k]
                    if r[0]==s and len(r)>3:
                        c=random.choice(r[1:-1])
                        r.remove(c)
                        k2=random.choice(VK2)
                        sol.E2[k2].insert(-1,c)
                        break

# --- 3 TWO OPT ---
def cost_route(r,dist):
    c=0
    for i in range(len(r)-1):
        c+=dist[r[i],r[i+1]]
    return c

def two_opt(route,dist):
    best=route
    improved=True
    while improved:
        improved=False
        for i in range(1,len(route)-2):
            for j in range(i+1,len(route)-1):
                new=route[:i]+route[i:j][::-1]+route[j:]
                if cost_route(new,dist)<cost_route(best,dist):
                    best=new
                    improved=True
        route=best
    return best

def repair_2opt(sol):
    for k,r in sol.E1.items():
        sol.E1[k]=two_opt(r,D1)

    for k,r in sol.E2.items():
        rr=[r[0]]+[x-3 for x in r[1:-1]]+[r[0]-3]
        rr=two_opt(rr,D2)
        sol.E2[k]=[rr[0]]+[x+3 for x in rr[1:-1]]+[rr[0]]

# =========================
# COST
# =========================
def cost(sol):
    total=0

    for k,r in sol.E1.items():
        if len(r)>2: total+=F1[k]
        for i in range(len(r)-1):
            total+=C1[k]*D1[r[i],r[i+1]]

    for k,r in sol.E2.items():
        if len(r)>2: total+=F2[k]
        for i in range(len(r)-1):
            total+=C2[k]*D2[r[i]-3,r[i+1]-3]

    return total

# =========================
# MAIN S-ACO
# =========================
def run_saco():
    best=None
    T=300
    
    for t in range(1,T+1):
        season=get_season(t,T)
        ants=ANTS[season]
        sols=[]
    
        for _ in range(ants):
            s=Sol()
            build_e1(s)
            build_e2(s)
    
            # repair 1
            repair_customer(s)
            if not feasible(s): continue
    
            # repair 2
            repair_satellite(s)
            if not feasible(s): continue
    
            # repair 3
            repair_2opt(s)
            if not feasible(s): continue
    
            s.cost=cost(s)
            sols.append(s)
    
        if not sols:
            continue
    
        itbest=min(sols,key=lambda x:x.cost)
    
        if best is None or itbest.cost<best.cost:
            best=itbest
    
        tau1 = np.maximum(tau1*(1-rho), 1e-6)
        tau2 = np.maximum(tau2*(1-rho), 1e-6)
    
    
        for r in best.E1.values():
            for i in range(len(r)-1):
                tau1[r[i],r[i+1]]+=Qpher/best.cost
    
        for r in best.E2.values():
            for i in range(len(r)-1):
                tau2[r[i]-3,r[i+1]-3]+=Qpher/best.cost

    return best