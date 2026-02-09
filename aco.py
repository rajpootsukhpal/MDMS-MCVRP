import numpy as np
import random

# -------------------------
# PARAMETERS
# -------------------------
alpha = 1.5
beta  = 3
rho   = 0.2
Qpher = 100

N_ANTS = 40
ITER   = 1000

D1 = None
D2 = None
# pheromones
tau1 = np.ones_like(D1, dtype=float)
tau2 = np.ones_like(D2, dtype=float)

# =========================================================
# SOLUTION CLASS
# =========================================================
class Sol:
    def __init__(self):
        self.E1 = {}
        self.E2 = {}
        self.cost = 1e18


# =========================================================
# NODE INDEX HELPERS
# =========================================================
def node_to_e2_index(n):
    if n in VS:
        return sat_to_idx[n]
    if n in VC:
        return len(VS) + cust_to_idx[n]
    raise ValueError("invalid node")

def e2_index_to_node(i):
    if i < len(VS):
        return VS[i]
    return VC[i-len(VS)]


# =========================================================
# SAFE SELECTION
# =========================================================
def select_next(i, cand, tau, dist):

    if len(cand) == 1:
        return cand[0]

    vals = []
    for j in cand:
        pher = tau[i,j]
        heur = 1/(dist[i,j] + 1e-6)
        vals.append((pher**alpha)*(heur**beta))

    vals = np.array(vals)
    s = vals.sum()

    if s <= 0 or not np.isfinite(s):
        return random.choice(cand)

    vals /= s
    return random.choices(cand, weights=vals)[0]


# =========================================================
# BUILD ECHELON 1
# =========================================================
def build_e1(sol):

    sats = VS.copy()
    random.shuffle(sats)
    groups = np.array_split(sats, len(VK1))

    for k,g in zip(VK1,groups):
        if len(g)==0:
            continue
        d = random.choice(VD)
        sol.E1[k] = [d] + list(g) + [d]


# =========================================================
# BUILD ECHELON 2
# =========================================================
def build_e2(sol):

    customers = VC.copy()
    random.shuffle(customers)
    chunks = np.array_split(customers, len(VK2))

    for k,ch in zip(VK2,chunks):

        if len(ch)==0:
            continue

        s = random.choice(VS)
        route=[s]
        cur=s
        rem=list(ch)

        while rem:

            i = node_to_e2_index(cur)
            cand=[node_to_e2_index(c) for c in rem]

            nxt = select_next(i, cand, tau2, D2)
            real = e2_index_to_node(nxt)

            route.append(real)
            rem.remove(real)
            cur=real

        route.append(s)
        sol.E2[k]=route


# =========================================================
# FEASIBILITY
# =========================================================
def feasible(sol):

    sat_load={s:np.zeros(len(VP)) for s in VS}

    for r in sol.E2.values():
        s=r[0]
        for c in r[1:-1]:
            sat_load[s]+=Q2[cust_to_idx[c]]

    for s in VS:
        if any(sat_load[s] > Q1[sat_to_idx[s]]):
            return False

    served=[]
    for r in sol.E2.values():
        served+=r[1:-1]

    if sorted(served)!=sorted(VC):
        return False

    return True


# =========================================================
# COST
# =========================================================
def cost(sol):

    total=0

    for k,r in sol.E1.items():
        if len(r)>2:
            total+=F1[k]
        for i in range(len(r)-1):
            total+=C1[k]*D1[r[i],r[i+1]]

    for k,r in sol.E2.items():
        if len(r)>2:
            total+=F2[k]
        for i in range(len(r)-1):
            a=node_to_e2_index(r[i])
            b=node_to_e2_index(r[i+1])
            total+=C2[k]*D2[a,b]

    return total


# =========================================================
# SIMPLE REPAIR
# =========================================================
def repair(sol):

    served=set()
    for r in sol.E2.values():
        for c in r[1:-1]:
            served.add(c)

    missing=[c for c in VC if c not in served]

    for c in missing:
        k=random.choice(VK2)
        if k in sol.E2:
            sol.E2[k].insert(-1,c)


# =========================================================
# MAIN ACO
# =========================================================
def run_aco():

    global tau1, tau2
    best=None

    for it in range(ITER):

        solutions=[]

        for _ in range(N_ANTS):

            s=Sol()
            build_e1(s)
            build_e2(s)

            repair(s)

            if not feasible(s):
                continue

            s.cost = cost(s)
            solutions.append(s)

        if not solutions:
            continue

        itbest=min(solutions, key=lambda x:x.cost)

        if best is None or itbest.cost < best.cost:
            best = itbest

        # pheromone update
        tau1 = np.maximum(tau1*(1-rho), 1e-6)
        tau2 = np.maximum(tau2*(1-rho), 1e-6)

        for r in best.E1.values():
            for i in range(len(r)-1):
                tau1[r[i],r[i+1]] += Qpher/best.cost

        for r in best.E2.values():
            for i in range(len(r)-1):
                a=node_to_e2_index(r[i])
                b=node_to_e2_index(r[i+1])
                tau2[a,b] += Qpher/best.cost

    return best

