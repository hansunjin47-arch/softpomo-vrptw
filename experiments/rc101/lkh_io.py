"""LKH-3 file IO extracted from clustering_routing/R_lkh.py."""
import os
# LKH 실행 자체는 history.py에서 담당한다. 이 파일은 입력 작성과 출력 경로 파싱만 담당한다.
from typing import List, Optional

def load_solomon(path: str) -> dict:
    coords, demands, tw_open, tw_close, service_time = [], [], [], [], []
    vehicle_limit = vehicle_capacity = None

    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    name = lines[0]
    i = 0
    while i < len(lines):
        upper = lines[i].upper()
        if upper == "VEHICLE":
            vals = lines[i + 2].split()
            vehicle_limit = int(vals[0])
            vehicle_capacity = float(vals[1])
            i += 3
        elif upper.startswith("CUST"):
            i += 2
        else:
            parts = lines[i].split()
            if len(parts) >= 7 and parts[0].isdigit():
                coords.append((float(parts[1]), float(parts[2])))
                demands.append(float(parts[3]))
                tw_open.append(float(parts[4]))
                tw_close.append(float(parts[5]))
                service_time.append(float(parts[6]))
            i += 1

    return {
        "name": name,
        "coords": coords,
        "demands": demands,
        "tw_open": tw_open,
        "tw_close": tw_close,
        "service_time": service_time,
        "vehicle_limit": vehicle_limit,
        "vehicle_capacity": vehicle_capacity,
        "n_customers": len(coords) - 1,
    }

def _write_vrp_file(
    inst: dict,
    tt: List[List[int]],
    vrp_path: str,
    n_vehicles: int,
    zone_routes: Optional[List[List[int]]] = None,
    distance_limit: Optional[int] = None,
) -> None:
    """Write VRPLIB VRPTW problem file for LKH-3."""
    n_nodes = len(inst["coords"])
    depot_close = int(inst["tw_close"][0])
    # DISTANCE = max route duration. For rain TT, caller can pass a larger value.
    dist = distance_limit if distance_limit is not None else depot_close

    with open(vrp_path, "w") as f:
        f.write(f"NAME: problem\n")
        f.write(f"COMMENT: VRPTW\n")
        f.write(f"TYPE: CVRPTW\n")
        f.write(f"DIMENSION: {n_nodes}\n")
        f.write(f"VEHICLES: {n_vehicles}\n")
        f.write(f"CAPACITY: {int(inst['vehicle_capacity'])}\n")
        f.write(f"DISTANCE: {dist}\n")
        f.write("EDGE_WEIGHT_TYPE: EXPLICIT\n")
        f.write("EDGE_WEIGHT_FORMAT: FULL_MATRIX\n")
        f.write("EDGE_WEIGHT_SECTION\n")
        for row in tt:
            f.write(" ".join(str(v) for v in row) + "\n")
        f.write("DEMAND_SECTION\n")
        for i in range(n_nodes):
            f.write(f"{i + 1} {int(inst['demands'][i])}\n")
        f.write("TIME_WINDOW_SECTION\n")
        for i in range(n_nodes):
            f.write(f"{i + 1} {int(inst['tw_open'][i])} {int(inst['tw_close'][i])}\n")
        f.write("SERVICE_TIME_SECTION\n")
        for i in range(n_nodes):
            f.write(f"{i + 1} {int(inst['service_time'][i])}\n")

        # Zone restriction: fix each customer to its vehicle via PICKUP_AND_DELIVERY workaround
        # LKH-3 does not support hard zone assignment natively; we skip soft zone hints here.
        # (Zone-based ordering is handled by initial tour hinting if needed.)

        f.write("DEPOT_SECTION\n1\n-1\nEOF\n")

def _write_par_file(
    vrp_wsl: str,
    tour_wsl: str,
    par_path: str,
    time_limit: int,
    runs: int,
    max_trials: int,
    mtsp_objective: Optional[str] = None,
) -> None:
    """Write LKH-3 parameter file (Windows path, content uses WSL paths)."""
    with open(par_path, "w") as f:
        f.write(f"PROBLEM_FILE = {vrp_wsl}\n")
        f.write(f"OUTPUT_TOUR_FILE = {tour_wsl}\n")
        f.write(f"TIME_LIMIT = {time_limit}\n")
        f.write(f"MAX_TRIALS = {max_trials}\n")
        f.write(f"RUNS = {runs}\n")
        f.write("TRACE_LEVEL = 0\n")
        f.write("SEED = 42\n")
        if mtsp_objective:
            f.write(f"MTSP_OBJECTIVE = {mtsp_objective}\n")

def _parse_tour(tour_path: str, n_customers: int) -> List[List[int]]:
    """Parse LKH-3 CVRPTW tour file → list of 0-indexed customer routes.

    LKH-3 adds V depot copies for V vehicles, so DIMENSION = n_customers + V.
    Depot copies have node index 1 or > n_customers + 1.
    """
    if not os.path.exists(tour_path):
        return []
    n_cust_nodes = n_customers + 1  # depot=1, customers=2..n_customers+1
    routes: List[List[int]] = []
    current: List[int] = []
    in_tour = False
    with open(tour_path, "r") as f:
        for line in f:
            line = line.strip()
            if line == "TOUR_SECTION":
                in_tour = True
                continue
            if not in_tour:
                continue
            if line in ("-1", "EOF"):
                break
            node = int(line)
            is_depot = (node == 1 or node > n_cust_nodes)
            if is_depot:
                if current:
                    routes.append(current)
                    current = []
            else:
                current.append(node - 1)  # 0-indexed: customers become 1..n_customers
    if current:
        routes.append(current)
    return [r for r in routes if r]
