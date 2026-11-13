# Citi Bike Overnight Rebalancing Optimization System (Midtown Manhattan)

An operational optimization framework developed in Python using **Gurobi** (`gurobipy`) to solve the overnight bike rebalancing problem across 165 stations in Midtown Manhattan. The system integrates historical GBFS feed snapshots, regularized Poisson regression demand forecasting, a fleet-conserving dynamic programming target allocation, Mixed-Integer Linear Programming (MILP) with Miller-Tucker-Zemlin (MTZ) subtour elimination, custom dynamic callback controls for heuristic and search phase stagnation, and geographic route visualization using `matplotlib`, `contextily`, and `pyproj`.

---

## 1. Project Overview & Key Upgrades

This project addresses the overnight Citi Bike rebalancing problem between 14:00 PM and 06:00 AM prior to the morning peak (06:00 AM – 10:00 AM). 

### Key Features & Model Refinements
* **Midtown Scope (14th St to 59th St)**: Comprehensive inclusion of all 165 active stations in Midtown Manhattan without downsampling, selected due to peak activity volume, activity per station, and absolute net flow density across historical trip records.
* **Effective Dock Capacity ($C_i$)**: Calculated as available bikes + available docks at midnight. Unavailable or damaged bikes/docks are excluded as they cannot recover overnight.
* **Demand Forecasting & Fleet-Conserving Allocation**: Combines a regularized Poisson regression model for morning pickups ($P_i$) and station-weekday averages for returns ($R_i$). Target inventories ($I_i^*$) are optimized via dynamic programming to minimize replayed blocked trips while strictly preserving total fleet inventory ($\sum_{i} I_i^* = \sum_{i} I_i^0$).
* **Optimized Depot Locations**: Discrete 2-median facility placement over station coordinates yields dual depot hubs at **Broadway & W 25th St** (Truck 1) and **W 47th St & 6th Ave** (Truck 2).
* **MTZ Subtour Elimination**: Polynomial Miller-Tucker-Zemlin formulations for exact subtour prevention across multi-stop routing tours.
* **Dynamic Stagnation Callbacks**: Automated phase transition control to advance out of the No Relaxation (NoRel) heuristic after 50 seconds without objective improvement, and terminate MIP search after 100 seconds without improvement.

---

## 2. Mathematical Model Formulation

### Objective Function

The MILP model minimizes total operational cost, trading off truck travel distance against penalties for station target inventory mismatches:

$$\min \alpha \sum_{k \in \mathcal{K}} \sum_{(i,j) \in \mathcal{A}_k} D_{ij} x_{ijk} + \beta \sum_{i \in \mathcal{I}} u_i$$

### Sets and Parameters

| Notation | Description | Value / Definition |
| :--- | :--- | :--- |
| $\mathcal{I}$ | Set of Midtown Manhattan stations | $n = 165$ stations |
| $\mathcal{K}$ | Fleet of rebalancing trucks | $\|\mathcal{K}\| = 2$ (`Truck_1`, `Truck_2`) |
| $S_k$ | Starting and ending depot for truck $k$ | Depot 1: $(40.7429^\circ \text{N}, -73.9892^\circ \text{W})$, Depot 2: $(40.7582^\circ \text{N}, -73.9820^\circ \text{W})$ |
| $\mathcal{V}_k$ | Node set for truck $k$ | $\mathcal{V}_k = \mathcal{I} \cup \{S_k\}$ |
| $\mathcal{A}_k$ | Feasible directed arcs for truck $k$ | $\mathcal{A}_k = \{(i,j) \mid i, j \in \mathcal{V}_k, i \neq j\}$ |
| $I_i^0$ | Initial available inventory at midnight | Extracted from GBFS status snapshot |
| $C_i$ | Effective station dock capacity | Available bikes + available docks at midnight |
| $I_i^*$ | Target inventory at station $i$ | Prepared via fleet-conserving dynamic programming |
| $G_i$ | Net required inventory change | $G_i = I_i^* - I_i^0$ ($G_i > 0$ deficit; $G_i < 0$ surplus) |
| $D_{ij}$ | Grid distance approximation (km) | $D_{ij} = \text{round}\left(84 \cdot \|\text{lon}_i - \text{lon}_j\| + 111 \cdot \|\text{lat}_i - \text{lat}_j\|, 3\right)$ |
| $Q$ | Maximum truck payload capacity | $Q = 25$ bikes |
| $W_k$ | Maximum distance allowance per truck | $W_k = 25.0$ km |
| $\alpha$ | Distance travel cost coefficient | $\$1.50 / \text{km}$ |
| $\beta$ | Target mismatch penalty coefficient | $\$5.00 / \text{bike}$ shortfall or surplus |
| $M$ | Load-tracking Big-M constant | $M = 50$ ($M = 2Q$) |

### Decision Variables

* $x_{ijk} \in \{0, 1\}$: Equals 1 if truck $k$ travels directly from node $i$ to node $j$; 0 otherwise.
* $z_{ik} \in \mathbb{Z}, -Q \le z_{ik} \le Q$: Net bikes delivered to station $i$ by truck $k$ ($z_{ik} < 0$ indicates pickup).
* $l_{ik} \in [0, Q]$: Load of truck $k$ immediately after visiting station $i$.
* $y_{ik} \in [0, n]$: MTZ continuous visit-order variable for subtour elimination.
* $u_i \in \mathbb{R}_{\ge 0}$: Absolute inventory deviation penalty from target stock at station $i$.

---

## 3. Key Constraints Summary

1. **Depot Flow & Single Visit Constraints**:
   $$\sum_{j \in \mathcal{I}} x_{S_k j k} = \sum_{i \in \mathcal{I}} x_{i S_k k} \le 1, \quad \forall k \in \mathcal{K}$$
   $$\sum_{j \in \mathcal{V}_k \setminus \{i\}} x_{j i k} = \sum_{j \in \mathcal{V}_k \setminus \{i\}} x_{i j k}, \quad \forall i \in \mathcal{I}, k \in \mathcal{K}$$
   $$\sum_{k \in \mathcal{K}} \sum_{j \in \mathcal{V}_k \setminus \{i\}} x_{j i k} \le 1, \quad \forall i \in \mathcal{I}$$

2. **Inventory Transfer Limits & Mismatch Tracking**:
   $$-\min(I_i^0, Q) \sum_{j \in \mathcal{V}_k \setminus \{i\}} x_{j i k} \le z_{ik} \le \min(C_i - I_i^0, Q) \sum_{j \in \mathcal{V}_k \setminus \{i\}} x_{j i k}, \quad \forall i \in \mathcal{I}, k \in \mathcal{K}$$
   $$u_i \ge G_i - \sum_{k \in \mathcal{K}} z_{ik}, \quad u_i \ge -G_i + \sum_{k \in \mathcal{K}} z_{ik}, \quad \forall i \in \mathcal{I}$$

3. **Truck Load Propagation & Return Conditions**:
   $$l_{S_k k} = 0, \quad 0 \le l_{ik} \le Q \sum_{j \in \mathcal{V}_k \setminus \{i\}} x_{j i k}, \quad \forall i \in \mathcal{I}, k \in \mathcal{K}$$
   $$-M(1 - x_{ijk}) \le l_{jk} - l_{ik} + z_{jk} \le M(1 - x_{ijk}), \quad \forall i \in \mathcal{V}_k \setminus \{j\}, j \in \mathcal{I}, k \in \mathcal{K}$$
   $$l_{ik} \le Q(1 - x_{i S_k k}), \quad \forall i \in \mathcal{I}, k \in \mathcal{K}$$
   $$\sum_{i \in \mathcal{I}} z_{ik} = 0, \quad \forall k \in \mathcal{K}$$

4. **Distance Budget & Subtour Elimination**:
   $$\sum_{(i,j) \in \mathcal{A}_k} D_{ij} x_{ijk} \le W_k, \quad \forall k \in \mathcal{K}$$
   $$0 \le y_{ik} \le n \sum_{j \in \mathcal{V}_k \setminus \{i\}} x_{j i k}, \quad \forall i \in \mathcal{I}, k \in \mathcal{K}$$
   $$y_{ik} \ge 1 - n(1 - x_{S_k i k}), \quad \forall i \in \mathcal{I}, k \in \mathcal{K}$$
   $$y_{jk} \ge y_{ik} + 1 - (n+1)(1 - x_{ijk}), \quad \forall i, j \in \mathcal{I}, i \neq j, k \in \mathcal{K}$$

---

## 4. Optimization Control & Stagnation Callback

The solver incorporates a custom dynamic Python callback (`create_stagnation_callback`) to manage search phases and prevent unproductive runtime:

```
                          ┌────────────────────────┐
                          │   Start Optimization   │
                          └───────────┬────────────┘
                                      │
                                      ▼
                        ┌────────────────────────────┐
                        │   Phase 0: NoRel Heuristic │
                        └─────────────┬──────────────┘
                                      │
                ┌─────────────────────┴─────────────────────┐
                │                                           │
       New Best Solution                             No Improvement for
       Found (< 50s stall)                              50 Seconds
                │                                           │
                ▼                                           ▼
      Reset Stagnation Timer                      model.cbProceed()
                │                        (Advance to Branch & Cut Search)
                └─────────────────────┬─────────────────────┘
                                      │
                                      ▼
                        ┌────────────────────────────┐
                        │ Phase 1 & 2: MIP Search    │
                        └─────────────┬──────────────┘
                                      │
                ┌─────────────────────┴─────────────────────┐
                │                                           │
       New Best Solution                             No Improvement for
       Found (< 100s stall)                             100 Seconds
                │                                           │
                ▼                                           ▼
      Reset Stagnation Timer                       model.terminate()
                                                (Return Best Incumbent)
```

### Stagnation Parameters
* **No Relaxation Heuristic Stagnation (50s)**: If NoRel (`GRB.Callback.MIP_PHASE == 0`) fails to discover a better integer feasible solution for 50 consecutive seconds, `model.cbProceed()` forces Gurobi to transition into LP relaxation and Branch-and-Cut MIP search immediately.
* **MIP Search Stagnation (100s)**: If standard MIP search (`GRB.Callback.MIP_PHASE >= 1`) makes no progress on the best incumbent solution for 100 consecutive seconds, `model.terminate()` cleanly stops solver execution and exports the current best incumbent solution.

---

## 5. Setup & Installation

### Step 1: Install Dependencies

Install the required Python environment dependencies:

```bash
pip install gurobipy contextily pyproj matplotlib pandas requests numpy
```

Or via `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Step 2: Configure Gurobi License

Ensure a valid Gurobi license file (`gurobi.lic`) is located in your user home directory or in the root project folder:

```bash
export GRB_LICENSE_FILE=/path/to/gurobi.lic
```

---

## 6. Running the Solver & CLI Usage

Run the optimization pipeline using `main.py`. The script provides command-line flags to adjust execution time limits, truck capacity, max distance allowances, and objective penalties.

```bash
python main.py [OPTIONS]
```

### Supported CLI Flags

| Parameter Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `-p`, `--time-limit` | `float` | `600.0` | Maximum solver time limit in seconds. |
| `-Q`, `--truck-capacity` | `int` | `25` | Maximum bike payload per truck ($Q$). |
| `-w`, `--max-distance` | `float` | `25.0` | Travel distance limit per truck in kilometers ($W_k$). |
| `-a`, `--alpha` | `float` | `1.5` | Travel cost per kilometer ($\alpha$). |
| `-b`, `--beta` | `float` | `5.0` | Inventory mismatch penalty per bike ($\beta$). |

### Execution Examples

* **Solve Midtown Model (1500s overall time limit backstop):**
  ```bash
  python main.py -p 1500
  ```

* **Custom Payload and Travel Allowances:**
  ```bash
  python main.py -p 1200 -Q 30 -w 30.0 -a 2.0 -b 6.0
  ```

---

## 7. Empirical Results & Benchmark Performance

Evaluating the MILP model on the 165-station Midtown dataset demonstrates significant reductions in target inventory mismatches:

| Metric | Baseline (No Rebalancing) | Gurobi MILP | Absolute Change | % Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **Abs. Inventory Mismatch** | $884\text{ bikes}$ | **$228\text{ bikes}$** | $-656\text{ bikes}$ | **$-74.2\%$** |
| **Approx. Total Travel Distance** | $0.000\text{ km}$ | **$49.445\text{ km}$** | $+49.445\text{ km}$ | N/A |
| **Travel Cost Term** ($\alpha = 1.5$) | $0.0000$ | **$74.1675$** | $+74.1675$ | N/A |
| **Mismatch Penalty Term** ($\beta = 5.0$) | $4,420.0000$ | **$1,140.0000$** | $-3,280.0000$ | **$-74.2\%$** |
| **Total Objective Value** | **$4,420.0000$** | **$1,214.1675$** | **$-3,205.8325$** | **$-72.5\%$** |

### Vehicle Performance Summary

| Truck ID | Depot Location | Coordinates | Stops Made | Total Dist. (km) | Budget Used | Bikes Picked Up / Delivered |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Truck 1** | Broadway & W 25 St | $(40.7429, -73.9892)$ | $51$ | $24.632\text{ km}$ | $98.5\%$ | $197\text{ bikes}$ |
| **Truck 2** | W 47 St & 6 Ave | $(40.7582, -73.9820)$ | $46$ | $24.813\text{ km}$ | $99.3\%$ | $132\text{ bikes}$ |
| **Total** | — | — | **$97$** | **$49.445\text{ km}$** | — | **$329\text{ bikes}$** |

> **Key Operational Finding:** Distance allowance $W_k = 25.0\text{ km}$ acts as the binding constraint limiting further mismatch reduction, rather than instantaneous vehicle payload capacity $Q$.

---

## 8. Operational Limitations & Future Extensions

1. **Street-Network Routing & Travel Time Constraints**:
   Current distances rely on Manhattan grid projections ($D_{ij}$). Integrating real-world street network directions (e.g., OpenStreetMap / OSRM) and travel/service time estimates will improve route schedule fidelity before the 06:00 AM cutoff.
2. **Dynamic Intra-Day Rebalancing**:
   The current framework optimizes static overnight rebalancing. Future work involves extending the formulation into a multi-period dynamic MIP that accounts for real-time customer trip activity during operational hours.