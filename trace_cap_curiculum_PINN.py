import os
import sys
import logging
import warnings

# ==========================================
# 1. השתקת אזהרות
# ==========================================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '0'
os.environ['AUTOGRAPH_VERBOSITY'] = '0'
warnings.filterwarnings('ignore')

import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.integrate import solve_ivp

tf.keras.backend.clear_session()
tf.keras.backend.set_floatx('float64')

# ==========================================
# 2. לוח בקרה - Control Panel
# ==========================================
EQUATION_TYPE   = 'standard'            
NU = 0.005 / np.pi

# תצורת Repulsive RAR קלאסי 
N_BOUNDARY = 500
N_INITIAL_COLLOCATION = 100                 
N_ADD_RAR = 100                        
REPULSION_RADIUS = 0.01               
N_DENSE_GRID = 10000                 
BATCH_SIZE = 500         

# תצורת Trace-Cap Curriculum (החידוש המחקרי!)
# החסם על היחס Tr(PDE) / Tr(Anchors). 
# מתחילים נמוך כדי ללמוד מבנה גלובלי, ומעלים (מרפים את החסם) לחידוד גל ההלם.
TRACE_RATIO_CAP_INITIAL = 5.0
TRACE_RATIO_CAP_FINAL = 50.0
# אם חסר מועמדים שעומדים בחסם, נוריד את המכסה של N_ADD_RAR כדי לא להפר את הבטיחות

# תצורת אימון - Adam
LEARNING_RATE = 0.001
EPOCHS_PRETRAIN = 1000               
EPOCHS_ADAM = 30000                  
RESAMPLE_FREQ_ADAM = 1000            
PRINT_FREQ = 500                     
NTK_SAMPLE_SIZE = 1500 

# תצורת אימון - L-BFGS Chunking
LBFGS_MAX_ITER = 20000
LBFGS_CHUNKS = 1                     
LBFGS_FTOL = 1.0 * np.finfo(float).eps

print(f"=== Trace-Cap Curriculum RAR PINN ===")
print(f"Equation: {EQUATION_TYPE.upper()} Burgers | Nu: {NU:.5f}")
print(f"Curriculum: Trace Ratio Cap relaxes from {TRACE_RATIO_CAP_INITIAL} to {TRACE_RATIO_CAP_FINAL}")
print(f"L-BFGS Config: {LBFGS_CHUNKS} Chunks\n")

# ==========================================
# 3. Ground Truth (RK45) 
# ==========================================
def solve_burgers_numerical(eq_type, nu, nx=2000, nt_eval=100):
    x = np.linspace(-1, 1, nx)
    dx = x[1] - x[0]
    u0 = -np.sin(np.pi * x)

    def rhs(t, u):
        u[0] = 0; u[-1] = 0
        u_x = np.zeros_like(u)
        u_x[1:-1] = (u[2:] - u[:-2]) / (2 * dx)
        u_xx = np.zeros_like(u)
        u_xx[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / (dx**2)
        if eq_type == 'standard': return -u * u_x + nu * u_xx
        else: return -(u**2) * u_x + nu * u_xx

    t_eval = np.linspace(0, 1, nt_eval)
    sol = solve_ivp(rhs, [0, 1], u0, t_eval=t_eval, method='RK45')
    return sol.t, x, sol.y.T

gt_t, gt_x, gt_u = solve_burgers_numerical(EQUATION_TYPE, NU)

def get_exact_value(t_req, x_req):
    points = np.array([t_req.flatten(), x_req.flatten()]).T
    T_grid, X_grid = np.meshgrid(gt_t, gt_x, indexing='ij')
    points_src = np.array([T_grid.flatten(), X_grid.flatten()]).T
    values_src = gt_u.flatten()
    u_interp = griddata(points_src, values_src, points, method='linear')
    return u_interp.reshape(t_req.shape)

x_l2_eval, t_l2_eval = np.linspace(-1, 1, 100), np.linspace(0, 1, 100)
X_grid_l2, T_grid_l2 = np.meshgrid(x_l2_eval, t_l2_eval)
X_flat_l2 = tf.cast(np.hstack((X_grid_l2.flatten()[:, None], T_grid_l2.flatten()[:, None])), tf.float64)
U_exact_l2 = get_exact_value(T_grid_l2, X_grid_l2)

# ==========================================
# 4. המודל ואתחול משתנים
# ==========================================
def get_boundary_data(n_b):
    x_ic = np.random.uniform(-1, 1, (n_b, 1)); t_ic = np.zeros((n_b, 1))
    u_ic = -np.sin(np.pi * x_ic)
    t_bc = np.random.uniform(0, 1, (n_b, 1))
    X_b = np.concatenate([np.hstack((x_ic, t_ic)), np.hstack((-1 * np.ones((n_b, 1)), t_bc)), np.hstack((1 * np.ones((n_b, 1)), t_bc))])
    U_b = np.concatenate([u_ic, np.zeros((n_b, 1)), np.zeros((n_b, 1))])
    return tf.cast(X_b, tf.float64), tf.cast(U_b, tf.float64)

X_b, U_b = get_boundary_data(N_BOUNDARY)

x_c_np = np.random.uniform(-1, 1, (N_INITIAL_COLLOCATION, 1))
t_c_np = np.random.uniform(0, 1, (N_INITIAL_COLLOCATION, 1))

class PINN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.hidden_layers = [tf.keras.layers.Dense(20, activation='tanh', dtype='float64') for _ in range(4)] # שמנו 4 שכבות לפי הדיון הקודם!
        self.out = tf.keras.layers.Dense(1, dtype='float64')
    def call(self, x):
        for l in self.hidden_layers: x = l(x)
        return self.out(x)

model = PINN()
optim = tf.keras.optimizers.Adam(LEARNING_RATE)

history = {'epoch': [], 'trace_pde': [], 'trace_bc': [], 'ratio': [], 'l2_error': [], 'trace_cap': []}
global_step = 0

@tf.function
def compute_loss(x_col, t_col):
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(x_col); tape2.watch(t_col)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(x_col); tape1.watch(t_col)
            u = model(tf.concat([x_col, t_col], axis=1))
        u_x = tape1.gradient(u, x_col)
        u_t = tape1.gradient(u, t_col)
    u_xx = tape2.gradient(u_x, x_col)
    del tape1; del tape2
    
    f = u_t + u * u_x - NU * u_xx if EQUATION_TYPE == 'standard' else u_t + (u**2) * u_x - NU * u_xx
    lf = tf.reduce_mean(tf.square(f))
    lb = tf.reduce_mean(tf.square(model(X_b) - U_b))
    return lb, lf

@tf.function
def train_bc():
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(tf.square(model(X_b) - U_b))
    optim.apply_gradients(zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables))

@tf.function
def train_step_adam(x_col, t_col):
    with tf.GradientTape() as tape:
        lb, lf = compute_loss(x_col, t_col)
        loss = lb + lf
    grads_total = tape.gradient(loss, model.trainable_variables)
    optim.apply_gradients(zip(grads_total, model.trainable_variables))
    return loss

@tf.function
def compute_ntk_traces(x_samp, t_samp, X_b_samp):
    with tf.GradientTape(persistent=True) as tape_w:
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(x_samp); tape2.watch(t_samp)
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch(x_samp); tape1.watch(t_samp)
                u = model(tf.concat([x_samp, t_samp], axis=1))
            u_x = tape1.gradient(u, x_samp)
            u_t = tape1.gradient(u, t_samp)
        u_xx = tape2.gradient(u_x, x_samp)
        del tape1; del tape2
        f = u_t + u * u_x - NU * u_xx if EQUATION_TYPE == 'standard' else u_t + (u**2) * u_x - NU * u_xx
        u_bc = model(X_b_samp)

    J_f = tape_w.jacobian(f, model.trainable_variables, experimental_use_pfor=False)
    J_b = tape_w.jacobian(u_bc, model.trainable_variables, experimental_use_pfor=False)
    del tape_w

    J_f_flat = tf.concat([tf.reshape(j, [tf.shape(f)[0], -1]) for j in J_f], axis=1)
    J_b_flat = tf.concat([tf.reshape(j, [tf.shape(u_bc)[0], -1]) for j in J_b], axis=1)
    
    # חישוב טרייס נקודתי עבור כל דגימת PDE
    pointwise_trace_pde = tf.reduce_sum(tf.square(J_f_flat), axis=1)
    
    return tf.reduce_sum(pointwise_trace_pde), tf.reduce_sum(tf.square(J_b_flat)), pointwise_trace_pde, f

@tf.function
def compute_candidates_batch(x_batch, t_batch):
    with tf.GradientTape(persistent=True) as tape_w:
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(x_batch); tape2.watch(t_batch)
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch(x_batch); tape1.watch(t_batch)
                u = model(tf.concat([x_batch, t_batch], axis=1))
            u_x = tape1.gradient(u, x_batch)
            u_t = tape1.gradient(u, t_batch)
        u_xx = tape2.gradient(u_x, x_batch)
        del tape1; del tape2
        
        f = u_t + u * u_x - NU * u_xx if EQUATION_TYPE == 'standard' else u_t + (u**2) * u_x - NU * u_xx

    J_f = tape_w.jacobian(f, model.trainable_variables, experimental_use_pfor=False)
    del tape_w
    J_f_flat = tf.concat([tf.reshape(j, [tf.shape(f)[0], -1]) for j in J_f], axis=1)
    pointwise_trace = tf.reduce_sum(tf.square(J_f_flat), axis=1)
    
    res_abs = tf.abs(f)
    return tf.reshape(res_abs, [-1]), tf.reshape(pointwise_trace, [-1])


def get_current_trace_cap(epoch, total_epochs):
    # פונקציה ליניארית (או אחרת) להרפיית החסם
    progress = epoch / float(total_epochs)
    return TRACE_RATIO_CAP_INITIAL + progress * (TRACE_RATIO_CAP_FINAL - TRACE_RATIO_CAP_INITIAL)


def trace_cap_rar_step(x_current, t_current, current_ratio_cap):
    # 1. חישוב טרייסים קיימים
    n_curr = len(x_current)
    idx_pde = np.random.choice(n_curr, min(NTK_SAMPLE_SIZE, n_curr), replace=False)
    idx_bc = np.random.choice(len(X_b.numpy()), min(NTK_SAMPLE_SIZE, len(X_b.numpy())), replace=False)
    
    x_c_tf = tf.cast(x_current[idx_pde], tf.float64)
    t_c_tf = tf.cast(t_current[idx_pde], tf.float64)
    X_bc_tf = tf.gather(X_b, idx_bc)
    
    tr_pde_t, tr_bc_t, _, _ = compute_ntk_traces(x_c_tf, t_c_tf, X_bc_tf)
    base_tr_pde = tr_pde_t.numpy()
    base_tr_bc = tr_bc_t.numpy() + 1e-8
    
    # חישוב תקציב טרייס נותר (Budget) להוספה
    max_allowed_pde_trace = current_ratio_cap * base_tr_bc
    trace_budget = max(0, max_allowed_pde_trace - base_tr_pde)
    
    # 2. דגימת מועמדים
    x_dense = np.random.uniform(-1, 1, (N_DENSE_GRID, 1))
    t_dense = np.random.uniform(0, 1, (N_DENSE_GRID, 1))
    all_res = np.zeros(N_DENSE_GRID)
    all_traces = np.zeros(N_DENSE_GRID)
    
    for i in range(0, N_DENSE_GRID, BATCH_SIZE):
        end_idx = min(i + BATCH_SIZE, N_DENSE_GRID)
        x_batch = tf.cast(x_dense[i:end_idx], tf.float64)
        t_batch = tf.cast(t_dense[i:end_idx], tf.float64)
        
        res_out, traces_out = compute_candidates_batch(x_batch, t_batch)
        all_res[i:end_idx] = res_out.numpy()
        all_traces[i:end_idx] = traces_out.numpy()
        
    # מיון מועמדים לפי שארית (הכי גרועים קודם) - מנגנון Greedy
    sorted_indices = np.argsort(all_res)[::-1]
    
    x_new, t_new = [], []
    existing_pts = np.hstack((x_current, t_current))
    r_sq = REPULSION_RADIUS**2
    accumulated_trace = 0.0
    
    for idx in sorted_indices:
        if len(x_new) >= N_ADD_RAR: break
        
        cand_trace = all_traces[idx]
        
        # חסם הטרייס - אם הנקודה תחרוג מהתקציב, נדלג עליה!
        if accumulated_trace + cand_trace > trace_budget:
            continue
            
        pt = np.array([x_dense[idx, 0], t_dense[idx, 0]])
        
        # NMS
        if len(existing_pts) > 0 and np.min(np.sum((existing_pts - pt)**2, axis=1)) < r_sq: continue
        if len(x_new) > 0:
            new_pts_arr = np.hstack((np.array(x_new).reshape(-1,1), np.array(t_new).reshape(-1,1)))
            if np.min(np.sum((new_pts_arr - pt)**2, axis=1)) < r_sq: continue
                
        # אישור הנקודה
        x_new.append(pt[0])
        t_new.append(pt[1])
        accumulated_trace += cand_trace
        
    actual_added = len(x_new)
    if actual_added > 0:
        return np.vstack((x_current, np.array(x_new).reshape(-1, 1))), np.vstack((t_current, np.array(t_new).reshape(-1, 1))), actual_added
    return x_current, t_current, 0


def evaluate_and_log(epoch, optimizer, action, current_cap, added_pts=0):
    n_curr = len(x_c_np)
    idx_pde = np.random.choice(n_curr, min(NTK_SAMPLE_SIZE, n_curr), replace=False)
    idx_bc = np.random.choice(len(X_b.numpy()), min(NTK_SAMPLE_SIZE, len(X_b.numpy())), replace=False)
    
    tr_pde_t, tr_bc_t, _, _ = compute_ntk_traces(tf.cast(x_c_np[idx_pde], tf.float64), tf.cast(t_c_np[idx_pde], tf.float64), tf.gather(X_b, idx_bc))
    tr_pde, tr_bc = tr_pde_t.numpy(), tr_bc_t.numpy()
    current_ratio = tr_pde / (tr_bc + 1e-8)
    
    U_pred_l2 = model(X_flat_l2).numpy().reshape(X_grid_l2.shape)
    rel_err = np.linalg.norm(U_exact_l2 - U_pred_l2) / np.linalg.norm(U_exact_l2)
    
    added_str = f"| Added: {added_pts:3d}" if added_pts > 0 else "| Added:   0"
    print(f"Step: {epoch:5d} | Opt: {optimizer:5s} | Ratio/Cap: {current_ratio:5.1f}/{current_cap:5.1f} | Error: {rel_err:6.2%} | Action: {action:12s} {added_str}")
    
    history['epoch'].append(epoch)
    history['trace_pde'].append(tr_pde)
    history['trace_bc'].append(tr_bc)
    history['ratio'].append(current_ratio)
    history['l2_error'].append(rel_err)
    history['trace_cap'].append(current_cap)
    
    return current_ratio

# ==========================================
# 5. לולאת אימון - Adam
# ==========================================
print(f"--- Stage 0: Pre-training BC ({EPOCHS_PRETRAIN} epochs) ---")
for i in range(EPOCHS_PRETRAIN): 
    train_bc()

print(f"\n--- Stage 1: Adam Training (Trace-Cap Curriculum) ---")
TOTAL_STEPS_ESTIMATE = EPOCHS_ADAM + LBFGS_MAX_ITER

for epoch in range(EPOCHS_ADAM + 1):
    current_cap = get_current_trace_cap(epoch, TOTAL_STEPS_ESTIMATE)
    
    if epoch > 0 and epoch % PRINT_FREQ == 0:
        if epoch % RESAMPLE_FREQ_ADAM == 0:
            x_c_np, t_c_np, pts_added = trace_cap_rar_step(x_c_np, t_c_np, current_cap)
            evaluate_and_log(epoch, "RAR-A", "Curriculum", current_cap, pts_added)
        else:
            evaluate_and_log(epoch, "Adam", "בדיקת שפיות", current_cap)

    if epoch < EPOCHS_ADAM:
        x_c_tf, t_c_tf = tf.cast(x_c_np, tf.float64), tf.cast(t_c_np, tf.float64)
        train_step_adam(x_c_tf, t_c_tf)

global_step = EPOCHS_ADAM

# ==========================================
# 6. L-BFGS Chunked Fine-Tuning 
# ==========================================
print(f"\n--- Stage 2: Chunked L-BFGS Fine-Tuning ---")

@tf.function
def compute_loss_and_grads_tf_final(x_col, t_col):
    with tf.GradientTape() as tape:
        lb, lf = compute_loss(x_col, t_col)
        loss = lb + lf
    grads = tape.gradient(loss, model.trainable_variables)
    flat_grads = tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)
    return loss, flat_grads

def get_loss_and_grad_wrapper(x_col_tf, t_col_tf):
    def loss_and_grad_wrapper(weights_1d):
        start = 0
        for var in model.trainable_variables:
            size = tf.size(var).numpy()
            var.assign(tf.reshape(weights_1d[start:start+size], var.shape))
            start += size
        loss, grads = compute_loss_and_grads_tf_final(x_col_tf, t_col_tf)
        return loss.numpy().astype(np.float64), grads.numpy().astype(np.float64)
    return loss_and_grad_wrapper

def lbfgs_callback(xk):
    global global_step
    global_step += 1
    if global_step % PRINT_FREQ == 0:
        current_cap = get_current_trace_cap(global_step, TOTAL_STEPS_ESTIMATE)
        evaluate_and_log(global_step, "LBFGS", "ליטוש", current_cap)

chunk_iters = LBFGS_MAX_ITER // LBFGS_CHUNKS

for chunk in range(LBFGS_CHUNKS):
    print(f"\n-- Starting L-BFGS Chunk {chunk + 1}/{LBFGS_CHUNKS} --")
    
    x_c_tf, t_c_tf = tf.cast(x_c_np, tf.float64), tf.cast(t_c_np, tf.float64)
    wrapper_func = get_loss_and_grad_wrapper(x_c_tf, t_c_tf)

    scipy.optimize.minimize(
        fun=wrapper_func, 
        x0=np.concatenate([w.numpy().flatten() for w in model.trainable_variables]), 
        method='L-BFGS-B', jac=True, callback=lbfgs_callback,
        options={'maxiter': chunk_iters, 'maxfun': chunk_iters, 'ftol': LBFGS_FTOL}
    )
    
    if chunk < LBFGS_CHUNKS - 1:
        current_cap = get_current_trace_cap(global_step, TOTAL_STEPS_ESTIMATE)
        x_c_np, t_c_np, pts_added = trace_cap_rar_step(x_c_np, t_c_np, current_cap)
        evaluate_and_log(global_step, "RAR-L", "Curriculum", current_cap, pts_added)

print(f"\n--- Optimization Complete ---")

# ==========================================
# 7. הפקת גרפים ואנליזת טרייסים
# ==========================================
U_pred = model.predict(X_flat_l2, verbose=0).reshape(X_grid_l2.shape)
final_l2 = np.linalg.norm(U_exact_l2 - U_pred) / np.linalg.norm(U_exact_l2)
print(f"Final Validation L2 Error (On Ground Truth): {final_l2:.2%}")

fig, ax = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f"Burgers Eq | Trace-Cap Curriculum RAR + L-BFGS Chunks | Error: {final_l2:.2%}", fontweight='bold')

c0 = ax[0].contourf(T_grid_l2, X_grid_l2, U_pred, levels=50, cmap='jet')
fig.colorbar(c0, ax=ax[0])
ax[0].set_title('PINN Prediction (Contour)')
ax[0].set_xlabel('Time (t)'); ax[0].set_ylabel('Space (x)')

c1 = ax[1].contourf(T_grid_l2, X_grid_l2, np.abs(U_exact_l2 - U_pred), levels=50, cmap='inferno')
fig.colorbar(c1, ax=ax[1])
ax[1].set_title('Absolute Error')
ax[1].set_xlabel('Time (t)'); ax[1].set_ylabel('Space (x)')

for t_val, color in {0.2: 'green', 0.5: 'gold', 0.8: 'blue'}.items():
    idx = np.argmin(np.abs(t_l2_eval - t_val))
    ax[2].plot(x_l2_eval, U_exact_l2[idx, :], ls='-', lw=2.5, alpha=0.5, color=color)
    ax[2].plot(x_l2_eval, U_pred[idx, :], ls=':', lw=3, color=color, label=f't={t_val}')
ax[2].set_title('Wave Profiles')
ax[2].legend()

plt.tight_layout()
plt.savefig("final_pinn_results.png")

# --- גרף האנליזה המורחב (כולל החסם הדינמי) ---
fig2, ax2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle("Trace-Cap Curriculum Diagnostics", fontweight='bold')

ax2[0].plot(history['epoch'], history['trace_pde'], label='Trace($K_{PDE}$)', color='darkred', lw=2)
ax2[0].plot(history['epoch'], history['trace_bc'], label='Trace($K_{BC}$)', color='darkblue', lw=2)
ax2[0].axvline(EPOCHS_ADAM, color='black', ls='--', alpha=0.5, label='L-BFGS Start')
ax2[0].set_yscale('log')
ax2[0].set_xlabel('Epochs / Steps')
ax2[0].set_ylabel('Spectral Power')
ax2[0].set_title('NTK Spectral Traces & L2 Error')
ax2[0].grid(alpha=0.3)

ax2_l2 = ax2[0].twinx()
l2_percent = np.array(history['l2_error']) * 100
ax2_l2.plot(history['epoch'], l2_percent, label='L2 Error (%)', color='darkorange', lw=2.5, ls='-.')
ax2_l2.set_ylabel('L2 Error (%)', color='darkorange')

lines_1, labels_1 = ax2[0].get_legend_handles_labels()
lines_2, labels_2 = ax2_l2.get_legend_handles_labels()
ax2[0].legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right')

# ציור יחס הטרייסים מול מכסת הטרייס (Trace Cap)
ax2[1].plot(history['epoch'], history['ratio'], label='Actual Ratio (PDE/BC)', color='purple', lw=2)
ax2[1].plot(history['epoch'], history['trace_cap'], label='Curriculum Trace Cap $\\tau$', color='black', ls=':', lw=2)
ax2[1].axvline(EPOCHS_ADAM, color='black', ls='--', alpha=0.5)
ax2[1].set_yscale('log')
ax2[1].set_title('Trace Ratio vs. Curriculum Cap')
ax2[1].set_xlabel('Epochs / Steps')
ax2[1].set_ylabel('Ratio (Log Scale)')
ax2[1].grid(alpha=0.3)
ax2[1].legend()

plt.tight_layout()
plt.savefig("trace_diagnostics.png")

# --- הדפסת טופולוגיית הנקודות הסופית (Initial vs Curriculum RAR) ---
plt.figure(figsize=(7, 5))
plt.scatter(t_c_np[:N_INITIAL_COLLOCATION], x_c_np[:N_INITIAL_COLLOCATION], 
            color='gray', s=3, alpha=0.4, label='Initial Points')
if len(t_c_np) > N_INITIAL_COLLOCATION:
    plt.scatter(t_c_np[N_INITIAL_COLLOCATION:], x_c_np[N_INITIAL_COLLOCATION:], 
                color='red', s=8, alpha=0.8, label='Curriculum Added Points')

plt.title('Final Collocation Points Topology (Curriculum Filtered)', fontweight='bold')
plt.xlabel('Time (t)')
plt.ylabel('Space (x)')
plt.legend(loc='upper right')
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("final_collocation_points.png")
plt.show()