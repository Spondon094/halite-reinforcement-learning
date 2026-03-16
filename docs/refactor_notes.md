# Refactor Notes

## Main cleanup changes

1. **Separated notebook code into reusable modules**
   - The original notebook mixed explanation, demos, and implementation.
   - The refactor isolates code into files under `src/`.

2. **Removed duplicated helper definitions**
   - `getDirTo()` appeared multiple times in the notebook.
   - It now exists once per module in a documented form.

3. **Made ship state handling safer**
   - In the original notebook, `ship_states[ship.id]` could be accessed before initialization.
   - The refactor uses a safe default state.

4. **Removed outdated TensorFlow v1 session setup**
   - The notebook used `tf.compat.v1` session management.
   - The cleaned script uses a simpler TensorFlow 2 style.

5. **Replaced deprecated optimizer argument**
   - `Adam(lr=...)` was updated to `Adam(learning_rate=...)`.

6. **Clarified the RL policy role**
   - The original code mixed learned actions with heuristic rules in a way that was hard to follow.
   - The refactor makes this hybrid structure explicit.

7. **Improved reward shaping**
   - Original notebook: reused cumulative score each step.
   - Refactor: uses score delta, which is easier to interpret.

## Issues present in the original notebook

### A. Single-ship control bias
The RL logic mainly acts on `me.ships[0]`, so the policy does not really scale to a full multi-ship game.

### B. Global state bug risk
The original `update_L1()` function incremented `ship_` without a `global` declaration, which would fail if called as written.

### C. Sparse state encoding
The network only sees the raw halite map, not richer strategic information such as:
- ship cargo,
- ship locations,
- shipyard locations,
- enemy ship locations,
- turn number.

### D. Notebook reproducibility
Because code and prose were interleaved, it was hard to rerun the project cleanly from top to bottom.

## Recommended modernization path

1. Add checkpoint saving and loading.
2. Add evaluation scripts against multiple fixed bots.
3. Build a richer observation encoder.
4. Move from single-ship control to coordinated multi-ship action generation.
5. Add experiment tracking for rewards and hyperparameters.
