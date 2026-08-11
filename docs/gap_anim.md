# Animation layering — `AnimationRig`, bone masks, blend spaces, additive blending, root motion

Status: landed on `feat/anim`. Nothing here is wired into the renderer, the
frame packet or `GraphicsSettings` — this is pure CPU animation infrastructure
plus one playground demonstration. **`animation_player.hpp` and
`animation_player.cpp` are byte-for-byte unmodified.**

This file is the integrator's material. Fold the four sections below into
`docs/user_guide.md`, `docs/technical_spec.md`, `docs/architecture.md` and
`CLAUDE.md` respectively.

---

## 1. User guide

### 1.1 When to reach for a rig, and when not to

`AnimationPlayer` is not deprecated and must not be "unified" away. It plays one
clip (plus one crossfade partner), costs 16,552 bytes per instance, and is what
every existing doc describes. Use it for anything that only ever plays one thing
at a time — an idle villager, a windmill, a door.

`AnimationRig` is for a character that needs a **second simultaneous animation
source**: an upper-body action over locomotion, an additive aim offset, a
breathing layer, a blend space. It costs 27,296 bytes per instance plus cursor
storage, and it needs a `PoseScratch` (15,360 bytes) per **evaluating thread**.

A rig holding exactly one Override layer at weight 1 with a full mask produces
**bitwise identical** output to an `AnimationPlayer` playing the same clip,
crossfades included. That is pinned by
`AnimationRig.SingleOverrideLayerMatchesAnimationPlayerBitwise`.

### 1.2 The smallest complete example

```cpp
#include "bracken/bracken.hpp"

namespace anim = bracken::animation;

class Game : public bracken::Application {
    anim::ClipBindings bindings_;   // ONE per model, shared const by every character
    anim::AnimationRig rig_;
    anim::PoseScratch  scratch_;    // one per evaluating THREAD, not per character
    int upper_layer_ = -1;
    int attack_clip_ = -1;
    float upper_weight_ = 0.0f;

    bool on_init() override {
        const bracken::Model* model = models().get_model(character_);
        bindings_.build(model->skeleton, model->animations);
        rig_.bind(model->skeleton, model->animations, bindings_);

        // Layer 0 is the BASE. Its mode/weight/mask are ignored — it writes the
        // composed pose directly.
        anim::LayerDesc base;
        base.name = "locomotion";
        (void)rig_.add_layer(std::move(base));

        // Layer 1: an additive upper body, masked to the spine subtree.
        const anim::JointIndex spine = anim::find_joint(model->skeleton, "Spine");
        anim::LayerDesc upper;
        upper.name   = "upper_body";
        upper.mode   = anim::BlendMode::Additive;
        upper.weight = 0.0f;                       // faded in by gameplay
        upper.mask.set_subtree(model->skeleton, spine.value);
        upper.additive_reference_clip = -1;        // -1 == the skeleton's rest pose
        upper_layer_ = rig_.add_layer(std::move(upper));

        attack_clip_ = anim::find_clip_index(model->animations, "Attack");
        rig_.play_clip(upper_layer_, attack_clip_, 0.0f);
        rig_.set_layer_loop(upper_layer_, false);
        return true;
    }

    void on_simulate(bracken::SimFrame& sim) override {
        const float dt = sim.delta_time();
        const bracken::Model* model = models().get_model(character_);

        sm_.set_float(param_speed_, speed_);
        sm_.update(rig_, /*layer=*/0);           // the machine drives layer 0 only

        const float target = attacking_ ? 1.0f : 0.0f;
        upper_weight_ += (target - upper_weight_) * std::min(1.0f, dt * 18.0f);
        rig_.set_layer_weight(upper_layer_, upper_weight_);

        rig_.evaluate(model->skeleton, model->animations, dt, scratch_);

        // Identical to the player from here on: IK, attachments, the palette.
        anim::apply_body_lean(rig_.mutable_bone_matrices(), rig_.mutable_world_transforms(),
                              model->skeleton, spine_index_, lean_, 0.0f);

        auto palette = sim.frame().alloc_bone_palette(joint_count_);
        rig_.write_palette(palette.matrices);
        sim.frame().draw_skinned(character_, transform_, palette.id);
    }
};
```

Three things in that example are the whole API:

* **`ClipBindings` is per MODEL, not per character.** It is the clip→joint→
  channel table. Building one per instance is the memory regression the type
  exists to avoid — eight blend-space samples at 128 joints would otherwise cost
  8 KB of pointers per character.
* **`PoseScratch` is per evaluating THREAD.** One per `parallel_for` chunk. Two
  rigs sharing one scratch concurrently corrupt each other silently; there is no
  cheap assert for it.
* **Layer 0 is the base and its weight/mask are ignored.** The pose has to start
  somewhere.

### 1.3 Bone masks and the pop at the boundary

```cpp
anim::BoneMask upper;
upper.set_subtree(skeleton, spine_index);        // root + every descendant
upper.set(jaw_index, false);                     // carve one back out
const int marked = upper.count();
```

`set_subtree` requires joints ordered parents-before-children, which every
skeleton the model loader produces satisfies. Out-of-range `set`/`test` are
no-ops, not UB.

**A binary mask pops.** The joint at the boundary jumps between two poses while
its parent does not, and it reads as a hitch at the waist when the layer fades
in. The engine does not smooth this for you. Supply a per-joint ramp:

```cpp
std::array<float, anim::MAX_BONES> feather{};   // MUST outlive the layer:
                                                 // LayerDesc stores the SPAN
anim::feather_by_depth(skeleton, hips_index, /*falloff=*/2, feather);
upper_desc.joint_weights = std::span<const float>(feather.data(), feather.size());
```

Root the ramp at the mask root's **parent**, not at the mask root: depth 0 gets
weight 0, and rooting at the spine would give the spine itself no weight — which
is usually the joint the layer most needs.

### 1.4 Additive layers

An additive layer adds `weight * (source - reference)` on top of everything
below it. The reference is normally the rest pose (`additive_reference_clip =
-1`) or frame 0 of the additive clip's own source animation.

The rotation delta is **post-multiplied, in the bone's own frame**:

```
delta  = inverse(reference.rotation) * source.rotation
result = slerp(base.rotation, base.rotation * delta, weight)
```

That means "this additive clip means the same thing regardless of where the base
put the bone" — an aim offset authored as a 30-degree lift stays a 30-degree
lift of the arm after locomotion has swung the arm elsewhere. The alternative
(`source * inverse(reference)` applied on the LEFT) is a parent-frame delta and
the two differ whenever the base has rotated the joint, which on a running
character is always.

Consequences a content author must know:

* a joint with **no channel** in the additive clip contributes a **zero delta**
  and keeps whatever the base was doing with it. This is why an additive
  upper-body layer keeps the off-hand swinging and a masked *Override* layer
  does not (an Override layer's un-channelled joints fall back to the rest pose);
* the clip should start and end at the reference so the layer fades in and out
  of nothing;
* **non-uniform scale in an additive layer is unsupported.** Scale is applied as
  a ratio and the pipeline bakes scale into the local matrix by scaling columns,
  so a scaled parent shears a rotated child. Keep additive clips rotation- and
  translation-only.

### 1.5 Blend spaces

```cpp
anim::BlendSpace1D locomotion;
(void)locomotion.add_sample(anim::find_clip_index(clips, "Idle"),   0.0f);
(void)locomotion.add_sample(anim::find_clip_index(clips, "Walk"), 150.0f);
(void)locomotion.add_sample(anim::find_clip_index(clips, "Run"),  400.0f);
locomotion.finalize();                       // sorts by position — required
rig.set_blend_space_1d(0, locomotion);
rig.set_blend_position(0, measured_speed);   // per frame
```

```cpp
anim::BlendSpace2D strafe;                   // polar: (0,0) means "not moving"
(void)strafe.add_sample(idle_clip,     glm::vec2(0.0f, 0.0f));
(void)strafe.add_sample(forward_clip,  glm::vec2(0.0f, 1.0f));
(void)strafe.add_sample(back_clip,     glm::vec2(0.0f, -1.0f));
(void)strafe.add_sample(left_clip,     glm::vec2(-1.0f, 0.0f));
(void)strafe.add_sample(right_clip,    glm::vec2(1.0f, 0.0f));
rig.set_blend_space_2d(0, strafe);
rig.set_blend_position(0, glm::vec2(strafe_axis, forward_axis));
```

Blend spaces take clip **indices**, not names — resolve them once with
`find_clip_index`. Installing a blend space on a layer **allocates** (it sizes
that layer's samplers), so do it at setup or on state entry, never per frame.

`LayerDesc::sync_duration` (default on) keeps every active sample at the same
normalized phase, so a walk/run blend does not skate. Turn it off when the
samples are unrelated poses rather than variants of one cycle.

Set `LayerDesc::cache_cursors = false` on blend-space layers unless a profile
says otherwise: cursors cost 24 B per joint per sampler and change no float.

### 1.6 Root motion — read this before enabling it

**No asset in this repository carries usable root translation, and the shipped
character will stop moving if you enable this on it.** The playground's XZ
motion comes from the physics character controller; the character's only
position channel is a vertical hips bob.

For root motion to work an artist must author:

1. the motion on a **joint inside `skin.joints`** — conventionally `Hips`.
   `ModelLoader` drops any glTF channel whose target node is outside the skin
   (counting it in `ModelLoadReport::channels_non_skin_node`), and root motion is
   conventionally authored on an Armature node **above** the skin, so the usual
   export produces a clip the engine cannot see;
2. that joint at the **origin in XZ** in the rest pose. The host applies the
   delta in the character's frame, so a non-zero XZ reference gets rotated by the
   accumulated yaw and the character drifts;
3. a loop whose start and end positions are compatible, or the wrap seam jumps.

```cpp
anim::RootMotionConfig cfg;
cfg.joint          = anim::find_joint(skeleton, "Hips").value;
cfg.translation_xz = true;
cfg.translation_y  = false;   // a hips bob is NOT locomotion — leave this off
cfg.rotation_yaw   = true;
cfg.strip_from_pose = true;   // if false, the host WILL double-apply
rig.set_root_motion(cfg);

// ...after evaluate(), every frame, exactly once:
const anim::RootMotionDelta d = rig.root_motion();
character_yaw = character_yaw * d.rotation;
world_position += glm::mat3_cast(character_yaw) * d.translation;
```

The engine never moves anything. The delta is returned as a value the host must
consume; ignoring it makes the character animate on the spot, and leaving
`strip_from_pose` off makes it travel at double speed.

### 1.7 State machines and layers

`AnimationStateMachine::update(AnimationRig&, int layer)` drives one layer.
`layer >= 0` overrides every state's own `AnimState::layer`; `layer < 0` (the
default) uses the state's field, so one machine can own a base layer and another
an upper-body layer without either knowing about the other.

A state can install a blend space instead of a clip:

```cpp
const int speed = sm.param_index("speed");     // resolve BEFORE building the state

anim::AnimState locomotion;
locomotion.name   = "Locomotion";
locomotion.layer  = 0;
locomotion.source = anim::LayerSource::Blend1D;
locomotion.blend_param_x = speed;
(void)locomotion.blend_space_1d.add_sample(idle_clip, 0.0f);
(void)locomotion.blend_space_1d.add_sample(run_clip, 400.0f);
locomotion.blend_space_1d.finalize();
sm.add_state(locomotion);
```

Every existing behaviour is preserved: transition priority, AND/OR conditions and
the `!loop && !playing` clip-ended trigger all run through **one** shared
evaluator (`next_transition`), which both `update` overloads call. The
`AnimationPlayer` overload is untouched.

`bind_clips` returns `false` for a machine whose states are blend spaces — a
blend-space state has no `clip_name` to resolve. That is expected, not an error.

### 1.8 Attachments on a rig

`attachment.hpp` gained `PoseView` overloads. `AnimationRig::pose()` produces
one; the `AnimationPlayer` overloads are now one-line forwarders to the same
code, so the two can never disagree.

```cpp
const glm::mat4 sword = attachment_transform(model_transform, rig.pose(), hand, grip);
const glm::mat4 held  = lantern_attach_.world(model_transform, rig.pose());
```

---

## 2. Spec

### 2.1 New files

| File | Contents |
|---|---|
| `src/bracken/animation/pose.hpp` / `.cpp` | `BoneTransform`, `PoseSpan`/`ConstPoseSpan`, `PoseView`, `BoneMask`, `feather_by_depth`, `rest_pose`, `BlendParams`, `blend_override`, `blend_additive`, `pose_to_matrices`, `PoseScratch` |
| `src/bracken/animation/clip_sampler.hpp` / `.cpp` | `sample_clip_joint_into`, `ClipBindings`, `ClipSampler`, `sample_clip_pose` |
| `src/bracken/animation/blend_space.hpp` / `.cpp` | `k_max_blend_samples`, `k_blend_weight_epsilon`, `WeightedClip`, `find_clip_index`, `BlendSpace1D`, `BlendSpace2D`, `effective_duration` |
| `src/bracken/animation/root_motion.hpp` / `.cpp` | `RootMotionDelta`, `RootMotionConfig`, `twist_about_y`, `extract_root_motion`, `strip_root_motion` |
| `src/bracken/animation/animation_rig.hpp` / `.cpp` | `BlendMode`, `LayerSource`, `k_max_animation_layers`, `LayerDesc`, `AnimationRig` |

Tests: `pose.test.cpp`, `blend_space.test.cpp`, `root_motion.test.cpp`,
`animation_rig.test.cpp`, `animation_rig_parallel.test.cpp`. Benchmarks appended
to `animation.bench.cpp`.

### 2.2 Data structures and sizes (measured, x86-64 / GCC / glm default alignment)

| Type | `sizeof` |
|---|---|
| `BoneTransform` | 40 |
| `BoneMask` | 16 (`(MAX_BONES + 63) / 64` = 2 words) |
| `BlendSpace1D` | 104 |
| `BlendSpace2D` | 132 |
| `LayerDesc` | 88 |
| `ClipSampler` | 40 (+ heap: 24 B/joint when caching cursors) |
| `RootMotionConfig` | 8 |
| `WeightedClip` | 16 |
| `PoseScratch` | 15,360 (3 × `MAX_BONES` × 40) |
| `AnimationPlayer` | 16,552 (unchanged) |
| `AnimationRig` | 27,296 |

`AnimationRig`'s 27,296 bytes are 8,192 (`bone_matrices_`) + 8,192
(`world_transforms_`) + 5,120 (`composed_`, the composed local pose) + ~5,632
for the eight-entry layer table + bookkeeping. The layer table is a fixed array,
so it is paid whether or not the layers are used.

Heap on top, per rig: a `Clip` layer with `cache_cursors` holds two cursor
vectors of 24 B/joint (6,144 B at 128 joints); a blend-space layer holds one per
sample (24,576 B for a full eight-sample layer at 128 joints). `cache_cursors =
false` reduces both to zero and changes no float.

**`MAX_BONES` is unchanged at 128.** Every new fixed-size array is expressed in
terms of the constant, never the literal.

### 2.3 Constants

| Constant | Value | Where | What it bounds |
|---|---|---|---|
| `k_max_animation_layers` | 8 | `animation/animation_rig.hpp` | layers per rig; the array extent of the layer table, so raising it costs ~704 B per extra slot in *every* rig |
| `k_max_blend_samples` | 8 | `animation/blend_space.hpp` | samples per blend space; bounds the 2D estimator's O(n²) loop (8 → 56 pair evaluations per resolve per layer per character) and the sampler array per layer |
| `k_blend_weight_epsilon` | 1e-4 | `animation/blend_space.hpp` | weights below this are culled rather than sampled |
| `BlendSpace2D::k_direction_scale` | 2.0 | `animation/blend_space.hpp` | how much an angular difference weighs against a radial one. **Tuned constant with no measurement in this tree** — it is the value the technique is published with |
| `BoneMask::k_words` | `(MAX_BONES + 63) / 64` = 2 | `animation/pose.hpp` | mask storage, tracks `MAX_BONES` |

### 2.4 Algorithms

**`pose_to_matrices`** is steps (4)(5)(6) of
`AnimationPlayer::compute_bone_matrices`, hoisted unchanged: TRS → local matrix
(`mat4_cast`, columns scaled, column 3 = translation), one ascending
`world[i] = world[parent] * local[i]` pass, then
`bone[i] = world[i] * inverse_bind_matrix`. The local matrix is built **in place**
in `world_transforms` and accumulated over itself, so no scratch buffer is
needed — parents-before-children is load-bearing twice.

**`blend_override`** is `mix` / `slerp` / `mix` per joint, the same three calls in
the same order as the player's crossfade. Effective weight ≤ 0 leaves the
destination bitwise untouched; ≥ 1 **assigns** rather than interpolating. Both
end-point cases are what make the zero-weight and single-layer identities exact
rather than approximate.

**`blend_additive`**: rotation post-multiply in the bone's own frame (§1.4),
translation as a plain difference, scale as a ratio with a zero-reference guard
(a reference component below 1e-8 yields a ratio of 1).

**`BlendSpace1D::resolve`**: clamp to `[first, last]`, bracket with a linear
scan, emit at most two entries with weights `1-t` and `t`, culling either below
`k_blend_weight_epsilon`. Exact at every sample, partition of unity, C0,
clamping rather than extrapolating outside the authored range. Coincident
samples produce `t = 0` rather than a divide by zero.

**`BlendSpace2D::resolve`** is **gradient-band freeform directional** (polar).
For each sample `i`, with `r_k = |p_k|`:

```
avg   = (r_i + r_j) / 2
v_ij  = ( (r_j - r_i) / avg,  signed_angle(p_i, p_j) * k_direction_scale )
v_ip  = ( (r   - r_i) / avg,  signed_angle(p_i, p)   * k_direction_scale )
h_ij  = clamp01( 1 - dot(v_ip, v_ij) / dot(v_ij, v_ij) )
w_i   = min over j != i of h_ij
```

then cull and normalize. The angle term is zero when either operand has ~zero
magnitude (a centre sample has no direction), and `dot(v_ij, v_ij) ~ 0`
(coincident samples) yields `h = 1`.

*Why this and not Delaunay/barycentric.* A triangulation needs a triangulation
step whose degenerate cases (collinear samples, duplicate points, a query outside
the hull) each need a defined answer — more code than the estimator itself, for
the one requirement that mattered here. It also behaves badly for the case the
feature exists to serve: a centre sample plus a ring of directions triangulates
into long thin slivers and a query near the centre picks whichever sliver it
lands in. The polar formulation understands that `(0,0)` means "not moving" and
that the angle between two directions matters more than their cartesian
distance, which is the semantics of a locomotion blend.

*Known property, not a bug:* the estimator is **C0 but not C1**. `min()` creates
a crease wherever the binding constraint changes, so a character crossing one
gets a kink rather than a smooth roll. This is inherent to gradient band (Unity
ships the same artifact). Polar coordinates are additionally singular at the
origin — the angular term's gradient goes as `1/|p|` — which is why the
continuity test excludes a small disc there.

*Defined degeneracies, all tested:* zero samples → zero entries; one sample →
weight 1 everywhere; coincident samples → even split, deterministic; every
sample at the origin → uniform weights; a query outside the cover → the single
nearest sample at weight 1 (never an empty result, never a divide by a zero sum).

**Blend-space composition** is a normalized incremental blend: start from the
first contribution, fold each subsequent one in at `w_k / (accumulated + w_k)`.
Exactly the first sample when there is one, and independent of the weights'
absolute scale.

**Phase sync**: the layer holds one normalized phase, advanced by
`dt * speed / effective_duration` where
`effective_duration = Σ w_i · (duration_i / speed_scale_i) / Σ w_i`, and sample
`i` is evaluated at `phase * duration_i`.

**Root motion**: `twist_about_y` is a swing-twist decomposition about +Y,
returning identity (never NaN) when `w² + y² ≈ 0`. Extraction differences the
root channel between two clip times, masked by the config; on a wrap it composes
`(duration - t_prev)` with `(t_now - 0)`, translations added and rotations
composed **post × pre**. A joint with no channel yields a zero delta rather than
reading the rest pose as motion. Stripping resets the root's XZ (and/or Y) to the
reference and cancels the yaw with `inverse(twist_about_y(q)) * q`, preserving
pitch and roll and leaving every other joint bitwise untouched.

For a blend-space base layer, root translation is the weighted sum over active
samples and the rotation comes from the **dominant** sample. **That rotation is
discontinuous when the dominant sample changes** — a documented limitation; a
slerp chain over an unordered active set has no canonical order.

### 2.5 Error handling and diagnostics

* `AnimationRig::add_layer` returns `-1` (and logs a `WARN`) past
  `k_max_animation_layers`; the rig stays evaluable.
* `BlendSpace{1D,2D}::add_sample` returns `false` past `k_max_blend_samples` or
  for a negative clip index. Both are `[[nodiscard]]`.
* Every out-of-range layer index on a setter is a no-op; every getter returns a
  neutral value; `layer(int)` returns a discard reference whose writes are
  dropped.
* `AnimationRig::evaluate` carries one `BRACKEN_CONTRACT`: the rig must have been
  bound for the skeleton's joint count. It is a per-frame host precondition, so
  it is live in release.
* `AnimationRig::bind` logs an `ERROR` when the bindings were built for a
  different clip count and a `WARN` when the skeleton exceeds `MAX_BONES`.

---

## 3. Architecture

### 3.1 Where it sits

Entirely inside the existing `animation/` module — **no new directory**, so
`docs/architecture.md`'s module map needs no new row. The module graph stays a
DAG: `pose.hpp` → `animation_types.hpp`; `clip_sampler.hpp` → `animation_types`,
`pose`; `blend_space.hpp` → `animation_types`; `root_motion.hpp` →
`animation_types`, `clip_sampler`, `pose`; `animation_rig.hpp` → all of the
above; `animation_state_machine.hpp` gained `animation_rig.hpp` and
`blend_space.hpp`; `attachment.hpp` gained `pose.hpp`. Nothing reaches `scene/`,
`render/`, `contracts/` or the root.

In the frame, an `AnimationRig` sits exactly where an `AnimationPlayer` sat: host
sim work inside `on_simulate`, before foot IK and body lean, before the bone
palette is written into packet memory. **The frame packet, `ScenePrep`, the
renderer and `GraphicsSettings` are untouched** — the rig emits the same palette
through the same `alloc_bone_palette`, and `scene_prep.cpp`'s
`min(model joints, palette joints, MAX_BONES)` clamp is unaffected.

### 3.2 Evaluation order inside `evaluate()`

1. `root_delta_` reset to zero (so reading it twice cannot double-apply);
2. for each layer in **index order**: advance its clock (even when muted — a
   layer faded back in must not resume where it was faded out), then
   * layer 0 writes the composed pose **directly** (its mode/weight/mask are
     ignored, which is what removes the `lerp(x, y, 1.0f)` that would spoil
     bit-identity),
   * every other layer samples into `scratch.layer` and is composed with
     `blend_override` or `blend_additive`;
3. root motion extracted from layer 0 and, when `strip_from_pose`, stripped from
   the composed pose;
4. `pose_to_matrices`;
5. shrink-only tail identity fill — the freed slots are re-identitied **only**
   when the joint count drops, exactly as `AnimationPlayer` does it (a per-update
   rewrite of both `MAX_BONES` arrays is ~8 KB of dead stores per instance per
   frame).

**Layer order is a documented contract, not an accident**: swapping two
overlapping layers changes the result, and `AnimationRig.LayerCompositionIsIndexOrderDependent`
asserts it so.

### 3.3 Threading tiers and lifetimes

| Thing | Tier | Notes |
|---|---|---|
| `AnimationRig::evaluate` | `sim-stage` / `worker-safe` per instance | no allocation, no logging, touches only this rig, its arguments and the scratch |
| `AnimationRig::bind`, `add_layer`, `set_blend_space_*` | `owned`, setup-time | these are the calls that allocate |
| `ClipBindings` | `worker-safe` once built | const-shared across instances; **aliases the clip vector**, so it must be rebuilt if the model is reloaded and must outlive every rig bound to it |
| `ClipSampler` | `owned` | per-instance mutable cursors, exactly like `AnimationPlayer`'s |
| `PoseScratch` | `owned`, **per evaluating thread** | one per `parallel_for` chunk; two rigs must never share one concurrently |
| `BlendSpace1D` / `2D` | `worker-safe` | `resolve` is const and stateless |
| `blend_override` / `blend_additive` / `pose_to_matrices` / `rest_pose` / root motion | `worker-safe` | operate only on their arguments |

`AnimationRig` is **move-only**, for the same reason `AnimationPlayer` is: a
copied rig would share keyframe cursors with the original, which was a historical
source of animation glitches.

Determinism: nothing here has an atomic, a mutex or a shared write. Evaluation is
a pure function of (rig state, skeleton, clips, dt), so the parallel test's
demand for bitwise-equal output at any worker count is satisfiable by
construction — and is asserted rather than assumed.

### 3.4 The duplication, and why it is deliberate

`clip_sampler.cpp`'s `sample_clip_joint_into` is a **character-for-character
copy** of the file-static `sample_clip_joint` in `animation_player.cpp`, not a
refactor of it. `animation_player.cpp` is byte-for-byte unmodified.

The reason: the engine's screenshot baselines are pinned against the playground
character's final pose, which feeds `rebuild_shadow_capsules` and therefore
traced sun shadows. Moving a function out of a translation unit is *almost
certainly* value-neutral on x86-64/ARM64 — the body does no arithmetic of its
own — but "almost certainly" is what moves a baseline nobody can re-measure
cheaply. A duplicate proven equal by test is cheaper than a moved baseline.

The anti-drift guard is not the comment. It is
`AnimationRig.SingleOverrideLayerMatchesAnimationPlayerBitwise` and
`AnimationRig.SingleOverrideLayerReproducesACrossfadeBitwise`, which `memcmp`
120 frames of rig output against `AnimationPlayer` output. Editing one copy
without the other turns them red immediately. `pose.test.cpp`'s
`PoseToMatricesReproducesAnimationPlayerBitwise` covers the matrix half the same
way.

### 3.5 One deliberate behavioural difference from `AnimationPlayer`

`AnimationPlayer::update` returns early when the model has no clips, so a
never-animated skeleton has no world transforms and every attachment on it reads
identity (`attachment.hpp` documents this and tells you to add a single-keyframe
clip). `AnimationRig::evaluate` produces the **rest pose** instead, which is a
usable answer. Documented on `evaluate()` and asserted by
`AnimationRig.ARigWithNoLayersProducesTheRestPose`.

---

## 4. CLAUDE.md delta

### 4.1 Module table — replace the `animation/` row

> | `animation/` | `bracken::animation` | GLB skeleton, `AnimationPlayer` (crossfade), data-driven state machine, two-bone IK — **plus a layer stack**: `AnimationRig` composes up to `k_max_animation_layers` weighted, `BoneMask`-masked `Override`/`Additive` layers onto a base pose in local TRS space (`pose.hpp` is the pose type the engine never had), each layer driven by a clip or by a 1D/2D blend space (`blend_space.hpp`, gradient-band freeform directional in 2D), then run through the same hierarchy walk the player uses. `root_motion.hpp` extracts the root/hips delta for the host to apply and strips it from the pose. `AnimationPlayer` is **not** deprecated by this: it is the one-clip fast path at 16.5 KB against the rig's 27 KB, and a one-layer rig is bitwise identical to it. |

### 4.2 Load-bearing constants — add

> - `k_max_animation_layers = 8` (`animation/animation_rig.hpp`) — layers per
>   `AnimationRig`. It is the extent of a fixed array inside every rig, so raising
>   it costs ~704 B per extra slot in every rig whether the layers are used or not.
> - `k_max_blend_samples = 8` (`animation/blend_space.hpp`) — samples per blend
>   space. Bounds the 2D estimator's O(n²) inner loop (8 samples → 56 pair
>   evaluations per resolve, per layer, per character) **and** the array of
>   `ClipSampler`s a layer holds, which at 128 joints with cursors cached is
>   24 B/joint each.
> - **`MAX_BONES` was NOT raised.** `BoneMask` is
>   `std::array<uint64_t, (MAX_BONES + 63) / 64>` and `PoseScratch` is three
>   `std::array<BoneTransform, MAX_BONES>`; both track the constant rather than
>   the literal 128.

### 4.3 Key patterns — add

> - **Animation layering** composes in **local TRS space**, before the hierarchy
>   walk — that is the seam the engine previously did not have, because per-joint
>   TRS lived for four statements as locals inside
>   `AnimationPlayer::compute_bone_matrices` and was collapsed straight to a
>   `mat4`. `AnimationRig` layer 0 is the BASE and its weight/mask are ignored by
>   design; layers compose in INDEX ORDER and that order is a contract. The three
>   pose buffers a composition needs live in a caller-supplied `PoseScratch`, one
>   per evaluating THREAD (not per character), so per-character memory does not
>   scale with layer count.

### 4.4 Gotchas — add

> - **A blend space is exact at its samples and C0 between them, but not C1.**
>   `BlendSpace2D`'s `min()` over pair constraints creates creases where the
>   binding constraint changes; a character crossing one gets a visible kink. It
>   is inherent to gradient band (Unity ships the same artifact) and it is the
>   price of not maintaining a triangulation. Polar coordinates are also singular
>   at the origin.
> - **A binary `BoneMask` pops at the boundary.** The joint at the edge jumps
>   while its parent does not. `LayerDesc::joint_weights` is the hook and
>   `feather_by_depth` builds a serviceable ramp — root it at the mask root's
>   PARENT, or the mask root itself gets weight 0. The engine does not feather for
>   you.
> - **Additive rotation is POST-multiplied (bone-local):**
>   `dst.r * (inverse(ref.r) * src.r)`. The parent-frame alternative looks fine
>   until the base pose is rotated, which on a running character is always.
> - **Root motion has no data in this tree, and turning it on will freeze a
>   controller-driven character.** No shipped clip carries horizontal root
>   translation; the playground's only hips channel is a vertical bob, and glTF
>   root motion authored above the skin is dropped at load
>   (`ModelLoadReport::channels_non_skin_node`). See `docs/skinned_mesh_format.md`
>   for what an artist must author: motion on a joint INSIDE `skin.joints`, that
>   joint at the origin in XZ in the rest pose.
> - **`LayerDesc::joint_weights` stores a SPAN, not a copy.** A local feather
>   array dangles the moment the setup function returns.
> - **`clip_sampler.cpp`'s `sample_clip_joint_into` is a deliberate duplicate of
>   the file-static `sample_clip_joint` in `animation_player.cpp`.** The player was
>   left byte-for-byte unmodified so the screenshot pins could not move; the guard
>   against the two copies drifting is
>   `AnimationRig.SingleOverrideLayerMatchesAnimationPlayerBitwise`, which memcmps
>   120 frames of both. Edit one, edit the other.

### 4.5 Screenshot baselines — THE PINS MOVE, AND THEY WERE NOT RE-MEASURED

The playground's character now runs on an `AnimationRig`:

* the state machine drives **layer 0** and no longer has an `Attack` state —
  `Attack` was a whole-body `Override` state that every locomotion state fell
  into at priority 10, which stopped the legs dead so the villager had to stand
  still to swing;
* **layer 1** is a masked, feathered, **additive** upper-body layer that plays
  the `Attack` clip, faded in by `attack_timer_`.

That changes the character's pose whenever an attack is in flight, and the
character's pose feeds `rebuild_shadow_capsules`, which feeds traced sun shadows.
**Every row of every pin table must be re-measured**, not copied. The smoke
driver fires an attack on `frame % 120 == 60` and the screenshot is captured at
frame 220 of 240, so the raster rows are affected too.

`docs/skinned_mesh_format.md` also needs a **"Root motion"** subsection under
"Animation channels": the motion must be authored on a joint inside
`skin.joints` (conventionally `Hips`) because channels targeting nodes outside
the skin are dropped at load, and that joint must sit at the origin in XZ in the
rest pose.

### 4.6 Documentation the playground port made STALE

These were deliberately not edited (five agents were working in parallel and all
five would have conflicted on them). They now describe a playground that no
longer exists, and every one of them is a line the integrator must fix:

| File:line | What it now says that is false |
|---|---|
| `docs/technical_spec.md:299` | "states Idle/Walk/Run/**Attack**", "transitions on parameters `speed` (float) and **`attacking` (bool)**", "any→Attack on `attacking IS_TRUE` with crossfade 0.08 and priority 10; Attack is non-looping and exits to Idle on `IS_FALSE`", "`param_index("speed"/"attacking")`". The machine now has three states and one parameter; Attack is a rig layer. |
| `docs/technical_spec.md:301` | "`anim_sm_.update(anim_player_)` then `anim_player_.update(...)`", and the foot-IK sentence citing `anim_player_.world_transforms()`. |
| `docs/technical_spec.md:303` | "copy `anim_player_.bone_matrices()` into `pal.matrices`" — it is `anim_rig_.write_palette(palette.matrices)` now. |
| `docs/user_guide.md:315`, `:317` | the `on_simulate` walkthrough, same two claims. |
| `docs/user_guide.md:3446-3459` | the "how a host animates a character" recipe. Worth rewriting as the rig recipe with the player one kept beside it — both are supported. |
| `docs/user_guide.md:3598`, `:3659-3664` | the state-machine section, which quotes the Attack state and `param_attacking_` verbatim. |
| `docs/user_guide.md:3724-3799` | the procedural-layers walkthrough and its numbered summary. |
| `docs/user_guide.md:7793` | `sword_.world(character_transform, anim_player_)` — still a valid API (the `AnimationPlayer` overload was kept), so this one is correct but no longer matches the playground. |

`docs/multithreading_plan.md:162-164` uses `anim_player_` in an ILLUSTRATIVE
snippet about job safety, not as a claim about the playground. It is still true
and can stay.

None of these fail `core/docs_coherence.test.cpp` — the identifiers it pairs
with files are unaffected — which is exactly why they need a human to catch
them.
