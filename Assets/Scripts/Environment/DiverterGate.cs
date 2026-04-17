using UnityEngine;

/// <summary>
/// Gate FSM states. Activate() is only valid in Retracted.
/// Committed is entered by Activate() and waits for a package to cross the
/// CommitZoneTrigger before transitioning to Deploying.
/// </summary>
public enum GateState { Retracted, Committed, Deploying, Deployed, Retracting }

/// <summary>
/// Paddle gate with 4-state FSM: Retracted → Deploying → Deployed → Retracting → Retracted.
///
/// Paddle rotates between <see cref="_retractedEulerAngles"/> and <see cref="_deployedEulerAngles"/>
/// via a kinematic Rigidbody using MoveRotation (so non-kinematic packages collide correctly).
///
/// Adaptive cooldown after a full cycle completes:
///   cooldown = baseCooldown / max(beltSpeed, 0.1), clamped [minCooldown, maxCooldown].
/// The max(..., 0.1) guard prevents division-by-zero at zero belt speed.
///
/// Place in: Assets/Scripts/Environment/
/// Attach to: a GameObject with a Kinematic Rigidbody and a BoxCollider for the paddle.
/// </summary>
[RequireComponent(typeof(Rigidbody))]
public class DiverterGate : MonoBehaviour
{
    // ── Identity ────────────────────────────────────────────────────
    [Header("Branch Identity")]
    [Tooltip("Which branch this gate diverts to. 0 = Branch1, 1 = Branch2, 2 = Branch3.")]
    [SerializeField] private int _branchIndex;

    // ── Paddle rotation ─────────────────────────────────────────────
    [Header("Paddle Rotation (local Euler angles)")]
    [Tooltip("Paddle rotation when gate is out of the lane.")]
    [SerializeField] private Vector3 _retractedEulerAngles = Vector3.zero;

    [Tooltip("Paddle rotation when gate is blocking the trunk lane.")]
    [SerializeField] private Vector3 _deployedEulerAngles = new Vector3(0f, 0f, 80f);

    // ── Timing ──────────────────────────────────────────────────────
    [Header("Timing")]
    [Tooltip("Seconds for Deploying and Retracting transitions (each).")]
    [SerializeField] private float _transitionDuration = 0.2f;

    [Tooltip("Seconds to hold the Deployed state before retracting.")]
    [SerializeField] private float _deployedHoldDuration = 0.2f;

    [Tooltip("Cooldown at belt speed = 1.0 m/s. Scaled inversely with speed.")]
    [SerializeField] private float _baseCooldown = 0.5f;

    [Tooltip("Minimum cooldown clamp (seconds).")]
    [SerializeField] private float _minCooldown = 0.15f;

    [Tooltip("Maximum cooldown clamp (seconds). Also used to normalise remaining.")]
    [SerializeField] private float _maxCooldown = 1.0f;

    // ── Pre-commit semantics (Option C) ─────────────────────────────
    [Header("Pre-commit Trigger")]
    [Tooltip("Trigger volume placed upstream of the paddle. When the gate is " +
             "Committed and a Package enters this trigger, Deploying begins. " +
             "Should sit ~(beltSpeed × transitionDuration) metres upstream " +
             "so the deploy completes just as the package reaches the paddle.")]
    [SerializeField] private CommitZoneTrigger _commitZone;

    [Tooltip("Maximum time to wait in Committed state before auto-cancelling. " +
             "If no package crosses the commit zone within this window, the " +
             "gate returns to Retracted with no cooldown applied.")]
    [SerializeField] private float _commitTimeout = 3.0f;

    // ── Runtime state ───────────────────────────────────────────────
    private Rigidbody _rb;
    private GateState _state = GateState.Retracted;
    private float _phaseElapsed;
    private float _cooldownRemaining;
    private Quaternion _retractedRot;
    private Quaternion _deployedRot;

    // ── Public accessors ────────────────────────────────────────────
    public GateState CurrentState => _state;
    public int BranchIndex => _branchIndex;

    /// <summary>Remaining cooldown as [0,1], normalised by _maxCooldown.</summary>
    public float NormalisedCooldownRemaining => Mathf.Clamp01(_cooldownRemaining / _maxCooldown);

    /// <summary>True only when Retracted AND cooldown expired.</summary>
    public bool IsActionable => _state == GateState.Retracted && _cooldownRemaining <= 0f;

    /// <summary>
    /// Public accessor for the commit zone trigger. Used by RewardDistributor
    /// for intent shaping subscriptions.
    /// </summary>
    public CommitZoneTrigger CommitZone => _commitZone;

    // ── Unity callbacks ─────────────────────────────────────────────
    private void Awake()
    {
        _rb = GetComponent<Rigidbody>();
        Debug.Assert(_rb.isKinematic,
            $"[DiverterGate] {name} Rigidbody must be Kinematic.");
        Debug.Assert(_commitZone != null,
            $"[DiverterGate] {name} _commitZone is not assigned. Pre-commit " +
            $"semantics require a CommitZoneTrigger child placed upstream of the paddle.",
            this);

        _retractedRot = Quaternion.Euler(_retractedEulerAngles);
        _deployedRot = Quaternion.Euler(_deployedEulerAngles);
        _rb.MoveRotation(transform.parent != null
            ? transform.parent.rotation * _retractedRot
            : _retractedRot);

        if (_commitZone != null)
        {
            _commitZone.OnPackageEntered += HandleCommitZoneEntered;
        }
    }

    private void OnDestroy()
    {
        if (_commitZone != null)
        {
            _commitZone.OnPackageEntered -= HandleCommitZoneEntered;
        }
    }

    private void FixedUpdate()
    {
        float dt = Time.fixedDeltaTime;

        if (_cooldownRemaining > 0f)
            _cooldownRemaining -= dt;

        switch (_state)
        {
            case GateState.Committed:
                // Waiting for a package to cross the commit zone trigger.
                // HandleCommitZoneEntered() advances us to Deploying when one does.
                // Auto-cancel after _commitTimeout if no package arrives.
                _phaseElapsed += dt;
                if (_phaseElapsed >= _commitTimeout)
                {
                    _state = GateState.Retracted;
                    _phaseElapsed = 0f;
                    // No StartCooldown() — nothing physical happened, so no cooldown owed.
                }
                break;

            case GateState.Deploying:
                _phaseElapsed += dt;
                AnimateRotation(_retractedRot, _deployedRot, _phaseElapsed / _transitionDuration);
                if (_phaseElapsed >= _transitionDuration) { _state = GateState.Deployed; _phaseElapsed = 0f; }
                break;

            case GateState.Deployed:
                _phaseElapsed += dt;
                if (_phaseElapsed >= _deployedHoldDuration) { _state = GateState.Retracting; _phaseElapsed = 0f; }
                break;

            case GateState.Retracting:
                _phaseElapsed += dt;
                AnimateRotation(_deployedRot, _retractedRot, _phaseElapsed / _transitionDuration);
                if (_phaseElapsed >= _transitionDuration)
                {
                    _state = GateState.Retracted;
                    _phaseElapsed = 0f;
                    StartCooldown();
                }
                break;
        }
    }

    // ── Public API ──────────────────────────────────────────────────
    /// <summary>
    /// Attempt to activate the gate. Returns true if accepted, false if
    /// ignored (not in Retracted state or cooldown still active).
    /// </summary>
    public bool Activate()
    {
        if (!IsActionable) return false;
        // Pre-commit semantics: enter Committed and wait for a package to
        // cross the commit zone. Deploying starts only when the trigger fires.
        _state = GateState.Committed;
        _phaseElapsed = 0f;
        return true;
    }

    /// <summary>
    /// Called by the CommitZoneTrigger child via its event when a Package
    /// physically enters the commit zone. Only advances to Deploying if we
    /// are currently Committed; otherwise the event is ignored (a package
    /// might cross the zone naturally during cooldown or while a previous
    /// commit is still in transit).
    /// </summary>
    private void HandleCommitZoneEntered(Package pkg)
    {
        if (_state != GateState.Committed) return;
        _state = GateState.Deploying;
        _phaseElapsed = 0f;
    }

    /// <summary>
    /// Hard reset to Retracted with no cooldown. Called by EnvironmentManager on episode reset.
    /// </summary>
    public void ResetToRetracted()
    {
        _state = GateState.Retracted;
        _phaseElapsed = 0f;
        _cooldownRemaining = 0f;
        _rb.MoveRotation(transform.parent != null
            ? transform.parent.rotation * _retractedRot
            : _retractedRot);
    }

    // ── Helpers ─────────────────────────────────────────────────────
    private void AnimateRotation(Quaternion from, Quaternion to, float t)
    {
        t = Mathf.Clamp01(t);
        Quaternion local = Quaternion.Slerp(from, to, t);
        Quaternion world = transform.parent != null ? transform.parent.rotation * local : local;
        _rb.MoveRotation(world);
    }

    private void StartCooldown()
    {
        float speed = BeltSpeedController.Instance != null
            ? BeltSpeedController.Instance.CurrentSpeed
            : 2f;
        _cooldownRemaining = Mathf.Clamp(_baseCooldown / Mathf.Max(speed, 0.1f), _minCooldown, _maxCooldown);
    }
}