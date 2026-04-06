using UnityEngine;

/// <summary>Gate FSM states. Activate() is only valid in Retracted.</summary>
public enum GateState { Retracted, Deploying, Deployed, Retracting }

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

    // ── Unity callbacks ─────────────────────────────────────────────
    private void Awake()
    {
        _rb = GetComponent<Rigidbody>();
        Debug.Assert(_rb.isKinematic,
            $"[DiverterGate] {name} Rigidbody must be Kinematic.");

        _retractedRot = Quaternion.Euler(_retractedEulerAngles);
        _deployedRot = Quaternion.Euler(_deployedEulerAngles);
        _rb.MoveRotation(transform.parent != null
            ? transform.parent.rotation * _retractedRot
            : _retractedRot);
    }

    private void FixedUpdate()
    {
        float dt = Time.fixedDeltaTime;

        if (_cooldownRemaining > 0f)
            _cooldownRemaining -= dt;

        switch (_state)
        {
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
        _state = GateState.Deploying;
        _phaseElapsed = 0f;
        return true;
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