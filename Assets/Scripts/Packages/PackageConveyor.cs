using UnityEngine;

/// <summary>
/// Applies a capped acceleration force to push packages along the belt's
/// forward direction.  Uses OnCollisionStay/OnCollisionExit with
/// "BeltSurface"-tagged colliders to track whether the package is on a belt.
///
/// Attach to the same GameObject as <see cref="Package"/>.
/// All physics logic runs in FixedUpdate — NEVER Update.
/// Uses ForceMode.Acceleration — NEVER MovePosition, VelocityChange, or
/// direct rb.velocity assignment.
/// </summary>
[RequireComponent(typeof(Rigidbody))]
public class PackageConveyor : MonoBehaviour
{
    // ── Configuration ───────────────────────────────────────────────
    [Header("Belt Force")]
    [Tooltip("Maximum acceleration applied along belt forward (m/s²). " +
             "Reduce to 10–15 if packages vibrate behind a gate.")]
    [SerializeField] private float _maxForce = 20f;

    [Tooltip("Proportional gain multiplier to overcome Rigidbody drag.")]
    [SerializeField] private float _forceGain = 50f;

    // ── Runtime state ───────────────────────────────────────────────
    private Rigidbody _rb;
    private bool _isOnBelt;
    private BeltSurface _currentBelt;

    private void Awake()
    {
        _rb = GetComponent<Rigidbody>();
        Debug.Assert(_rb != null, $"[PackageConveyor] {name} missing Rigidbody.");
    }

    /// <summary>
    /// Reset conveyor state when package is returned to pool or re-spawned.
    /// Called externally if needed, but also resets naturally via OnCollisionExit
    /// when the package is deactivated.
    /// </summary>
    public void ResetConveyorState()
    {
        _isOnBelt = false;
        _currentBelt = null;
    }

    // ── Physics ─────────────────────────────────────────────────────
    private void FixedUpdate()
    {
        
        if (!_isOnBelt || _currentBelt == null) return;

        float beltSpeed = BeltSpeedController.Instance.CurrentSpeed;
        Vector3 beltForward = _currentBelt.WorldForward;

        // Current speed along belt forward direction
        float currentForwardSpeed = Vector3.Dot(_rb.velocity, beltForward);

        // Speed difference: how much faster the belt is than the package
        float speedDiff = beltSpeed - currentForwardSpeed;

        // Clamp to max force and apply as acceleration (mass-independent)
        float force = Mathf.Clamp(speedDiff * _forceGain, -_maxForce, _maxForce);
        _rb.AddForce(force * beltForward, ForceMode.Acceleration);

    }

    // ── Collision tracking ──────────────────────────────────────────
    // OnCollisionStay fires every physics frame while touching.
    // This keeps _isOnBelt true and updates _currentBelt if the package
    // transitions from trunk to branch (or vice versa).

    private void OnCollisionStay(Collision collision)
    {
        if (!collision.gameObject.CompareTag("BeltSurface")) return;

        BeltSurface belt = collision.gameObject.GetComponent<BeltSurface>();
        if (belt == null) return;

        _isOnBelt = true;
        _currentBelt = belt;
    }

    // OnCollisionExit fires when the package lifts off or leaves the belt.
    // Only clear state if we're leaving a BeltSurface — collisions with
    // rails or other packages should not reset belt tracking.

    private void OnCollisionExit(Collision collision)
    {
        if (!collision.gameObject.CompareTag("BeltSurface")) return;

        // Only clear if we're exiting the belt we were tracking
        BeltSurface belt = collision.gameObject.GetComponent<BeltSurface>();
        if (belt == _currentBelt)
        {
            _isOnBelt = false;
            _currentBelt = null;
        }
    }

    /// <summary>
    /// OnCollisionExit does NOT fire when SetActive(false) is called.
    /// This ensures conveyor state resets when a package returns to pool.
    /// </summary>
    private void OnDisable()
    {
        _isOnBelt = false;
        _currentBelt = null;
    }
}