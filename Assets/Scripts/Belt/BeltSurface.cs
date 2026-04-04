using UnityEngine;

/// <summary>
/// Identifies a GameObject as a belt segment and stores its constant forward
/// conveying direction in world space.  Attach to every trunk and branch belt.
/// Tag the GameObject as "BeltSurface" so PackageConveyor (Step 05) can detect it.
/// </summary>
public class BeltSurface : MonoBehaviour
{
    [Header("Belt Direction")]
    [Tooltip("Local-space forward axis used for conveying. " +
             "Default (0,0,1) means the object's blue Z-arrow.")]
    [SerializeField] private Vector3 _localForward = Vector3.forward;

    /// <summary>World-space forward direction, recalculated every physics step.</summary>
    public Vector3 WorldForward { get; private set; }

    private void Awake()
    {
        Debug.Assert(gameObject.CompareTag("BeltSurface"),
            $"[BeltSurface] {name} must be tagged 'BeltSurface'.");
        Debug.Assert(_localForward.sqrMagnitude > 0.001f,
            $"[BeltSurface] {name} has near-zero _localForward.");

        RecalculateWorldForward();
    }

    private void FixedUpdate()
    {
        // Belts are static in this prototype, but recalculating is cheap
        // and future-proofs against runtime rotation.
        RecalculateWorldForward();
    }

    private void RecalculateWorldForward()
    {
        WorldForward = transform.TransformDirection(_localForward).normalized;
    }

    // ── Editor helpers ──────────────────────────────────────────────
    private void OnDrawGizmosSelected()
    {
        Vector3 origin = transform.position + Vector3.up * 0.15f;
        Vector3 dir = transform.TransformDirection(_localForward).normalized;
        Gizmos.color = Color.cyan;
        Gizmos.DrawRay(origin, dir * 2f);
    }
}