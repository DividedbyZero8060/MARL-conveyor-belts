using System;
using UnityEngine;

/// <summary>
/// Trigger zone placed at the end of each branch (scored) or at the
/// trunk end (fallthrough / missed).
///
/// When a <see cref="Package"/> enters the trigger:
///   • Branch zone — compares labels → fires <see cref="OnCorrectSort"/>
///     or <see cref="OnIncorrectSort"/>.
///   • Fallthrough zone (<see cref="_isFallthrough"/> = true) → fires
///     <see cref="OnMissedPackage"/>.
///
/// After any event the package's Rigidbody velocity is zeroed and
/// it is returned to the object pool via <see cref="Package.ReturnToPool"/>.
///
/// Place in: Assets/Scripts/Environment/
/// </summary>
[RequireComponent(typeof(BoxCollider))]
public class DestinationZone : MonoBehaviour
{
    // ── Configuration ───────────────────────────────────────────────
    [Header("Zone Identity")]
    [Tooltip("Which destination this zone accepts as correct. " +
             "Ignored when Is Fallthrough is true.")]
    [SerializeField] private DestinationLabel _acceptedLabel;

    [Tooltip("If true, this zone is the trunk-end fallthrough. " +
             "Any package entering fires OnMissedPackage regardless of label.")]
    [SerializeField] private bool _isFallthrough;

    // ── Events ──────────────────────────────────────────────────────
    /// <summary>Fired when a package with the matching label enters a branch zone.</summary>
    public event Action<Package> OnCorrectSort;

    /// <summary>Fired when a package with a non-matching label enters a branch zone.</summary>
    public event Action<Package> OnIncorrectSort;

    /// <summary>Fired when any package enters the trunk-end fallthrough zone.</summary>
    public event Action<Package> OnMissedPackage;

    // ── Public accessors ────────────────────────────────────────────
    /// <summary>The destination label this zone scores against.</summary>
    public DestinationLabel AcceptedLabel => _acceptedLabel;

    /// <summary>Whether this is the trunk-end fallthrough zone.</summary>
    public bool IsFallthrough => _isFallthrough;

    // ── Mono callbacks ──────────────────────────────────────────────
    private void Awake()
    {
        BoxCollider col = GetComponent<BoxCollider>();
        Debug.Assert(col.isTrigger,
            $"[DestinationZone] {name} BoxCollider must have isTrigger = true.");
    }

    private void OnTriggerEnter(Collider other)
    {
        // Only react to packages.
        Package pkg = other.GetComponent<Package>();
        if (pkg == null) return;

        if (_isFallthrough)
        {
            OnMissedPackage?.Invoke(pkg);
        }
        else if (pkg.DestinationLabel == _acceptedLabel)
        {
            OnCorrectSort?.Invoke(pkg);
        }
        else
        {
            OnIncorrectSort?.Invoke(pkg);
        }

        RecyclePackage(pkg);
    }

    // ── Helpers ─────────────────────────────────────────────────────
    /// <summary>
    /// Zero velocity, zero angular velocity, then return to pool.
    /// Must zero BEFORE ReturnToPool so the package is clean when reused.
    /// </summary>
    private static void RecyclePackage(Package pkg)
    {
        Rigidbody rb = pkg.GetComponent<Rigidbody>();
        if (rb != null)
        {
            rb.velocity = Vector3.zero;
            rb.angularVelocity = Vector3.zero;
        }

        pkg.ReturnToPool();
    }

    /// <summary>
    /// Reassigns this zone's accepted label.  Called by EnvironmentManager
    /// during per-episode destination shuffling.
    /// </summary>
    public void SetAcceptedLabel(DestinationLabel label)
    {
        _acceptedLabel = label;
    }
}