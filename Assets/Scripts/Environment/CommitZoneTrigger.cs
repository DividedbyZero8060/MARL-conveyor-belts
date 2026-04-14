using System;
using UnityEngine;

/// <summary>
/// Companion trigger for DiverterGate's pre-commit semantics.
///
/// Place as a child GameObject under each DiverterGate. The transform
/// position should sit upstream of the paddle's deflection point by
/// approximately (belt_speed × transition_duration) metres, so a package
/// that crosses this trigger reaches the paddle just as the deploy completes.
///
/// At belt speed 2 m/s with transition duration 0.2s, that is 0.4 m upstream.
///
/// The trigger fires the OnPackageEntered event whenever a Package enters its
/// volume. DiverterGate subscribes to this event and advances from Committed
/// to Deploying only when the gate is currently in the Committed state. The
/// trigger fires unconditionally on every package — gating logic lives in
/// DiverterGate.
/// </summary>
[RequireComponent(typeof(BoxCollider))]
public class CommitZoneTrigger : MonoBehaviour
{
    /// <summary>Fired when a Package enters this trigger volume.</summary>
    public event Action<Package> OnPackageEntered;

    private void Awake()
    {
        BoxCollider box = GetComponent<BoxCollider>();
        if (!box.isTrigger)
        {
            Debug.LogWarning(
                $"[CommitZoneTrigger] '{name}' BoxCollider was not set to isTrigger. Forcing true.",
                this);
            box.isTrigger = true;
        }
    }

    private void OnTriggerEnter(Collider other)
    {
        Package pkg = other.GetComponentInParent<Package>();
        if (pkg == null) return;
        OnPackageEntered?.Invoke(pkg);
    }

#if UNITY_EDITOR
    private void OnDrawGizmosSelected()
    {
        BoxCollider box = GetComponent<BoxCollider>();
        if (box == null) return;
        Gizmos.color = new Color(1f, 0.5f, 0f, 0.25f);
        Gizmos.matrix = transform.localToWorldMatrix;
        Gizmos.DrawCube(box.center, box.size);
        Gizmos.color = new Color(1f, 0.5f, 0f, 1f);
        Gizmos.DrawWireCube(box.center, box.size);
    }
#endif
}