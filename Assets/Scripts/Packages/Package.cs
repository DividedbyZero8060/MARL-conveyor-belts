using UnityEngine;

/// <summary>
/// Runtime component on each package prefab instance.
/// Call <see cref="Initialise"/> to configure from a <see cref="PackageTypeSO"/>.
/// Call <see cref="ReturnToPool"/> to deactivate and zero physics state.
/// </summary>
[RequireComponent(typeof(Rigidbody))]
[RequireComponent(typeof(MeshRenderer))]
public class Package : MonoBehaviour
{
    // ── Cached references ───────────────────────────────────────────
    private Rigidbody _rb;
    private MeshRenderer _meshRenderer;
    private MaterialPropertyBlock _mpb;

    // ── Public state ────────────────────────────────────────────────
    /// <summary>The ScriptableObject this package was configured with.</summary>
    public PackageTypeSO PackageType { get; private set; }

    /// <summary>Destination label shortcut (avoids PackageType null checks).</summary>
    public DestinationLabel DestinationLabel { get; private set; }

    private void Awake()
    {
        _rb = GetComponent<Rigidbody>();
        _meshRenderer = GetComponent<MeshRenderer>();
        _mpb = new MaterialPropertyBlock();

        Debug.Assert(_rb != null, $"[Package] {name} missing Rigidbody.");
        Debug.Assert(_meshRenderer != null, $"[Package] {name} missing MeshRenderer.");

        // Rigidbody defaults (per workflow spec)
        _rb.useGravity = true;
        _rb.drag = 1.0f;
        _rb.interpolation = RigidbodyInterpolation.Interpolate;
        _rb.collisionDetectionMode = CollisionDetectionMode.ContinuousDynamic;
        _rb.constraints = RigidbodyConstraints.FreezeRotation;
    }

    /// <summary>
    /// Configure this package instance from a type definition.
    /// Called by <see cref="PackageSpawner"/> when activating from pool.
    /// </summary>
    public void Initialise(PackageTypeSO type)
    {
        Debug.Assert(type != null, $"[Package] {name} initialised with null type.");

        PackageType = type;
        DestinationLabel = type.destinationLabel;

        // Scale: uniform cube
        float d = type.dimensions;
        transform.localScale = new Vector3(d, d, d);

        // Mass
        _rb.mass = type.mass;

        // Colour via MaterialPropertyBlock (no material instance created)
        _meshRenderer.GetPropertyBlock(_mpb);
        _mpb.SetColor("_BaseColor", type.colour);
        _meshRenderer.SetPropertyBlock(_mpb);
    }

    /// <summary>
    /// Deactivate and zero all physics state before returning to pool.
    /// </summary>
    public void ReturnToPool()
    {
        _rb.velocity = Vector3.zero;
        _rb.angularVelocity = Vector3.zero;
        gameObject.SetActive(false);
    }
}