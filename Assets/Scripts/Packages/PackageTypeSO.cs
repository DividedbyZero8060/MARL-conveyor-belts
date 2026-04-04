using UnityEngine;

/// <summary>
/// Data asset defining a package type.
/// Create via Assets → Create → MARL → Package Type.
/// Three instances needed: SmallParcel, MediumBox, LargeCrate.
/// </summary>
[CreateAssetMenu(fileName = "NewPackageType", menuName = "MARL/Package Type")]
public class PackageTypeSO : ScriptableObject
{
    [Header("Identity")]
    [Tooltip("Human-readable name shown in debug overlay.")]
    public string displayName;

    [Tooltip("Target destination for this package type.")]
    public DestinationLabel destinationLabel;

    [Header("Physical Properties")]
    [Tooltip("Uniform scale for the cube mesh (meters).")]
    [Min(0.01f)]
    public float dimensions = 0.3f;

    [Tooltip("Rigidbody mass in kg.")]
    [Min(0.01f)]
    public float mass = 1f;

    [Header("Visuals")]
    [Tooltip("Tint colour applied to the MeshRenderer.")]
    public Color colour = Color.white;
}