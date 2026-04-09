using System;
using UnityEngine;

/// <summary>
/// Global episode manager (singleton). Owns episode timing, destination
/// shuffling, counter tracking, and the reset lifecycle.
///
/// Per-episode shuffle: a random permutation of {DestA, DestB, DestC} is
/// assigned to the 3 branch zones via DestinationZone.SetAcceptedLabel().
/// Agents query <see cref="GetDestinationForBranch"/> to observe the mapping.
///
/// Episode ends when either _episodeDuration elapses OR MissedPackages
/// reaches _maxMissedPackages. On end, ResetEpisode() runs automatically.
///
/// Place in: Assets/Scripts/Environment/
/// </summary>
public class EnvironmentManager : MonoBehaviour
{
    public static EnvironmentManager Instance { get; private set; }

    // ── Configuration ───────────────────────────────────────────────
    [Header("Episode Parameters")]
    [Tooltip("Episode duration in seconds.")]
    [SerializeField] private float _episodeDuration = 60f;

    [Tooltip("Episode ends early if this many packages are missed.")]
    [SerializeField] private int _maxMissedPackages = 10;

    // ── Scene references ────────────────────────────────────────────
    [Header("Scene References (size 3 for branches)")]
    [Tooltip("Destination zones at the 3 branch endpoints, in branch order.")]
    [SerializeField] private DestinationZone[] _branchZones = new DestinationZone[3];

    [Tooltip("Diverter gates at the 3 branch junctions, in branch order.")]
    [SerializeField] private DiverterGate[] _branchGates = new DiverterGate[3];

    [Tooltip("The trunk-end fallthrough zone (isFallthrough=true).")]
    [SerializeField] private DestinationZone _fallthroughZone;

    [Tooltip("The package spawner whose pool is reset between episodes.")]
    [SerializeField] private PackageSpawner _packageSpawner;

    [Header("Observation Mode")]
    [Tooltip("If true, agents observe 34 floats (no peer-agent features). If false, 38 floats. Used by SortingAgent.")]
    [SerializeField] private bool _partialObservability = false;

    [Header("Agent Group")]
    [Tooltip("The cooperative agent group. Its episode is ended atomically during environment reset.")]
    [SerializeField] private SortingAgentGroup _agentGroup;

    // ── Counters ────────────────────────────────────────────────────

    /// <summary>
    /// Whether agents should use the partial-observability observation layout (34 floats).
    /// False = full observability (38 floats). Used by SortingAgent.CollectObservations.
    /// </summary>
    public bool PartialObservability => _partialObservability;
    public int CorrectSorts { get; private set; }
    public int IncorrectSorts { get; private set; }
    public int MissedPackages { get; private set; }
    public int TotalSpawned => _packageSpawner != null ? _packageSpawner.TotalSpawned : 0;

    // ── Episode state ───────────────────────────────────────────────
    public int EpisodeIndex { get; private set; }
    public float EpisodeTimeRemaining => Mathf.Max(0f, _episodeDuration - _elapsed);

    // ── Events ──────────────────────────────────────────────────────
    public event Action OnEpisodeReset;
    public event Action OnEpisodeEnded;

    // ── Runtime state ───────────────────────────────────────────────
    private float _elapsed;
    private readonly DestinationLabel[] _branchDestinations = new DestinationLabel[3];
    private static readonly DestinationLabel[] AllLabels =
        { DestinationLabel.DestA, DestinationLabel.DestB, DestinationLabel.DestC };

    // ── Unity callbacks ─────────────────────────────────────────────
    private void Awake()
    {
        if (Instance != null && Instance != this) { Destroy(gameObject); return; }
        Instance = this;

        Debug.Assert(_branchZones != null && _branchZones.Length == 3,
            "[EnvironmentManager] Requires exactly 3 branch zones.");
        Debug.Assert(_branchGates != null && _branchGates.Length == 3,
            "[EnvironmentManager] Requires exactly 3 branch gates.");
        Debug.Assert(_fallthroughZone != null,
            "[EnvironmentManager] Requires fallthrough zone reference.");
        Debug.Assert(_packageSpawner != null,
            "[EnvironmentManager] Requires PackageSpawner reference.");
    }

    private void Start()
    {
        // Subscribe to zone events for counter tracking.
        // (Reward distribution is wired up separately in Step 10.)
        for (int i = 0; i < 3; i++)
        {
            _branchZones[i].OnCorrectSort += OnCorrectSortHandler;
            _branchZones[i].OnIncorrectSort += OnIncorrectSortHandler;
        }
        _fallthroughZone.OnMissedPackage += OnMissedPackageHandler;

        ResetEpisode();
    }

    private void OnDestroy()
    {
        if (_branchZones != null)
        {
            for (int i = 0; i < _branchZones.Length; i++)
            {
                if (_branchZones[i] == null) continue;
                _branchZones[i].OnCorrectSort -= OnCorrectSortHandler;
                _branchZones[i].OnIncorrectSort -= OnIncorrectSortHandler;
            }
        }
        if (_fallthroughZone != null)
            _fallthroughZone.OnMissedPackage -= OnMissedPackageHandler;
    }

    private void Update()
    {
        _elapsed += Time.deltaTime;
        if (_elapsed >= _episodeDuration)
            EndEpisode();
    }

    // ── Public API ──────────────────────────────────────────────────
    /// <summary>Returns the destination label currently assigned to the given branch [0..2].</summary>
    public DestinationLabel GetDestinationForBranch(int branchIndex)
    {
        Debug.Assert(branchIndex >= 0 && branchIndex < 3,
            $"[EnvironmentManager] Branch index {branchIndex} out of range.");
        return _branchDestinations[branchIndex];
    }



    /// <summary>
    /// Full episode reset: zero counters, pool all packages, retract all gates,
    /// shuffle destinations, fire OnEpisodeReset.
    /// </summary>
    [ContextMenu("Reset Episode")]
    public void ResetEpisode()
    {
        CorrectSorts = 0;
        IncorrectSorts = 0;
        MissedPackages = 0;
        _elapsed = 0f;
        EpisodeIndex++;

        _packageSpawner.ReturnAllToPool();
        for (int i = 0; i < _branchGates.Length; i++)
            _branchGates[i].ResetToRetracted();

        ShuffleDestinations();

        OnEpisodeReset?.Invoke();

        if (_agentGroup != null)
        {
            _agentGroup.EndEpisodeForAll();
        }
    }

    // ── Event handlers ──────────────────────────────────────────────
    private void OnCorrectSortHandler(Package _) { CorrectSorts++; }
    private void OnIncorrectSortHandler(Package _) { IncorrectSorts++; }
    private void OnMissedPackageHandler(Package _)
    {
        MissedPackages++;
        if (MissedPackages >= _maxMissedPackages) EndEpisode();
    }

    // ── Internal ────────────────────────────────────────────────────
    private void EndEpisode()
    {
        OnEpisodeEnded?.Invoke();
        ResetEpisode();
    }

    private void ShuffleDestinations()
    {
        // Fisher-Yates shuffle on a copy of AllLabels.
        var perm = new DestinationLabel[AllLabels.Length];
        Array.Copy(AllLabels, perm, AllLabels.Length);
        for (int i = perm.Length - 1; i > 0; i--)
        {
            int j = UnityEngine.Random.Range(0, i + 1);
            (perm[i], perm[j]) = (perm[j], perm[i]);
        }

        for (int i = 0; i < 3; i++)
        {
            _branchDestinations[i] = perm[i];
            _branchZones[i].SetAcceptedLabel(perm[i]);
        }
    }
}