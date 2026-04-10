using System.Collections.Generic;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

/// <summary>
/// Sorting agent skeleton (Step 08, file 4b).
///
/// Owns:
///  - References to its DiverterGate, BranchTracker, PackageDetector, and the
///    other two SortingAgents in the scene (peer references for full-obs mode).
///  - The trigger overlap set for PackageDetector. OnTriggerEnter/Exit on this
///    GameObject populate _overlappingPackages, which is passed into
///    PackageDetector.Refresh() during CollectObservations (4c).
///  - Public read-only properties used by peer agents in the full-obs branch:
///       NormalisedGateState     (0 / 0.5 / 1)
///       NearestPackageDistance  (normalised [0, 1], 1 = none in range)
///
/// Does NOT own (yet):
///  - CollectObservations body — implemented in 4c (SortingAgent.Observations.cs).
///  - Real action handling — Step 09. OnActionReceived is currently a no-op stub.
///  - Reward distribution — Step 10.
/// </summary>
[RequireComponent(typeof(BoxCollider))]
public partial class SortingAgent : Agent
{
    [Header("Agent Identity")]
    [Tooltip("Branch index this agent controls (0, 1, or 2). Must match its DiverterGate.BranchIndex.")]
    [SerializeField] private int _branchIndex;

    [Header("Wired References")]
    [Tooltip("The DiverterGate this agent controls.")]
    [SerializeField] private DiverterGate _gate;

    [Tooltip("The BranchTracker for this agent's branch (used for self-congestion sanity).")]
    [SerializeField] private BranchTracker _ownBranchTracker;

    [Tooltip("All three BranchTrackers in scene order (Branch 0, 1, 2). Used for the 3-float congestion observation.")]
    [SerializeField] private BranchTracker[] _allBranchTrackers = new BranchTracker[3];

    [Tooltip("The PackageDetector child component on this agent.")]
    [SerializeField] private PackageDetector _packageDetector;

    [Tooltip("The other two SortingAgents in scene (peer features, full-obs only). Leave empty in partial-obs mode.")]
    [SerializeField] private SortingAgent[] _peerAgents = new SortingAgent[2];

    [Header("Belt Forward (for detector)")]
    [Tooltip("World-space forward direction along the trunk belt. Passed to PackageDetector at Initialize.")]
    [SerializeField] private Vector3 _beltForward = Vector3.forward;

    // Live set of packages currently inside this agent's detector trigger volume.
    // Maintained by OnTriggerEnter / OnTriggerExit on the agent GameObject.
    
    private readonly List<Package> _overlappingPackages = new List<Package>(32);

    // Cached for peer queries — recomputed at the end of each CollectObservations
    // in 4c so peers reading it via NearestPackageDistance get a consistent value.
    private float _cachedNearestPackageDistance = 1f;

    // Per-episode activation telemetry (consumed by DebugOverlay at episode end).
    private int _activationCount;      // times OnActionReceived decided to activate the gate
    private int _decisionCount;         // total OnActionReceived calls this episode
    // ================================================================
    // Public read-only properties for peer agents (full-obs mode, 4c)
    // ================================================================

    /// <summary>Branch index (0, 1, 2).</summary>
    public int BranchIndex => _branchIndex;

    /// <summary>
    /// Gate state encoded for the observation vector:
    /// 0.0 = Retracted, 0.5 = Deploying or Retracting, 1.0 = Deployed.
    /// </summary>
    public float NormalisedGateState
    {
        get
        {
            if (_gate == null) return 0f;
            switch (_gate.CurrentState)
            {
                case GateState.Retracted: return 0f;
                case GateState.Deployed: return 1f;
                case GateState.Deploying:
                case GateState.Retracting: return 0.5f;
                default: return 0f;
            }
        }
    }

    /// <summary>
    /// Distance to the nearest in-range package along the belt, normalised to [0, 1].
    /// 1.0 means no package detected. Updated at the end of each CollectObservations call.
    /// </summary>
    public float NearestPackageDistance => _cachedNearestPackageDistance;

    // ================================================================
    // Lifecycle
    // ================================================================

    public override void Initialize()
    {
        Debug.Assert(_branchIndex >= 0 && _branchIndex <= 2,
            $"[SortingAgent] branchIndex must be in [0,2], got {_branchIndex}.", this);
        Debug.Assert(_gate != null,
            $"[SortingAgent {_branchIndex}] _gate is not assigned.", this);
        Debug.Assert(_ownBranchTracker != null,
            $"[SortingAgent {_branchIndex}] _ownBranchTracker is not assigned.", this);
        Debug.Assert(_packageDetector != null,
            $"[SortingAgent {_branchIndex}] _packageDetector is not assigned.", this);
        Debug.Assert(_allBranchTrackers != null && _allBranchTrackers.Length == 3,
            $"[SortingAgent {_branchIndex}] _allBranchTrackers must have exactly 3 entries.", this);
        for (int i = 0; i < 3; i++)
        {
            Debug.Assert(_allBranchTrackers[i] != null,
                $"[SortingAgent {_branchIndex}] _allBranchTrackers[{i}] is null.", this);
        }
        Debug.Assert(_peerAgents != null && _peerAgents.Length == 2,
            $"[SortingAgent {_branchIndex}] _peerAgents must have exactly 2 entries.", this);

        // Sanity: gate's branch index should match our own.
        if (_gate != null && _gate.BranchIndex != _branchIndex)
        {
            Debug.LogWarning(
                $"[SortingAgent {_branchIndex}] gate.BranchIndex ({_gate.BranchIndex}) " +
                $"does not match agent branchIndex ({_branchIndex}).", this);
        }

        // Force trigger mode on this agent's BoxCollider — it's the detector volume.
        BoxCollider box = GetComponent<BoxCollider>();
        if (!box.isTrigger)
        {
            Debug.LogWarning(
                $"[SortingAgent {_branchIndex}] BoxCollider was not set to isTrigger. Forcing true.", this);
            box.isTrigger = true;
        }

        // Push belt forward into the detector once.
        _packageDetector.SetBeltForward(_beltForward);

        // detect discrete vs continuous action mode from BehaviorParameters.
        DetectActionMode();
    }

    private void Start()
    {
        if (EnvironmentManager.Instance != null)
        {
            EnvironmentManager.Instance.OnEpisodeReset += HandleEnvironmentReset;
        }
        else
        {
            Debug.LogError($"[SortingAgent {_branchIndex}] EnvironmentManager.Instance is null in Start!", this);
        }
    }

    private void OnDestroy()
    {
        if (EnvironmentManager.Instance != null)
        {
            EnvironmentManager.Instance.OnEpisodeReset -= HandleEnvironmentReset;
        }
    }

    private void HandleEnvironmentReset()
    {
        _overlappingPackages.Clear();
        _cachedNearestPackageDistance = 1f;
        if (_packageDetector != null) _packageDetector.ResetState();
    }


    public override void OnEpisodeBegin()
    {
        _overlappingPackages.Clear();
        _cachedNearestPackageDistance = 1f;
        if (_packageDetector != null) _packageDetector.ResetState();
        _activationCount = 0;
        _decisionCount = 0;
    }

    // ================================================================
    // Trigger handling — populates the detector overlap set
    // ================================================================

    private void OnTriggerEnter(Collider other)
    {
        Package pkg = other.GetComponentInParent<Package>();
        if (pkg == null) return;
        if (!_overlappingPackages.Contains(pkg))
        {
            _overlappingPackages.Add(pkg);
        }
    }

    private void OnTriggerExit(Collider other)
    {
        Package pkg = other.GetComponentInParent<Package>();
        if (pkg == null) return;
        _overlappingPackages.Remove(pkg);
    }

    // ================================================================
    // Observation / action stubs
    // (CollectObservations is implemented in 4c via partial class)
    // ================================================================

    

    

    

    // ================================================================
    // Internal helpers exposed to the partial-class file in 4c
    // ================================================================

    /// <summary>
    /// Called by the 4c CollectObservations partial after Refresh, so peers
    /// reading NearestPackageDistance see the value computed this same step.
    /// </summary>
    internal void SetCachedNearestPackageDistance(float normalisedDistance)
    {
        _cachedNearestPackageDistance = normalisedDistance;
    }

    /// <summary>
    /// Consumes and resets the per-episode gate activation counters.
    /// Returns the activation rate: activations / decisions, in [0, 1].
    /// Called by DebugOverlay during HandleEpisodeEnded.
    /// </summary>
    public float ConsumeAndResetActivationRate()
    {
        float rate = _decisionCount > 0
            ? (float)_activationCount / _decisionCount
            : 0f;
        _activationCount = 0;
        _decisionCount = 0;
        return rate;
    }

    /// <summary>4c reads this directly to drive PackageDetector.Refresh().</summary>
    internal IReadOnlyList<Package> OverlappingPackages => _overlappingPackages;

    /// <summary>4c reads gate, own branch tracker, all branch trackers, peers.</summary>
    internal DiverterGate Gate => _gate;
    internal BranchTracker OwnBranchTracker => _ownBranchTracker;
    internal BranchTracker[] AllBranchTrackers => _allBranchTrackers;
    internal PackageDetector Detector => _packageDetector;
    internal SortingAgent[] PeerAgents => _peerAgents;



#if UNITY_EDITOR
    private void OnDrawGizmosSelected()
    {
        BoxCollider box = GetComponent<BoxCollider>();
        if (box == null) return;
        Gizmos.color = new Color(0f, 1f, 0f, 0.15f);
        Gizmos.matrix = transform.localToWorldMatrix;
        Gizmos.DrawCube(box.center, box.size);
        Gizmos.color = Color.green;
        Gizmos.DrawWireCube(box.center, box.size);
    }
#endif


}