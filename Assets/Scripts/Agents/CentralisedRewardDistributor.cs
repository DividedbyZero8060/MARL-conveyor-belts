using System;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Team-reward distributor for the Centralised PPO baseline (Step 15c).
///
/// Unlike the cooperative RewardDistributor (which calls AddGroupReward on
/// a SimpleMultiAgentGroup containing 3 agents) or the independent
/// IndependentRewardDistributor (which credits individual agents selfishly),
/// the centralised distributor has a single target: the CentralisedAgent.
///
/// Rules (team totals):
///   - Correct sort     → CentralisedAgent.AddReward(+1.0)
///   - Incorrect sort   → CentralisedAgent.AddReward(-1.5)
///   - Missed package   → CentralisedAgent.AddReward(-2.0)
///
/// These are the same team totals implied by the cooperative distribution
/// (3 × per-agent), just delivered in one place. The reward scale matches
/// SortingAgent.yaml's expectations so the comparison is fair.
///
/// Subscription strategy mirrors the other distributors: per-zone closure
/// captures the branch index at Start, handler refs stored for clean
/// OnDestroy unsubscribe. Implements IEventCounter so DebugOverlay can
/// display its counters via the same interface.
///
/// Place in: Assets/Scripts/Agents/
/// Wire INSTEAD of RewardDistributor/IndependentRewardDistributor for
/// centralised PPO runs. Not combined — that would double-count events.
/// </summary>
public class CentralisedRewardDistributor : MonoBehaviour, IEventCounter
{
    [Header("Wired References")]
    [Tooltip("The single CentralisedAgent that receives all team rewards.")]
    [SerializeField] private CentralisedAgent _centralisedAgent;

    [Tooltip("All 4 destination zones (3 branch zones + 1 fallthrough). " +
             "Order must match _zoneBranchIndices.")]
    [SerializeField] private DestinationZone[] _destinationZones = new DestinationZone[4];

    [Tooltip("Branch index for each zone in _destinationZones. Use -1 for fallthrough. " +
             "Matches the wiring convention of the cooperative RewardDistributor.")]
    [SerializeField] private int[] _zoneBranchIndices = new int[4];

    // Sacred team reward magnitudes. These are the full team totals
    // (equivalent to per-agent × 3 in the cooperative distributor).
    private const float CorrectSortTeamReward = 1.0f;
    private const float IncorrectSortTeamReward = -1.5f;
    private const float MissedPackageTeamReward = -2.0f;

    // Per-episode event counters for IEventCounter interface parity.
    public int CorrectSortEvents { get; private set; }
    public int IncorrectSortEvents { get; private set; }
    public int MissedPackageEvents { get; private set; }

    public int GetCorrectSortsForBranch(int branchIndex)
    {
        if (branchIndex < 0 || branchIndex >= _correctSortsByBranch.Length) return 0;
        return _correctSortsByBranch[branchIndex];
    }

    private readonly int[] _correctSortsByBranch = new int[3];

    // Closure-captured handlers stored for OnDestroy unsubscribe by reference.
    private readonly List<Action<Package>> _correctSortHandlers = new List<Action<Package>>();
    private readonly List<Action<Package>> _incorrectSortHandlers = new List<Action<Package>>();

    private void Awake()
    {
        Debug.Assert(_centralisedAgent != null,
            "[CentralisedRewardDistributor] _centralisedAgent not assigned.", this);
        Debug.Assert(_destinationZones != null && _destinationZones.Length > 0,
            "[CentralisedRewardDistributor] _destinationZones not assigned.", this);
        Debug.Assert(_zoneBranchIndices != null && _zoneBranchIndices.Length == _destinationZones.Length,
            "[CentralisedRewardDistributor] _zoneBranchIndices must match _destinationZones length.", this);

        // Sacred-number assertions. Catch accidental drift from spec.
        Debug.Assert(Mathf.Approximately(CorrectSortTeamReward, 1.0f),
            "[CentralisedRewardDistributor] CorrectSort reward drifted from spec (+1.0).");
        Debug.Assert(Mathf.Approximately(IncorrectSortTeamReward, -1.5f),
            "[CentralisedRewardDistributor] IncorrectSort reward drifted from spec (-1.5).");
        Debug.Assert(Mathf.Approximately(MissedPackageTeamReward, -2.0f),
            "[CentralisedRewardDistributor] Missed reward drifted from spec (-2.0).");
    }

    private void Start()
    {
        for (int i = 0; i < _destinationZones.Length; i++)
        {
            DestinationZone zone = _destinationZones[i];
            if (zone == null)
            {
                _correctSortHandlers.Add(null);
                _incorrectSortHandlers.Add(null);
                continue;
            }

            if (zone.IsFallthrough)
            {
                zone.OnMissedPackage += HandleMissedPackage;
                _correctSortHandlers.Add(null);
                _incorrectSortHandlers.Add(null);
            }
            else
            {
                int capturedBranchIndex = _zoneBranchIndices[i];
                Action<Package> correctHandler = (pkg) => HandleCorrectSortForBranch(pkg, capturedBranchIndex);
                Action<Package> incorrectHandler = (pkg) => HandleIncorrectSort(pkg);
                zone.OnCorrectSort += correctHandler;
                zone.OnIncorrectSort += incorrectHandler;
                _correctSortHandlers.Add(correctHandler);
                _incorrectSortHandlers.Add(incorrectHandler);
            }
        }

        if (EnvironmentManager.Instance != null)
        {
            EnvironmentManager.Instance.OnEpisodeReset += HandleEpisodeReset;
        }
        else
        {
            Debug.LogError(
                "[CentralisedRewardDistributor] EnvironmentManager.Instance is null in Start.", this);
        }
    }

    private void OnDestroy()
    {
        for (int i = 0; i < _destinationZones.Length; i++)
        {
            DestinationZone zone = _destinationZones[i];
            if (zone == null) continue;

            if (zone.IsFallthrough)
            {
                zone.OnMissedPackage -= HandleMissedPackage;
            }
            else
            {
                if (i < _correctSortHandlers.Count && _correctSortHandlers[i] != null)
                    zone.OnCorrectSort -= _correctSortHandlers[i];
                if (i < _incorrectSortHandlers.Count && _incorrectSortHandlers[i] != null)
                    zone.OnIncorrectSort -= _incorrectSortHandlers[i];
            }
        }

        if (EnvironmentManager.Instance != null)
        {
            EnvironmentManager.Instance.OnEpisodeReset -= HandleEpisodeReset;
        }
    }

    // ── Event handlers ──────────────────────────────────────────────

    private void HandleCorrectSortForBranch(Package pkg, int branchIndex)
    {
        CorrectSortEvents++;
        if (branchIndex >= 0 && branchIndex < _correctSortsByBranch.Length)
            _correctSortsByBranch[branchIndex]++;

        if (_centralisedAgent != null)
            _centralisedAgent.AddReward(CorrectSortTeamReward);
    }

    private void HandleIncorrectSort(Package pkg)
    {
        IncorrectSortEvents++;
        if (_centralisedAgent != null)
            _centralisedAgent.AddReward(IncorrectSortTeamReward);
    }

    private void HandleMissedPackage(Package pkg)
    {
        MissedPackageEvents++;
        if (_centralisedAgent != null)
            _centralisedAgent.AddReward(MissedPackageTeamReward);
    }

    private void HandleEpisodeReset()
    {
        CorrectSortEvents = 0;
        IncorrectSortEvents = 0;
        MissedPackageEvents = 0;
        for (int i = 0; i < _correctSortsByBranch.Length; i++)
            _correctSortsByBranch[i] = 0;
    }
}