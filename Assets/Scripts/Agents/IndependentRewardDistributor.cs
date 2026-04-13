using System;
using System.Collections.Generic;
using Unity.MLAgents;
using UnityEngine;

/// <summary>
/// Selfish per-agent reward distributor for the Independent DQN baseline (Step 15b).
///
/// Rules (mirrors the per-agent magnitudes of RewardDistributor but with
/// selfish credit assignment):
///   - Correct sort via branch i   → only agent i receives +1/3
///   - Incorrect sort via branch i → only agent i receives -1.5/3
///   - Missed package               → all 3 agents receive -2/3 each
///
/// Miss-cost sharing is deliberate. Pure selfish would leave misses
/// unattributed (no agent responsible), removing any incentive to act at
/// all — agents would converge on a do-nothing equilibrium. Shared miss
/// cost preserves the incentive to activate gates while keeping sort credit
/// selfish, which is the defining property of the independent baseline.
///
/// Does NOT call SimpleMultiAgentGroup.AddGroupReward. MA-POCA's group
/// reward channel is unused by DQN — the Python trainer reads per-agent
/// rewards directly via decision_steps.reward.
///
/// Subscription strategy mirrors the cooperative RewardDistributor:
/// per-zone closure captures the branch index at Start(); closures are
/// stored in a list so OnDestroy can unsubscribe them by reference.
///
/// Place in: Assets/Scripts/Agents/
/// Wire INSTEAD of RewardDistributor for DQN runs. Not both — the two
/// components would double-count every event.
/// </summary>
public class IndependentRewardDistributor : MonoBehaviour, IEventCounter
{
    [Header("Wired References")]
    [Tooltip("All 3 SortingAgents in branch-index order (agent[0] handles branch 0, etc.).")]
    [SerializeField] private SortingAgent[] _agents = new SortingAgent[3];

    [Tooltip("All 4 destination zones (3 branch zones + 1 fallthrough). Order must match _zoneBranchIndices.")]
    [SerializeField] private DestinationZone[] _destinationZones = new DestinationZone[4];

    [Tooltip("Branch index for each zone in _destinationZones. Use -1 for fallthrough. " +
             "Matches the wiring convention of the cooperative RewardDistributor.")]
    [SerializeField] private int[] _zoneBranchIndices = new int[4];

    // Sacred reward magnitudes. Per-agent values identical to the
    // cooperative distributor; only the distribution rule differs.
    private const float CorrectSortRewardPerAgent = 1f / 3f;   // +0.333
    private const float IncorrectSortRewardPerAgent = -1.5f / 3f; // -0.5
    private const float MissedPackageRewardPerAgent = -2f / 3f;   // -0.667

    // Per-episode event counters (exposed for telemetry parity with RewardDistributor).
    public int CorrectSortEvents { get; private set; }
    public int IncorrectSortEvents { get; private set; }
    public int MissedPackageEvents { get; private set; }

    /// <summary>Number of correct sorts attributed to the given branch this episode.</summary>
    public int GetCorrectSortsForBranch(int branchIndex)
    {
        if (branchIndex < 0 || branchIndex >= _correctSortsByBranch.Length) return 0;
        return _correctSortsByBranch[branchIndex];
    }

    private readonly int[] _correctSortsByBranch = new int[3];

    // Stored closure references so OnDestroy can unsubscribe them by reference.
    // Per-index entries: one handler per destination zone. Fallthrough zones store null.
    private readonly List<Action<Package>> _correctSortHandlers = new List<Action<Package>>();
    private readonly List<Action<Package>> _incorrectSortHandlers = new List<Action<Package>>();

    private void Awake()
    {
        Debug.Assert(_agents != null && _agents.Length == 3,
            "[IndependentRewardDistributor] Requires exactly 3 SortingAgents.", this);
        Debug.Assert(_destinationZones != null && _destinationZones.Length > 0,
            "[IndependentRewardDistributor] _destinationZones not assigned.", this);
        Debug.Assert(_zoneBranchIndices != null && _zoneBranchIndices.Length == _destinationZones.Length,
            "[IndependentRewardDistributor] _zoneBranchIndices must match _destinationZones length.", this);

        // Sacred-number assertions. Catch accidental drift from the spec.
        Debug.Assert(Mathf.Approximately(CorrectSortRewardPerAgent, 0.3333f)
            || Mathf.Approximately(CorrectSortRewardPerAgent, 1f / 3f),
            "[IndependentRewardDistributor] CorrectSort reward drifted from spec (+1/3).");
        Debug.Assert(Mathf.Approximately(IncorrectSortRewardPerAgent, -0.5f),
            "[IndependentRewardDistributor] IncorrectSort reward drifted from spec (-0.5).");
        Debug.Assert(Mathf.Approximately(MissedPackageRewardPerAgent, -0.6667f)
            || Mathf.Approximately(MissedPackageRewardPerAgent, -2f / 3f),
            "[IndependentRewardDistributor] Missed reward drifted from spec (-2/3).");
    }

    private void Start()
    {
        // Subscribe to each destination zone with a closure that captures the branch index.
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
                // Fallthrough: all agents share the miss cost. No branch index needed.
                zone.OnMissedPackage += HandleMissedPackage;
                _correctSortHandlers.Add(null);
                _incorrectSortHandlers.Add(null);
            }
            else
            {
                int capturedBranchIndex = _zoneBranchIndices[i];
                Action<Package> correctHandler = (pkg) => HandleCorrectSortForBranch(pkg, capturedBranchIndex);
                Action<Package> incorrectHandler = (pkg) => HandleIncorrectSortForBranch(pkg, capturedBranchIndex);
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
            Debug.LogError("[IndependentRewardDistributor] EnvironmentManager.Instance is null in Start.", this);
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

        // Selfish credit: ONLY the responsible agent gets the reward.
        if (branchIndex >= 0 && branchIndex < _agents.Length && _agents[branchIndex] != null)
        {
            _agents[branchIndex].AddReward(CorrectSortRewardPerAgent);
        }
    }

    private void HandleIncorrectSortForBranch(Package pkg, int branchIndex)
    {
        IncorrectSortEvents++;

        // Selfish blame: ONLY the responsible agent gets penalised.
        if (branchIndex >= 0 && branchIndex < _agents.Length && _agents[branchIndex] != null)
        {
            _agents[branchIndex].AddReward(IncorrectSortRewardPerAgent);
        }
    }

    private void HandleMissedPackage(Package pkg)
    {
        MissedPackageEvents++;

        // Shared blame: all agents take the miss cost equally. Preserves
        // the incentive to activate gates (otherwise do-nothing is optimal).
        for (int i = 0; i < _agents.Length; i++)
        {
            if (_agents[i] != null)
                _agents[i].AddReward(MissedPackageRewardPerAgent);
        }
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