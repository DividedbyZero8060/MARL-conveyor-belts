using System.Collections.Generic;
using UnityEngine;
/// <summary>
/// Cooperative reward distributor with optional counterfactual credit
/// and intent shaping.
///
/// TWO MODES (controlled by _useCounterfactualCredit toggle):
///
/// Mode A — Equal split (original sparse reward, _useCounterfactualCredit = false):
///   Correct sort   → +0.333 per agent  (+1.0 / 3)
///   Incorrect sort → -0.5   per agent  (-1.5 / 3)
///   Missed package → -0.667 per agent  (-2.0 / 3)
///
/// Mode B — Counterfactual credit (_useCounterfactualCredit = true):
///   Correct sort   → acting agent +0.667, others +0.167 each  (total +1.0)
///   Incorrect sort → acting agent -1.0,   others -0.25  each  (total -1.5)
///   Missed package → all -0.667 each                          (total -2.0)
///
///   Team totals are IDENTICAL in both modes. Only the per-agent split changes.
///   The acting agent is identified by the branch index of the DestinationZone
///   that received the package.
///
/// INTENT SHAPING (controlled by _useIntentShaping toggle):
///   When an agent's gate transitions from Committed to Deploying (meaning
///   the agent chose to intercept a package), a small immediate reward is
///   given based on whether the intercepted package's destination matches
///   the branch's assigned destination:
///     Match    → acting agent only gets +0.05
///     Mismatch → acting agent only gets -0.05
///   This converts the 6-events-per-episode sparse signal into a per-decision
///   dense signal without changing the optimal policy.
/// </summary>
public class RewardDistributor : MonoBehaviour, IEventCounter
{
    [Header("Reward Mode")]
    [Tooltip("If true, the agent whose gate fired gets a larger reward share " +
             "(counterfactual credit). If false, all agents share equally (original sparse).")]
    [SerializeField] private bool _useCounterfactualCredit = false;

    [Tooltip("If true, agents get a small immediate +0.05/-0.05 reward when " +
             "they commit to intercepting a package, based on destination match.")]
    [SerializeField] private bool _useIntentShaping = false;

    [Tooltip("Magnitude of the intent shaping reward. Small enough not to " +
             "dominate the terminal sort/miss rewards.")]
    [SerializeField] private float _intentShapingMagnitude = 0.05f;

    [Header("Wired References")]
    [Tooltip("All SortingAgents that receive shared reward. Typically size 3.")]
    [SerializeField] private SortingAgent[] _agents = new SortingAgent[3];

    [Tooltip("All DestinationZones in the scene, including the fallthrough zone at the trunk end.")]
    [SerializeField] private DestinationZone[] _destinationZones = new DestinationZone[4];

    [Tooltip("Branch index for each DestinationZone in _destinationZones, in the same order. " +
             "For fallthrough zones, use -1. Used to attribute correct sorts per branch.")]
    [SerializeField] private int[] _zoneBranchIndices = new int[4];

    [Header("Intent Shaping References (only needed if _useIntentShaping = true)")]
    [Tooltip("DiverterGates in branch-index order. Required for intent shaping to " +
             "detect commit-zone triggers. Leave empty if _useIntentShaping is false.")]
    [SerializeField] private DiverterGate[] _gates = new DiverterGate[0];

    // ================================================================
    // Reward constants
    // ================================================================

    // Mode A: equal split (original sacred numbers)
    private const float EQUAL_CORRECT = 1.0f / 3f;   // +0.333
    private const float EQUAL_INCORRECT = -1.5f / 3f;   // -0.500
    private const float EQUAL_MISSED = -2.0f / 3f;   // -0.667

    // Mode B: counterfactual credit (team totals unchanged)
    private const float CF_CORRECT_ACTOR = 0.667f;   // acting agent
    private const float CF_CORRECT_OTHER = 0.167f;   // other agents  (0.667 + 2*0.167 ≈ 1.0)
    private const float CF_INCORRECT_ACTOR = -1.000f;    // acting agent
    private const float CF_INCORRECT_OTHER = -0.250f;    // other agents  (-1.0 + 2*-0.25 = -1.5)
    // Missed: same as equal split (no acting agent)

    // Team totals for AddGroupReward (same in both modes)
    private const float TEAM_CORRECT = 1.0f;
    private const float TEAM_INCORRECT = -1.5f;
    private const float TEAM_MISSED = -2.0f;

    // Per-branch correct sort counter, indexed by DiverterGate.BranchIndex.
    private readonly int[] _correctSortsByBranch = new int[3];

    // Captured-closure handlers for per-zone subscriptions.
    // Stored so OnDestroy can unsubscribe them by reference.
    private readonly List<System.Action<Package>> _correctSortHandlers = new List<System.Action<Package>>();
    private readonly List<System.Action<Package>> _incorrectSortHandlers = new List<System.Action<Package>>();

    // Intent shaping handlers (one per gate commit zone)
    private readonly List<System.Action<Package>> _intentHandlers = new List<System.Action<Package>>();

    // Cumulative event counters, reset per episode.
    public int CorrectSortEvents { get; private set; }
    public int IncorrectSortEvents { get; private set; }
    public int MissedPackageEvents { get; private set; }

    // Intent shaping counter for debugging/TensorBoard
    public int IntentCorrectEvents { get; private set; }
    public int IntentIncorrectEvents { get; private set; }

    private void Awake()
    {
        Debug.Assert(_agents != null && _agents.Length > 0,
            "[RewardDistributor] _agents is empty.", this);
        for (int i = 0; i < _agents.Length; i++)
        {
            Debug.Assert(_agents[i] != null,
                $"[RewardDistributor] _agents[{i}] is null.", this);
        }

        Debug.Assert(_destinationZones != null && _destinationZones.Length > 0,
            "[RewardDistributor] _destinationZones is empty.", this);
        for (int i = 0; i < _destinationZones.Length; i++)
        {
            Debug.Assert(_destinationZones[i] != null,
                $"[RewardDistributor] _destinationZones[{i}] is null.", this);
        }

        if (_useIntentShaping)
        {
            Debug.Assert(_gates != null && _gates.Length == _agents.Length,
                "[RewardDistributor] Intent shaping enabled but _gates array size " +
                $"({(_gates != null ? _gates.Length : 0)}) != _agents.Length ({_agents.Length}).", this);
        }

        // Log mode at startup for clarity
        string mode = _useCounterfactualCredit ? "COUNTERFACTUAL CREDIT" : "EQUAL SPLIT";
        string intent = _useIntentShaping ? $"INTENT SHAPING (±{_intentShapingMagnitude})" : "NO INTENT SHAPING";
        Debug.Log($"[RewardDistributor] Mode: {mode}, {intent}");
    }

    private void Start()
    {
        Debug.Assert(_zoneBranchIndices != null && _zoneBranchIndices.Length == _destinationZones.Length,
            "[RewardDistributor] _zoneBranchIndices must have same length as _destinationZones.", this);

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

                // Correct sort handler (with branch index for counterfactual credit)
                System.Action<Package> correctHandler =
                    (pkg) => HandleCorrectSortForBranch(pkg, capturedBranchIndex);
                zone.OnCorrectSort += correctHandler;
                _correctSortHandlers.Add(correctHandler);

                // Incorrect sort handler (NOW also captures branch index)
                System.Action<Package> incorrectHandler =
                    (pkg) => HandleIncorrectSortForBranch(pkg, capturedBranchIndex);
                zone.OnIncorrectSort += incorrectHandler;
                _incorrectSortHandlers.Add(incorrectHandler);
            }
        }

        // Intent shaping: subscribe to each gate's commit zone
        if (_useIntentShaping && _gates != null)
        {
            for (int g = 0; g < _gates.Length; g++)
            {
                if (_gates[g] == null) continue;
                int capturedGateIndex = g;
                System.Action<Package> intentHandler =
                    (pkg) => HandleCommitZonePackage(pkg, capturedGateIndex);

                // Subscribe to the commit zone's OnPackageEntered event.
                // Requires DiverterGate to expose CommitZone publicly.
                CommitZoneTrigger commitZone = _gates[g].CommitZone;
                if (commitZone != null)
                {
                    commitZone.OnPackageEntered += intentHandler;
                    _intentHandlers.Add(intentHandler);
                }
                else
                {
                    Debug.LogWarning(
                        $"[RewardDistributor] Gate {g} has no CommitZone; intent shaping skipped for this gate.");
                    _intentHandlers.Add(null);
                }
            }
        }

        if (EnvironmentManager.Instance != null)
        {
            EnvironmentManager.Instance.OnEpisodeReset += HandleEpisodeReset;
        }
        else
        {
            Debug.LogError("[RewardDistributor] EnvironmentManager.Instance is null in Start!", this);
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

        // Unsubscribe intent handlers
        if (_useIntentShaping && _gates != null)
        {
            for (int g = 0; g < _gates.Length && g < _intentHandlers.Count; g++)
            {
                if (_gates[g] == null || _intentHandlers[g] == null) continue;
                CommitZoneTrigger commitZone = _gates[g].CommitZone;
                if (commitZone != null)
                    commitZone.OnPackageEntered -= _intentHandlers[g];
            }
        }

        if (EnvironmentManager.Instance != null)
        {
            EnvironmentManager.Instance.OnEpisodeReset -= HandleEpisodeReset;
        }
    }

    // ================================================================
    // Event handlers
    // ================================================================

    private void HandleCorrectSortForBranch(Package pkg, int branchIndex)
    {
        CorrectSortEvents++;
        if (branchIndex >= 0 && branchIndex < _correctSortsByBranch.Length)
            _correctSortsByBranch[branchIndex]++;

        if (_useCounterfactualCredit)
            DistributeWithCredit(TEAM_CORRECT, CF_CORRECT_ACTOR, CF_CORRECT_OTHER, branchIndex);
        else
            DistributeEqual(EQUAL_CORRECT, TEAM_CORRECT);
    }

    private void HandleIncorrectSortForBranch(Package pkg, int branchIndex)
    {
        IncorrectSortEvents++;

        if (_useCounterfactualCredit)
            DistributeWithCredit(TEAM_INCORRECT, CF_INCORRECT_ACTOR, CF_INCORRECT_OTHER, branchIndex);
        else
            DistributeEqual(EQUAL_INCORRECT, TEAM_INCORRECT);
    }

    private void HandleMissedPackage(Package pkg)
    {
        MissedPackageEvents++;
        // Missed packages have no acting agent — always equal split.
        DistributeEqual(EQUAL_MISSED, TEAM_MISSED);
    }

    /// <summary>
    /// Intent shaping: fires when a package enters a gate's commit zone.
    /// Only gives reward if the gate is in Deploying state (just transitioned
    /// from Committed, meaning the agent chose to intercept this package).
    /// </summary>
    private void HandleCommitZonePackage(Package pkg, int gateIndex)
    {
        if (!_useIntentShaping) return;
        if (gateIndex < 0 || gateIndex >= _gates.Length) return;

        DiverterGate gate = _gates[gateIndex];
        // The gate's own handler (subscribed in Awake, fires before ours)
        // transitions from Committed → Deploying. If we see Deploying,
        // this package just triggered the deploy — the agent committed to it.
        // Also check Committed in case our handler fires first (execution order).
        if (gate.CurrentState != GateState.Deploying && gate.CurrentState != GateState.Committed)
            return;

        // Check: does the package's destination match this branch's assignment?
        DestinationLabel branchDest = EnvironmentManager.Instance.GetDestinationForBranch(gateIndex);
        bool match = (pkg.DestinationLabel == branchDest);

        float intentReward = match ? _intentShapingMagnitude : -_intentShapingMagnitude;

        // Only the acting agent gets intent reward. No team distribution.
        if (gateIndex < _agents.Length && _agents[gateIndex] != null)
        {
            _agents[gateIndex].AddReward(intentReward);
        }

        if (match)
            IntentCorrectEvents++;
        else
            IntentIncorrectEvents++;
    }

    // ================================================================
    // Reward distribution helpers
    // ================================================================

    /// <summary>
    /// Equal split: same reward to every agent (original sparse behaviour).
    /// </summary>
    private void DistributeEqual(float rewardPerAgent, float teamTotal)
    {
        // MA-POCA group reward
        if (EnvironmentManager.Instance != null
            && EnvironmentManager.Instance._agentGroup != null
            && EnvironmentManager.Instance._agentGroup.Group != null)
        {
            EnvironmentManager.Instance._agentGroup.Group.AddGroupReward(teamTotal);
        }

        for (int i = 0; i < _agents.Length; i++)
        {
            if (_agents[i] != null)
                _agents[i].AddReward(rewardPerAgent);
        }
    }

    /// <summary>
    /// Counterfactual credit: the acting agent gets a larger share,
    /// others get a smaller share. Team total is unchanged.
    /// </summary>
    private void DistributeWithCredit(float teamTotal, float actorReward, float otherReward, int actingBranchIndex)
    {
        // MA-POCA group reward (team total unchanged)
        if (EnvironmentManager.Instance != null
            && EnvironmentManager.Instance._agentGroup != null
            && EnvironmentManager.Instance._agentGroup.Group != null)
        {
            EnvironmentManager.Instance._agentGroup.Group.AddGroupReward(teamTotal);
        }

        for (int i = 0; i < _agents.Length; i++)
        {
            if (_agents[i] == null) continue;
            float reward = (i == actingBranchIndex) ? actorReward : otherReward;
            _agents[i].AddReward(reward);
        }
    }

    public int GetCorrectSortsForBranch(int branchIndex)
    {
        if (branchIndex < 0 || branchIndex >= _correctSortsByBranch.Length) return 0;
        return _correctSortsByBranch[branchIndex];
    }

    private void HandleEpisodeReset()
    {
        CorrectSortEvents = 0;
        IncorrectSortEvents = 0;
        MissedPackageEvents = 0;
        IntentCorrectEvents = 0;
        IntentIncorrectEvents = 0;
        for (int i = 0; i < _correctSortsByBranch.Length; i++)
            _correctSortsByBranch[i] = 0;
    }
}