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

    [Header("Reward Mode")]
    [Tooltip("Extra penalty applied ONLY to the acting agent on incorrect sorts. " +
         "Breaks the fire-on-everything attractor by making wrong-firing costlier " +
         "than not-firing, while keeping correct-firing rewarding.")]
    [SerializeField] private bool _useInterceptionPenalty = false;

    [Tooltip("Magnitude of the extra penalty to the acting agent on incorrect sort. " +
             "Added on top of the normal equal-split reward. Keep small (0.05-0.15).")]
    [SerializeField] private float _interceptionPenaltyMagnitude = 0.1f;

    [Header("Responsibility-Attributed Miss")]
    [Tooltip("When a package is missed, the agent whose branch is assigned to that package's " +
             "destination pays the majority of the team miss penalty (-1.334), while the other " +
             "two agents pay a small share (-0.333 each). Team total remains -2.0 (sacred). " +
             "Mechanism: flips the lazy-agent inequality (fire-wrong -0.5 vs don't-fire-responsible " +
             "-1.334) so abstention-when-responsible is strongly penalised. Preserves the baseline " +
             "fire-when-wrong penalty (-0.5) so gradient dynamics teach selectivity before restraint. " +
             "Compatible with _useCounterfactualCredit (orthogonal — acts on miss events only).")]
    [SerializeField] private bool _useResponsibilityAttributedMiss = false;

    [Tooltip("Responsible agent's share of the team miss penalty. Default -1.334 = -2.0 × 2/3. " +
             "Combined with -0.333 × 2 non-responsible shares, sums to exactly -2.0 (sacred team total).")]
    [SerializeField] private float _responsibleMissShare = -1.334f;

    [Tooltip("Each non-responsible agent's share of the team miss penalty. Default -0.333 = -2.0 × 1/6. " +
             "Two non-responsible agents × -0.333 + one responsible × -1.334 = -2.0 exactly.")]
    [SerializeField] private float _nonResponsibleMissShare = -0.333f;

    [Header("Potential-Based Reward Shaping (Ng et al. 1999)")]
    [Tooltip("Enable policy-invariant dense reward shaping based on a routing potential. " +
             "Φ(s) = Σ_branch Σ_package_on_branch[+1 if matching destination else -1]. " +
             "F(s,s') = α·(γΦ(s') - Φ(s)) is added to the team reward each FixedUpdate. " +
             "Provides immediate commit-time feedback without changing the optimal policy. " +
             "Recommended: leave _useIntentShaping OFF when this is ON — they overlap.")]
    [SerializeField] private bool _usePotentialShaping = false;

    [Tooltip("Discount factor used inside the shaping term. MUST match your training " +
             "config's gamma (typically 0.99 per SortingAgent.yaml). Desync breaks the " +
             "policy-invariance guarantee.")]
    [SerializeField] private float _potentialGamma = 0.99f;

    [Tooltip("Scaling coefficient α on the shaping term. 0.1 gives commit-time feedback " +
             "of ±0.099 per team event, ~10% of the sparse correct-sort reward scale. " +
             "Strong enough to break gradient indifference, weak enough to let sparse " +
             "signal dominate long-horizon behaviour. Tune in [0.05, 0.20].")]
    [SerializeField] private float _potentialShapingCoefficient = 0.1f;

    [Tooltip("All three BranchTrackers in branch-index order (Branch 0, 1, 2). " +
             "Required when _usePotentialShaping is true. The same BranchTrackers " +
             "that feed the congestion observation.")]
    [SerializeField] private BranchTracker[] _branchTrackersForPotential = new BranchTracker[3];

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
        string pbrs = _usePotentialShaping
            ? $"PBRS (α={_potentialShapingCoefficient}, γ={_potentialGamma})"
            : "NO PBRS";
        string respMiss = _useResponsibilityAttributedMiss
            ? $"RESPONSIBILITY MISS (resp={_responsibleMissShare}, others={_nonResponsibleMissShare})"
            : "EQUAL MISS";
        Debug.Log($"[RewardDistributor] Mode: {mode}, {intent}, {pbrs}, {respMiss}");

        if (_useResponsibilityAttributedMiss)
        {
            // Validate the team-total invariant: 1 × resp + 2 × non-resp should equal -2.0.
            float teamTotal = _responsibleMissShare + 2f * _nonResponsibleMissShare;
            Debug.Assert(Mathf.Abs(teamTotal - TEAM_MISSED) < 0.01f,
                $"[RewardDistributor] Responsibility-attributed miss shares violate team total: " +
                $"resp ({_responsibleMissShare}) + 2 × non-resp ({_nonResponsibleMissShare}) " +
                $"= {teamTotal}, expected {TEAM_MISSED}. Adjust _responsibleMissShare or _nonResponsibleMissShare.", this);
        }

        if (_usePotentialShaping)
        {
            Debug.Assert(_branchTrackersForPotential != null && _branchTrackersForPotential.Length == 3,
                "[RewardDistributor] PBRS enabled but _branchTrackersForPotential must have exactly 3 entries.", this);
            for (int i = 0; i < _branchTrackersForPotential.Length; i++)
            {
                Debug.Assert(_branchTrackersForPotential[i] != null,
                    $"[RewardDistributor] PBRS enabled but _branchTrackersForPotential[{i}] is null.", this);
            }
            Debug.Assert(_potentialGamma > 0f && _potentialGamma <= 1f,
                $"[RewardDistributor] _potentialGamma out of (0,1]: {_potentialGamma}", this);
            Debug.Assert(_potentialShapingCoefficient >= 0f,
                $"[RewardDistributor] _potentialShapingCoefficient must be >= 0: {_potentialShapingCoefficient}", this);

            if (_useIntentShaping)
            {
                Debug.LogWarning(
                    "[RewardDistributor] Both _useIntentShaping and _usePotentialShaping are enabled. " +
                    "These overlap conceptually (both give commit-time signal). Recommend disabling " +
                    "_useIntentShaping for the PBRS experiment so the effect can be attributed cleanly.");
            }
        }
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

        // Extra per-agent penalty for the acting agent only.
        // Does NOT go through AddGroupReward — only affects the individual agent's
        // AddReward channel. Keeps MA-POCA's group critic unchanged.
        if (_useInterceptionPenalty && branchIndex >= 0 && branchIndex < _agents.Length)
        {
            if (_agents[branchIndex] != null)
                _agents[branchIndex].AddReward(-_interceptionPenaltyMagnitude);
        }
    }

    private void HandleMissedPackage(Package pkg)
    {
        MissedPackageEvents++;

        if (_useResponsibilityAttributedMiss && pkg != null)
        {
            DistributeMissWithResponsibility(pkg.DestinationLabel);
        }
        else
        {
            // Baseline: equal split across all agents.
            DistributeEqual(EQUAL_MISSED, TEAM_MISSED);
        }
    }

    /// <summary>
    /// Distribute the team miss penalty asymmetrically: the agent whose branch
    /// is currently assigned to the missed package's destination label pays
    /// _responsibleMissShare; the other two pay _nonResponsibleMissShare each.
    ///
    /// Team-channel (AddGroupReward) sees the full TEAM_MISSED (-2.0), unchanged
    /// from baseline — so MA-POCA's cooperative critic is untouched.
    ///
    /// Responsibility lookup reads EnvironmentManager.GetDestinationForBranch
    /// dynamically each call, so this works correctly under both curriculum
    /// (fixed mapping) and per-episode shuffling.
    /// </summary>
    private void DistributeMissWithResponsibility(DestinationLabel missedLabel)
    {
        // Team channel (unchanged from baseline)
        if (EnvironmentManager.Instance != null
            && EnvironmentManager.Instance._agentGroup != null
            && EnvironmentManager.Instance._agentGroup.Group != null)
        {
            EnvironmentManager.Instance._agentGroup.Group.AddGroupReward(TEAM_MISSED);
        }

        // Identify the responsible branch index by reading the current
        // destination mapping. Returns -1 if no branch is currently assigned
        // to this label (shouldn't happen in normal operation, but we degrade
        // gracefully to equal split in that case).
        int responsibleIdx = FindResponsibleBranchIndex(missedLabel);

        if (responsibleIdx < 0 || responsibleIdx >= _agents.Length)
        {
            // Fallback: no agent is responsible for this destination (misconfig).
            // Fall back to equal split so we don't silently lose the penalty.
            for (int i = 0; i < _agents.Length; i++)
            {
                if (_agents[i] != null)
                    _agents[i].AddReward(EQUAL_MISSED);
            }
            Debug.LogWarning(
                $"[RewardDistributor] Missed package with destination {missedLabel} " +
                "has no responsible branch in current mapping. Falling back to equal split.", this);
            return;
        }

        // Selfish distribution
        for (int i = 0; i < _agents.Length; i++)
        {
            if (_agents[i] == null) continue;
            float share = (i == responsibleIdx) ? _responsibleMissShare : _nonResponsibleMissShare;
            _agents[i].AddReward(share);
        }
    }

    /// <summary>
    /// Returns the branch index currently assigned to the given destination label,
    /// or -1 if no branch matches. Queries EnvironmentManager dynamically so
    /// shuffling and curriculum modes both work.
    /// </summary>
    private int FindResponsibleBranchIndex(DestinationLabel label)
    {
        if (EnvironmentManager.Instance == null) return -1;
        for (int branchIdx = 0; branchIdx < 3; branchIdx++)
        {
            if (EnvironmentManager.Instance.GetDestinationForBranch(branchIdx) == label)
                return branchIdx;
        }
        return -1;
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

        // PBRS: re-anchor the potential to 0 at episode start.
        // BranchTrackers zero their HashSets on this same event, so Φ==0 naturally
        // after this frame, but we zero _previousPotential explicitly to avoid any
        // one-frame stale-value issues during the reset transition.
        _previousPotential = 0f;
        _episodeShapingSum = 0f;
        _episodePhiSum = 0f;
        _episodePbrsSamples = 0;
    }

    // =================================================================
    // Potential-Based Reward Shaping (Ng, Harada & Russell 1999)
    // =================================================================
    //
    // Φ(s) = Σ_branch Σ_package_on_branch [ +1 if matching dest, -1 otherwise ]
    // F(s,s') = α · ( γ · Φ(s') - Φ(s) )
    //
    // Distributed as team reward: both SimpleMultiAgentGroup.AddGroupReward
    // (for MA-POCA's cooperative critic) and per-agent AddReward (for per-agent
    // readouts in MADDPG/DQN). Mirrors the existing sparse-reward distribution
    // pattern so PBRS stacks cleanly on top without breaking any toggle.

    private float _previousPotential = 0f;

    // Per-episode PBRS telemetry — exposed via the public properties below
    // so DebugOverlay (or any other StatsRecorder bridge) can log to TensorBoard.
    private float _episodeShapingSum = 0f;   // sum of F values delivered this episode (team units)
    private float _episodePhiSum = 0f;       // sum of Φ(s) samples this episode
    private int _episodePbrsSamples = 0;   // number of FixedUpdate ticks this episode

    /// <summary>Mean routing potential Φ over the current episode so far. 0 when PBRS disabled.</summary>
    public float MeanPotentialThisEpisode =>
        _episodePbrsSamples > 0 ? _episodePhiSum / _episodePbrsSamples : 0f;

    /// <summary>
    /// Cumulative team shaping reward (ΣF) delivered this episode so far.
    /// Should be close to zero over a complete episode if the telescope closes
    /// (i.e., all packages cleared by episode end). Large residuals indicate
    /// the terminal-correction approximation is biting.
    /// </summary>
    public float ShapingRewardSumThisEpisode => _episodeShapingSum;

    private void FixedUpdate()
    {
        if (!_usePotentialShaping) return;

        float currentPotential = ComputeRoutingPotential();
        float shapingTeamReward =
            _potentialShapingCoefficient *
            (_potentialGamma * currentPotential - _previousPotential);

        DistributeShaping(shapingTeamReward);

        _previousPotential = currentPotential;

        _episodeShapingSum += shapingTeamReward;
        _episodePhiSum += currentPotential;
        _episodePbrsSamples++;
    }

    /// <summary>
    /// Computes Φ(s) by iterating every BranchTracker and comparing each tracked
    /// package's destination label to the branch's currently-accepted label.
    /// O(total packages on branches). With pool size 40 and 3 branches, worst case
    /// is 40 comparisons per tick — negligible at 50Hz.
    /// </summary>
    private float ComputeRoutingPotential()
    {
        if (EnvironmentManager.Instance == null) return 0f;

        float phi = 0f;
        for (int b = 0; b < _branchTrackersForPotential.Length; b++)
        {
            BranchTracker tracker = _branchTrackersForPotential[b];
            if (tracker == null) continue;

            DestinationLabel branchDest = EnvironmentManager.Instance.GetDestinationForBranch(b);

            foreach (Package pkg in tracker.TrackedPackages)
            {
                if (pkg == null || !pkg.gameObject.activeInHierarchy) continue;
                phi += (pkg.DestinationLabel == branchDest) ? +1f : -1f;
            }
        }
        return phi;
    }

    /// <summary>
    /// Distribute the shaping reward identically to the sparse equal-split path:
    /// AddGroupReward(team) + per-agent AddReward(team / N). Both MA-POCA and
    /// MADDPG pick up the right channel.
    /// </summary>
    private void DistributeShaping(float teamShapingReward)
    {
        if (Mathf.Abs(teamShapingReward) < 1e-9f) return;

        float perAgent = teamShapingReward / _agents.Length;

        if (EnvironmentManager.Instance != null
            && EnvironmentManager.Instance._agentGroup != null
            && EnvironmentManager.Instance._agentGroup.Group != null)
        {
            EnvironmentManager.Instance._agentGroup.Group.AddGroupReward(teamShapingReward);
        }

        for (int i = 0; i < _agents.Length; i++)
        {
            if (_agents[i] != null)
                _agents[i].AddReward(perAgent);
        }
    }

}