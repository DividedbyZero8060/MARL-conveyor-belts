using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

/// <summary>
/// Single-agent centralised PPO baseline (Step 15c).
///
/// Sees all three branches at once and controls all three gates
/// simultaneously via a multi-discrete action space [2, 2, 2].
///
/// Observation: 114 floats (38 × 3 raw concatenation). The workflow
/// explicitly specifies this redundant layout rather than a deduplicated
/// 106-float version — the MLP trivially learns that shared fields
/// (belt speed, congestion) are duplicated across the three blocks.
/// Keeping it raw means the centralised baseline reuses ObservationBuilder
/// with zero special cases and the thesis reviewer can verify by
/// inspection that no information is hidden from the centralised view.
///
/// Action space: Discrete branches = [2, 2, 2].
///   action[0] ∈ {0, 1} : 1 = activate Gate 0 (if actionable)
///   action[1] ∈ {0, 1} : 1 = activate Gate 1 (if actionable)
///   action[2] ∈ {0, 1} : 1 = activate Gate 2 (if actionable)
///
/// Each branch is masked independently: if Gate i is not Retracted,
/// action[i] = 1 is masked out so the policy cannot propose an illegal
/// activation. Identical semantics to SortingAgent's per-agent mask.
///
/// Reward: team total delivered by CentralisedRewardDistributor via
/// AddReward(). Team magnitudes are +1.0 / -1.5 / -2.0. The agent
/// bypasses SimpleMultiAgentGroup entirely — there is no group.
///
/// Lifecycle:
///   - Initialize(): wires scene references, caches state
///   - OnEpisodeBegin(): no-op (EnvironmentManager drives resets globally)
///   - CollectObservations(): builds 114-float observation via ObservationBuilder
///   - WriteDiscreteActionMask(): masks each gate's activate action when
///     the gate is not Retracted
///   - OnActionReceived(): reads 3 discrete actions, calls Activate() on
///     each gate whose action bit is 1 (Activate itself is idempotent when
///     the gate isn't actionable)
///   - Heuristic(): not implemented — centralised baseline doesn't need
///     manual testing, use SortingAgent heuristic for that
///
/// Place in: Assets/Scripts/Agents/
/// Wire alongside EnvironmentManager in the scene hierarchy. Only active
/// for centralised PPO training runs. Disable the three SortingAgent
/// components (and enable this one) before launching the centralised run;
/// reverse for cooperative runs.
/// </summary>
public class CentralisedAgent : Agent
{
    [Header("Scene References")]
    [Tooltip("The three DiverterGates in branch order (0, 1, 2).")]
    [SerializeField] private DiverterGate[] _gates = new DiverterGate[3];

    [Tooltip("The three PackageDetectors in branch order (0, 1, 2). " +
             "These can remain under disabled SortingAgent GameObjects — " +
             "the detector components stay active regardless of their parent agent.")]
    [SerializeField] private PackageDetector[] _packageDetectors = new PackageDetector[3];

    [Tooltip("All three BranchTrackers in branch order (0, 1, 2). Shared reference " +
             "passed into every per-branch observation block.")]
    [SerializeField] private BranchTracker[] _branchTrackers = new BranchTracker[3];

    [Tooltip("Per-branch agent GameObjects providing the _overlappingPackages list " +
             "for each branch's PackageDetector. These are the existing SortingAgent " +
             "GameObjects — their MonoBehaviour can be disabled but their Transform " +
             "and OverlappingPackages list remain accessible via the component reference.")]
    [SerializeField] private SortingAgent[] _branchAgents = new SortingAgent[3];

    // Buffer large enough for full-obs layout (114 floats = 3 × 38).
    // Partial observability is not supported for the centralised baseline —
    // the whole point is an omniscient upper bound.
    private const int CentralisedObsSize = 3 * ObsIndices.FullObsSize;
    private readonly float[] _obsBuffer = new float[CentralisedObsSize];

    // Per-branch observation contexts, reused every decision.
    private ObservationBuilder.ObservationContext[] _contexts
        = new ObservationBuilder.ObservationContext[3];


    private void Start()
    {
        // Subscribe to EnvironmentManager.OnEpisodeEnded in Start() rather
        // than Initialize() so EnvironmentManager.Instance is guaranteed to
        // be non-null. Unity's lifecycle runs all Awake() before any Start(),
        // and Agent.Initialize() is called from Awake() — which may fire
        // before EnvironmentManager.Awake() has set the singleton.
        //
        // When OnEpisodeEnded fires, we call EndEpisode() on ourselves so
        // ML-Agents closes the rollout and starts a fresh trajectory.
        // EnvironmentManager.ResetEpisode() runs immediately after this
        // event, so by the next decision tick the environment is already
        // reset.
        if (EnvironmentManager.Instance != null)
        {
            EnvironmentManager.Instance.OnEpisodeEnded += HandleEnvironmentEpisodeEnded;
        }
        else
        {
            Debug.LogError(
                "[CentralisedAgent] EnvironmentManager.Instance is NULL in Start — " +
                "centralised episode boundaries will not fire and PPO will never " +
                "complete an episode. Check that EnvironmentManager GameObject is active.",
                this);
        }
    }
    public override void Initialize()
    {
        Debug.Assert(_gates != null && _gates.Length == 3,
            "[CentralisedAgent] Requires exactly 3 gates.", this);
        Debug.Assert(_packageDetectors != null && _packageDetectors.Length == 3,
            "[CentralisedAgent] Requires exactly 3 PackageDetectors.", this);
        Debug.Assert(_branchTrackers != null && _branchTrackers.Length == 3,
            "[CentralisedAgent] Requires exactly 3 BranchTrackers.", this);
        Debug.Assert(_branchAgents != null && _branchAgents.Length == 3,
            "[CentralisedAgent] Requires exactly 3 branch-owning SortingAgents.", this);

        for (int i = 0; i < 3; i++)
        {
            Debug.Assert(_gates[i] != null, $"[CentralisedAgent] _gates[{i}] not assigned.", this);
            Debug.Assert(_packageDetectors[i] != null,
                $"[CentralisedAgent] _packageDetectors[{i}] not assigned.", this);
            Debug.Assert(_branchTrackers[i] != null,
                $"[CentralisedAgent] _branchTrackers[{i}] not assigned.", this);
            Debug.Assert(_branchAgents[i] != null,
                $"[CentralisedAgent] _branchAgents[{i}] not assigned.", this);
        }

        
    }

    private void OnDestroy()
    {
        if (EnvironmentManager.Instance != null)
        {
            EnvironmentManager.Instance.OnEpisodeEnded -= HandleEnvironmentEpisodeEnded;
        }
    }

    private void HandleEnvironmentEpisodeEnded()
    {
        // Tell ML-Agents this agent's episode is done. The framework will
        // snapshot the rollout, trigger OnEpisodeBegin next decision tick,
        // and start accumulating a fresh trajectory. EnvironmentManager
        // .ResetEpisode runs immediately after this event fires, so the
        // environment is already set up for the next episode by the time
        // OnEpisodeBegin runs.
        EndEpisode();
    }

    public override void OnEpisodeBegin()
    {
        // EnvironmentManager.ResetEpisode owns the reset lifecycle. Nothing
        // to do here — gates are reset, packages are pooled, destinations
        // are shuffled, all by the environment manager before this callback
        // would fire from ML-Agents.
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Zero the full 114-float buffer. Each branch block is written in
        // place by WriteBranchObservation.
        for (int i = 0; i < CentralisedObsSize; i++) _obsBuffer[i] = 0f;

        // Compute all three branches' gate states and nearest-package distances
        // first, so peer features can be populated correctly per block.
        // This mirrors what SortingAgent would see when its _peerAgents list
        // references the other two SortingAgents — except we compute from the
        // gate and the branch agent's own cached nearest-package distance.
        float[] allGateStates = new float[3];
        float[] allNearestDistances = new float[3];
        for (int i = 0; i < 3; i++)
        {
            allGateStates[i] = ObservationBuilder.GetNormalisedGateState(_gates[i]);
            // Use the branch agent's cached NearestPackageDistance. This is
            // updated inside WriteBranchObservation below, so the value we
            // read here is from the PREVIOUS decision tick — same lag that
            // SortingAgent's peer reads have. Consistent semantics.
            allNearestDistances[i] = _branchAgents[i] != null
                ? _branchAgents[i].NearestPackageDistance
                : 1f;
        }

        // Write each branch block into the buffer at its 38-float offset.
        for (int branchIdx = 0; branchIdx < 3; branchIdx++)
        {
            // Peer arrays: for branch N, peers are the OTHER two branches.
            // Matches SortingAgent's _peerAgents convention where peers are
            // the two agents that are NOT self, in Inspector-assigned order.
            int peer0Idx = (branchIdx + 1) % 3;
            int peer1Idx = (branchIdx + 2) % 3;
            float[] peerGates = new float[]
            {
                allGateStates[peer0Idx],
                allGateStates[peer1Idx],
            };
            float[] peerDistances = new float[]
            {
                allNearestDistances[peer0Idx],
                allNearestDistances[peer1Idx],
            };

            _contexts[branchIdx] = new ObservationBuilder.ObservationContext
            {
                BranchIndex = branchIdx,
                Gate = _gates[branchIdx],
                PackageDetector = _packageDetectors[branchIdx],
                OverlappingPackages = _branchAgents[branchIdx] != null
                    ? _branchAgents[branchIdx].OverlappingPackages
                    : null,
                AllBranchTrackers = _branchTrackers,
                PeerGateStates = peerGates,
                PeerNearestDistances = peerDistances,
                NearestPackageDistanceOut = 0f,
            };

            int offset = branchIdx * ObsIndices.FullObsSize;
            ObservationBuilder.WriteBranchObservation(
                ref _contexts[branchIdx],
                _obsBuffer,
                offset,
                partialObservability: false);

            // Propagate the branch's cached nearest-package distance back to
            // the underlying SortingAgent so peer reads on the NEXT frame
            // see the fresh value. This is the same flow SortingAgent uses
            // after its own CollectObservations runs.
            if (_branchAgents[branchIdx] != null)
            {
                _branchAgents[branchIdx].SetCachedNearestPackageDistance(
                    _contexts[branchIdx].NearestPackageDistanceOut);
            }
        }

        // Commit the full 114-float observation to the sensor.
        for (int i = 0; i < CentralisedObsSize; i++)
        {
            sensor.AddObservation(_obsBuffer[i]);
        }
    }

    public override void WriteDiscreteActionMask(IDiscreteActionMask actionMask)
    {
        // Mask each branch independently. When gate i is not Retracted,
        // its activate-action (index 1) is disabled so PPO cannot propose it.
        for (int branchIdx = 0; branchIdx < 3; branchIdx++)
        {
            if (_gates[branchIdx] != null
                && _gates[branchIdx].CurrentState != GateState.Retracted)
            {
                actionMask.SetActionEnabled(branchIdx, 1, false);
            }
        }
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        // Read all three discrete action branches and activate the
        // corresponding gates. Activate() is idempotent when the gate
        // is not actionable — no need to re-check here.
        ActionSegment<int> discrete = actions.DiscreteActions;
        if (discrete.Length < 3) return;

        for (int branchIdx = 0; branchIdx < 3; branchIdx++)
        {
            if (discrete[branchIdx] == 1 && _gates[branchIdx] != null)
            {
                _gates[branchIdx].Activate();
            }
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        // Not intended for manual testing. Zero out the actions so heuristic
        // mode (if accidentally enabled) is a no-op rather than producing
        // random or confusing behaviour.
        ActionSegment<int> discrete = actionsOut.DiscreteActions;
        for (int i = 0; i < discrete.Length && i < 3; i++)
        {
            discrete[i] = 0;
        }
    }
}