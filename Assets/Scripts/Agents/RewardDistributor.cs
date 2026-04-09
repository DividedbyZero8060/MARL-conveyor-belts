using System.Collections.Generic;
using UnityEngine;
/// <summary>
/// Cooperative sparse reward distributor.
///
/// Shares all sorting events equally across all registered agents:
///   Correct sort   → +0.333 per agent  (+1.0 / 3)
///   Incorrect sort → -0.5   per agent  (-1.5 / 3)
///   Missed package → -0.667 per agent  (-2.0 / 3)
///
/// This is the ENTIRE reward signal. No time penalty. No continuous shaping.
/// No distance-to-package bonus. No gate-activation cost. Non-event steps
/// must deliver exactly 0.0 reward — that is the whole point of "sparse".
///
/// The distributor subscribes to every DestinationZone in the scene and
/// calls AddReward() on every registered SortingAgent for each event.
/// Python reads rewards via decision_steps.reward / terminal_steps.reward;
/// there is no GetTeamReward() bridge method and there must not be one.
/// </summary>
public class RewardDistributor : MonoBehaviour
{
    [Header("Reward Values (per agent, after /3 team split)")]
    [Tooltip("Reward delivered to EACH agent on a correct sort. Workflow sacred number: +1/3 = 0.333.")]
    [SerializeField] private float _correctSortReward = 1f / 3f;

    [Tooltip("Reward delivered to EACH agent on an incorrect sort. Workflow sacred number: -1.5/3 = -0.5.")]
    [SerializeField] private float _incorrectSortReward = -1.5f / 3f;

    [Tooltip("Reward delivered to EACH agent on a missed (fallthrough) package. Workflow sacred number: -2/3 = -0.667.")]
    [SerializeField] private float _missedPackageReward = -2f / 3f;

    [Header("Wired References")]
    [Tooltip("All SortingAgents that receive shared reward. Typically size 3.")]
    [SerializeField] private SortingAgent[] _agents = new SortingAgent[3];

    [Tooltip("All DestinationZones in the scene, including the fallthrough zone at the trunk end.")]
    [SerializeField] private DestinationZone[] _destinationZones = new DestinationZone[4];

    // Cumulative event counters, reset per episode. Exposed read-only for
    // the Step 13 debug overlay. These are NOT rewards — just event tallies.
    public int CorrectSortEvents { get; private set; }
    public int IncorrectSortEvents { get; private set; }
    public int MissedPackageEvents { get; private set; }

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

        // Sacred-number sanity asserts. If these fire, someone changed
        // the reward values and is about to break the training signal.
        Debug.Assert(Mathf.Abs(_correctSortReward - (1f / 3f)) < 1e-4f,
            $"[RewardDistributor] correctSortReward diverges from +1/3 ({_correctSortReward}).");
        Debug.Assert(Mathf.Abs(_incorrectSortReward - (-1.5f / 3f)) < 1e-4f,
            $"[RewardDistributor] incorrectSortReward diverges from -1.5/3 ({_incorrectSortReward}).");
        Debug.Assert(Mathf.Abs(_missedPackageReward - (-2f / 3f)) < 1e-4f,
            $"[RewardDistributor] missedPackageReward diverges from -2/3 ({_missedPackageReward}).");
    }

    private void Start()
    {
        // Subscribe to all destination zones.
        for (int i = 0; i < _destinationZones.Length; i++)
        {
            DestinationZone zone = _destinationZones[i];
            if (zone == null) continue;

            if (zone.IsFallthrough)
            {
                zone.OnMissedPackage += HandleMissedPackage;
            }
            else
            {
                zone.OnCorrectSort += HandleCorrectSort;
                zone.OnIncorrectSort += HandleIncorrectSort;
            }
        }

        // Subscribe to episode reset to clear counters.
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
                zone.OnCorrectSort -= HandleCorrectSort;
                zone.OnIncorrectSort -= HandleIncorrectSort;
            }
        }

        if (EnvironmentManager.Instance != null)
        {
            EnvironmentManager.Instance.OnEpisodeReset -= HandleEpisodeReset;
        }
    }

    // ================================================================
    // Event handlers — distribute shared reward
    // ================================================================

    private void HandleCorrectSort(Package pkg)
    {
        CorrectSortEvents++;
        DistributeReward(_correctSortReward);

    }

    private void HandleIncorrectSort(Package pkg)
    {
        IncorrectSortEvents++;
        DistributeReward(_incorrectSortReward);

    }

    private void HandleMissedPackage(Package pkg)
    {
        MissedPackageEvents++;
        DistributeReward(_missedPackageReward);

    }

    private void DistributeReward(float rewardPerAgent)
    {
        for (int i = 0; i < _agents.Length; i++)
        {
            SortingAgent agent = _agents[i];
            if (agent == null) continue;
            agent.AddReward(rewardPerAgent);
        }
    }

    private void HandleEpisodeReset()
    {
        
        CorrectSortEvents = 0;
        IncorrectSortEvents = 0;
        MissedPackageEvents = 0;
    }


}