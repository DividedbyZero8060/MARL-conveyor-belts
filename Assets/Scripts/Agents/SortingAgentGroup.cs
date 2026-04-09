using Unity.MLAgents;
using UnityEngine;

/// <summary>
/// Registers all SortingAgents into a SimpleMultiAgentGroup so that
/// episode boundaries are synchronised across agents and MA-POCA's
/// cooperative value decomposition sees them as a team.
///
/// The group handle is exposed read-only via Group so EnvironmentManager
/// can call EndGroupEpisode() on environment reset, ensuring ML-Agents
/// terminates the episode for all three agents atomically — essential
/// for correct terminal_steps handling on the Python side (Step 12).
///
/// Registration happens in Start() after BranchTracker / SortingAgent /
/// RewardDistributor have all subscribed to their own events, to keep
/// the scene-load initialisation order consistent.
/// </summary>
public class SortingAgentGroup : MonoBehaviour
{
    [Header("Wired References")]
    [Tooltip("All SortingAgents that are members of this cooperative group. Typically size 3.")]
    [SerializeField] private SortingAgent[] _agents = new SortingAgent[3];

    private SimpleMultiAgentGroup _group;

    /// <summary>
    /// The underlying MultiAgentGroup. Use EnvironmentManager.Instance to
    /// access this via the environment reset path; direct external use is
    /// discouraged.
    /// </summary>
    public SimpleMultiAgentGroup Group => _group;

    private void Awake()
    {
        Debug.Assert(_agents != null && _agents.Length > 0,
            "[SortingAgentGroup] _agents is empty.", this);
        for (int i = 0; i < _agents.Length; i++)
        {
            Debug.Assert(_agents[i] != null,
                $"[SortingAgentGroup] _agents[{i}] is null.", this);
        }

        _group = new SimpleMultiAgentGroup();
    }

    private void Start()
    {
        // Register every agent. SimpleMultiAgentGroup is idempotent on
        // duplicate registrations in current ML-Agents, but we guard anyway.
        for (int i = 0; i < _agents.Length; i++)
        {
            if (_agents[i] != null)
            {
                _group.RegisterAgent(_agents[i]);
            }
        }
        
    }

    private void OnDestroy()
    {
        // SimpleMultiAgentGroup implements IDisposable in recent ML-Agents.
        // Explicit dispose releases the registered agents cleanly on scene unload.
        _group?.Dispose();
        _group = null;
    }

    /// <summary>
    /// Called from EnvironmentManager.ResetEpisode() to end the ML-Agents
    /// episode for all agents at once, so Python's terminal_steps receives
    /// the final reward for every agent on the same step.
    /// </summary>
    public void EndEpisodeForAll()
    {
        if (_group == null) return;
        _group.EndGroupEpisode();
    }
}