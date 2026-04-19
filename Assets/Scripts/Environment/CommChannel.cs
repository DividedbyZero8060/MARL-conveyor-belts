using UnityEngine;

/// <summary>
/// Scene-level shared message buffer for communication ablation (Step 17).
///
/// Semantics (Option A — same-step write, next-step read):
///   - During OnActionReceived (end of decision step), each agent writes its
///     own message floats via WriteMessage(branchIndex, message[]).
///   - During CollectObservations (start of next decision step), each agent
///     reads its peers' messages via ReadMessage(peerBranchIndex).
///   - Messages produced in step T are observed by peers in step T+1, giving
///     a one-decision-step communication lag. Consistent with how peer gate
///     states are read via NearestPackageDistance in full-obs mode.
///
/// State:
///   - _messages[branchIdx][floatIdx] — latest message each agent has written.
///   - Cleared to zero on every episode reset (wired via EnvironmentManager
///     .OnEpisodeReset).
///
/// Bandwidth is inferred from the first call to WriteMessage (the message
/// float[] length). Valid bandwidths: 0 (unused), 1, 3. If bandwidth=0 the
/// channel is effectively a no-op — SortingAgent's observation/action code
/// skips CommChannel access when _commBandwidth == 0.
///
/// The component is a singleton per scene to mirror EnvironmentManager's
/// singleton pattern — there is exactly one comm channel per training area.
/// </summary>
public class CommChannel : MonoBehaviour
{
    public static CommChannel Instance { get; private set; }

    [Header("Configuration")]
    [Tooltip("Number of floats per message. Must match SortingAgent._commBandwidth " +
             "on all three agents and the Python --comm-bandwidth flag. Valid: 0, 1, 3.")]
    [SerializeField] private int _bandwidth = 0;

    [Tooltip("Number of agents (rows in the message buffer). Fixed at 3 for this project.")]
    [SerializeField] private int _numAgents = 3;

    /// <summary>Message buffer: [branchIndex][floatIndex]. Zero when bandwidth=0.</summary>
    private float[][] _messages;

    public int Bandwidth => _bandwidth;
    public int NumAgents => _numAgents;

    private void Awake()
    {
        if (Instance != null && Instance != this) { Destroy(gameObject); return; }
        Instance = this;

        Debug.Assert(_bandwidth == 0 || _bandwidth == 1 || _bandwidth == 3,
            $"[CommChannel] bandwidth must be 0, 1, or 3; got {_bandwidth}.", this);
        Debug.Assert(_numAgents > 0,
            $"[CommChannel] numAgents must be > 0; got {_numAgents}.", this);

        // Allocate buffer even for bandwidth=0 (empty rows) so code paths
        // that read without checking bandwidth don't null-crash.
        _messages = new float[_numAgents][];
        for (int i = 0; i < _numAgents; i++)
        {
            _messages[i] = new float[Mathf.Max(_bandwidth, 0)];
        }

        Debug.Log($"[CommChannel] bandwidth={_bandwidth}, numAgents={_numAgents}.");
    }

    private void Start()
    {
        if (EnvironmentManager.Instance != null)
        {
            EnvironmentManager.Instance.OnEpisodeReset += HandleEpisodeReset;
        }
        else
        {
            Debug.LogError("[CommChannel] EnvironmentManager.Instance is null in Start.", this);
        }
    }

    private void OnDestroy()
    {
        if (EnvironmentManager.Instance != null)
        {
            EnvironmentManager.Instance.OnEpisodeReset -= HandleEpisodeReset;
        }
        if (Instance == this) Instance = null;
    }

    /// <summary>
    /// Write an agent's latest message into the buffer. Called from
    /// SortingAgent.OnActionReceived. No-op when bandwidth=0.
    /// </summary>
    public void WriteMessage(int branchIndex, float[] message)
    {
        if (_bandwidth == 0) return;
        if (branchIndex < 0 || branchIndex >= _numAgents) return;
        if (message == null || message.Length != _bandwidth)
        {
            Debug.LogWarning(
                $"[CommChannel] WriteMessage branch {branchIndex}: " +
                $"message length {(message == null ? -1 : message.Length)} != bandwidth {_bandwidth}", this);
            return;
        }

        // Copy to buffer so caller is free to mutate their array afterwards.
        for (int i = 0; i < _bandwidth; i++)
        {
            _messages[branchIndex][i] = Mathf.Clamp01(message[i]);
        }
    }

    /// <summary>
    /// Read an agent's latest message. Returns a zero array when bandwidth=0
    /// or when the branch index is invalid. Returned array is the internal
    /// buffer — do NOT mutate. Callers should copy if they need to hold on.
    /// </summary>
    public float[] ReadMessage(int branchIndex)
    {
        if (_bandwidth == 0 || branchIndex < 0 || branchIndex >= _numAgents)
        {
            return System.Array.Empty<float>();
        }
        return _messages[branchIndex];
    }

    private void HandleEpisodeReset()
    {
        // Zero all messages at episode boundaries. Without this, stale messages
        // from the end of the previous episode would leak into the first few
        // decision steps of the next episode.
        if (_messages == null) return;
        for (int i = 0; i < _messages.Length; i++)
        {
            if (_messages[i] == null) continue;
            for (int j = 0; j < _messages[i].Length; j++)
            {
                _messages[i][j] = 0f;
            }
        }
    }
}