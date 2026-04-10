using System.Collections.Generic;
using Unity.MLAgents;
using UnityEngine;

/// <summary>
/// In-game debug overlay + custom TensorBoard metric writer for Step 13.
///
/// Two jobs:
///
/// 1. OnGUI panel shown during Play mode:
///    - episode time remaining, episode index
///    - correct / incorrect / missed counters (from RewardDistributor)
///    - current belt speed (normalised)
///    - current destination mapping per branch
///    - per-agent gate state and training mode
///    - running sort accuracy
///
/// 2. Custom StatsRecorder metrics written on every episode end via
///    EnvironmentManager.OnEpisodeEnded. Exactly 8 metrics:
///      Environment/SortAccuracy
///      Environment/Throughput
///      Environment/AvgQueueLength
///      Agent0/GateActivationRate, Agent1/..., Agent2/...
///      Agent0/CounterfactualCredit, Agent1/..., Agent2/...
///
/// AvgQueueLength is a per-episode running average sampled every FixedUpdate
/// tick from all three BranchTrackers. The running accumulator is reset in
/// HandleEpisodeReset and the average is computed in HandleEpisodeEnded.
///
/// Package colour tinting (green = correct branch, red = wrong) runs every
/// frame in Update() against all in-flight packages. O(packages x branches)
/// per frame, trivially fast at the workflow's ~30 packages/minute arrival.
///
/// Counterfactual credit framing:
///   CounterfactualCredit_i = correct_sorts_routed_via_branch_i / total_correct_sorts
///   Documented in the thesis Methodology chapter as a branch-attributed
///   proxy for per-agent contribution. NOT added to reward.
/// </summary>
public class DebugOverlay : MonoBehaviour
{
    [Header("Wired References")]
    [Tooltip("EnvironmentManager for episode state and destination mapping.")]
    [SerializeField] private EnvironmentManager _environmentManager;

    [Tooltip("RewardDistributor for event counters and per-branch attribution.")]
    [SerializeField] private RewardDistributor _rewardDistributor;

    [Tooltip("All three BranchTrackers in branch-index order.")]
    [SerializeField] private BranchTracker[] _branchTrackers = new BranchTracker[3];

    [Tooltip("All three SortingAgents in branch-index order.")]
    [SerializeField] private SortingAgent[] _agents = new SortingAgent[3];

    [Tooltip("PackageSpawner, used to enumerate in-flight packages for colouring.")]
    [SerializeField] private PackageSpawner _packageSpawner;

    [Header("Display")]
    [Tooltip("On-screen panel position.")]
    [SerializeField] private Vector2 _panelPosition = new Vector2(10, 10);

    [Tooltip("Panel width in pixels.")]
    [SerializeField] private float _panelWidth = 360f;

    [Tooltip("Enable on-screen OnGUI panel. Disable for headless training runs.")]
    [SerializeField] private bool _showPanel = true;

    [Tooltip("Tint in-flight packages green/red based on destination match.")]
    [SerializeField] private bool _tintPackages = true;

    // ---- AvgQueueLength running accumulator
    private float _queueSumOverEpisode;
    private int _queueSamplesThisEpisode;

    // ---- Cached GUIStyle for readable text on any background
    private GUIStyle _panelStyle;
    private GUIStyle _headerStyle;
    private bool _stylesInitialised;

    // ---- Material property block reused each frame for tinting
    private static readonly int _colorProperty = Shader.PropertyToID("_BaseColor");
    private MaterialPropertyBlock _mpb;


    private static readonly Color _colorDestA = new Color(0.3f, 0.5f, 1.0f);  // blue
    private static readonly Color _colorDestB = new Color(0.3f, 0.9f, 0.3f);  // green
    private static readonly Color _colorDestC = new Color(1.0f, 0.6f, 0.2f);  // orange

    private StatsRecorder _stats;

    private void Awake()
    {
        Debug.Assert(_environmentManager != null, "[DebugOverlay] _environmentManager not assigned.", this);
        Debug.Assert(_rewardDistributor != null, "[DebugOverlay] _rewardDistributor not assigned.", this);
        Debug.Assert(_branchTrackers != null && _branchTrackers.Length == 3,
            "[DebugOverlay] _branchTrackers must have exactly 3 entries.", this);
        Debug.Assert(_agents != null && _agents.Length == 3,
            "[DebugOverlay] _agents must have exactly 3 entries.", this);
        Debug.Assert(_packageSpawner != null, "[DebugOverlay] _packageSpawner not assigned.", this);

        _mpb = new MaterialPropertyBlock();
        _stats = Academy.Instance.StatsRecorder;
    }

    private void Start()
    {
        if (EnvironmentManager.Instance != null)
        {
            EnvironmentManager.Instance.OnEpisodeReset += HandleEpisodeReset;
            EnvironmentManager.Instance.OnEpisodeEnded += HandleEpisodeEnded;
        }
    }

    private void OnDestroy()
    {
        if (EnvironmentManager.Instance != null)
        {
            EnvironmentManager.Instance.OnEpisodeReset -= HandleEpisodeReset;
            EnvironmentManager.Instance.OnEpisodeEnded -= HandleEpisodeEnded;
        }
    }

    // =================================================================
    // Per-tick sampling
    // =================================================================

    private void FixedUpdate()
    {
        // Sample the instantaneous queue length (sum of all three branch counts)
        // every physics tick. HandleEpisodeEnded divides by sample count for
        // a per-episode average.
        float sum = 0f;
        for (int b = 0; b < _branchTrackers.Length; b++)
        {
            if (_branchTrackers[b] == null) continue;
            sum += _branchTrackers[b].Count;
        }
        _queueSumOverEpisode += sum;
        _queueSamplesThisEpisode++;
    }

    private void Update()
    {
        if (_tintPackages)
        {
            TintInFlightPackages();
        }
    }

    // =================================================================
    // Package colouring
    // =================================================================

    private void TintInFlightPackages()
    {
        Package[] allPackages = _packageSpawner.GetComponentsInChildren<Package>(includeInactive: false);
        for (int i = 0; i < allPackages.Length; i++)
        {
            Package pkg = allPackages[i];
            if (pkg == null || !pkg.gameObject.activeInHierarchy) continue;

            MeshRenderer mr = pkg.GetComponent<MeshRenderer>();
            if (mr == null) continue;

            Color tint;
            switch (pkg.DestinationLabel)
            {
                case DestinationLabel.DestA: tint = _colorDestA; break;
                case DestinationLabel.DestB: tint = _colorDestB; break;
                case DestinationLabel.DestC: tint = _colorDestC; break;
                default: tint = Color.white; break;
            }

            mr.GetPropertyBlock(_mpb);
            _mpb.SetColor(_colorProperty, tint);
            mr.SetPropertyBlock(_mpb);
        }
    }

    // =================================================================
    // Episode reset / end — metrics and accumulator
    // =================================================================

    private void HandleEpisodeReset()
    {
        _queueSumOverEpisode = 0f;
        _queueSamplesThisEpisode = 0;
    }

    private void HandleEpisodeEnded()
    {
        int correct = _rewardDistributor.CorrectSortEvents;
        int incorrect = _rewardDistributor.IncorrectSortEvents;
        int missed = _rewardDistributor.MissedPackageEvents;
        int totalResolved = correct + incorrect + missed;

        float accuracy = totalResolved > 0 ? (float)correct / totalResolved : 0f;
        float throughput = correct;
        float avgQueue = _queueSamplesThisEpisode > 0
            ? _queueSumOverEpisode / _queueSamplesThisEpisode
            : 0f;

        _stats.Add("Environment/SortAccuracy", accuracy);
        _stats.Add("Environment/Throughput", throughput);
        _stats.Add("Environment/AvgQueueLength", avgQueue);

        // Capture per-agent rates into locals so we can log them.
        float[] rates = new float[_agents.Length];
        // Per-agent activation rate — consumed from agent counters.
        for (int i = 0; i < _agents.Length; i++)
        {
            if (_agents[i] == null) continue;
            float rate = _agents[i].ConsumeAndResetActivationRate();
            _stats.Add($"Agent{i}/GateActivationRate", rate);
        }

        // Counterfactual credit per branch: correct sorts via this branch /
        // total correct sorts. If no correct sorts happened, all credits = 0.
        for (int i = 0; i < 3; i++)
        {
            int branchCorrect = _rewardDistributor.GetCorrectSortsForBranch(i);
            float credit = correct > 0 ? (float)branchCorrect / correct : 0f;
            _stats.Add($"Agent{i}/CounterfactualCredit", credit);
        }
       
    }

    // =================================================================
    // OnGUI panel
    // =================================================================

    private void OnGUI()
    {
        if (!_showPanel) return;
        EnsureStyles();

        int correct = _rewardDistributor.CorrectSortEvents;
        int incorrect = _rewardDistributor.IncorrectSortEvents;
        int missed = _rewardDistributor.MissedPackageEvents;
        int totalResolved = correct + incorrect + missed;
        float accuracy = totalResolved > 0 ? (float)correct / totalResolved : 0f;

        float panelHeight = 260f;
        Rect panel = new Rect(_panelPosition.x, _panelPosition.y, _panelWidth, panelHeight);
        GUI.Box(panel, GUIContent.none);

        float y = _panelPosition.y + 6f;
        float x = _panelPosition.x + 10f;
        float lineHeight = 18f;

        GUI.Label(new Rect(x, y, _panelWidth - 20, lineHeight), "=== DebugOverlay ===", _headerStyle);
        y += lineHeight;

        GUI.Label(new Rect(x, y, _panelWidth - 20, lineHeight),
            $"Episode #{_environmentManager.EpisodeIndex}  " +
            $"t_remaining: {_environmentManager.EpisodeTimeRemaining:F1}s",
            _panelStyle);
        y += lineHeight;

        GUI.Label(new Rect(x, y, _panelWidth - 20, lineHeight),
            $"Correct: {correct}   Incorrect: {incorrect}   Missed: {missed}",
            _panelStyle);
        y += lineHeight;

        GUI.Label(new Rect(x, y, _panelWidth - 20, lineHeight),
            $"Accuracy: {accuracy * 100f:F1}%   Spawned: {_environmentManager.TotalSpawned}",
            _panelStyle);
        y += lineHeight;

        float beltNorm = 0f;
        if (BeltSpeedController.Instance != null && BeltSpeedController.Instance.MaxSpeed > 0f)
        {
            beltNorm = BeltSpeedController.Instance.CurrentSpeed / BeltSpeedController.Instance.MaxSpeed;
        }
        GUI.Label(new Rect(x, y, _panelWidth - 20, lineHeight),
            $"Belt speed: {beltNorm:F2}   Partial obs: {_environmentManager.PartialObservability}",
            _panelStyle);
        y += lineHeight;

        // Destination mapping
        y += 4f;
        GUI.Label(new Rect(x, y, _panelWidth - 20, lineHeight), "Mapping:", _headerStyle);
        y += lineHeight;
        for (int b = 0; b < 3; b++)
        {
            DestinationLabel lbl = _environmentManager.GetDestinationForBranch(b);
            int branchCorrect = _rewardDistributor.GetCorrectSortsForBranch(b);
            int queueCount = _branchTrackers[b] != null ? _branchTrackers[b].Count : 0;
            GUI.Label(new Rect(x + 10, y, _panelWidth - 30, lineHeight),
                $"Branch {b}: {lbl}   q={queueCount}   ok={branchCorrect}",
                _panelStyle);
            y += lineHeight;
        }

        // Agent gate state
        y += 4f;
        GUI.Label(new Rect(x, y, _panelWidth - 20, lineHeight), "Agents:", _headerStyle);
        y += lineHeight;
        for (int i = 0; i < _agents.Length; i++)
        {
            if (_agents[i] == null) continue;
            string gateStr;
            float g = _agents[i].NormalisedGateState;
            if (g <= 0.01f) gateStr = "Retracted";
            else if (g >= 0.99f) gateStr = "Deployed";
            else gateStr = "Transit";
            GUI.Label(new Rect(x + 10, y, _panelWidth - 30, lineHeight),
                $"Agent {i}: gate={gateStr}",
                _panelStyle);
            y += lineHeight;
        }
    }

    private void EnsureStyles()
    {
        if (_stylesInitialised) return;
        _panelStyle = new GUIStyle(GUI.skin.label);
        _panelStyle.normal.textColor = Color.white;
        _panelStyle.fontSize = 12;
        _headerStyle = new GUIStyle(_panelStyle);
        _headerStyle.fontStyle = FontStyle.Bold;
        _stylesInitialised = true;
    }
}