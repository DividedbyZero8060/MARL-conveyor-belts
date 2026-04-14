using System.Collections.Generic;
using System.IO;
using UnityEngine;

/// <summary>
/// Automated N-episode evaluation harness for baseline runs (Step 15).
///
/// Usage:
///   1. Set EnvironmentManager into evaluation mode (any agents that should
///      participate must be wired up before Play starts).
///   2. Set TargetEpisodes in the Inspector.
///   3. For heuristic baseline: set UseHeuristicAutomatic = true and the
///      runner will flip every wired SortingAgent into automatic heuristic
///      mode at Start().
///   4. Press Play. The runner subscribes to EnvironmentManager.OnEpisodeEnded,
///      accumulates per-episode metrics, and after TargetEpisodes resets
///      writes a summary CSV to the OutputPath and stops Play mode.
///
/// Per-episode metrics captured:
///   - episode_index
///   - correct_sorts
///   - incorrect_sorts
///   - missed_packages
///   - total_resolved
///   - sort_accuracy   = correct / total_resolved
///   - throughput      = correct (per 60s episode)
///
/// Aggregate summary written to a header in the CSV: mean, std, min, max.
///
/// Designed to be reusable for the DQN and centralised PPO baselines too —
/// they don't use UseHeuristicAutomatic but share the same episode-counting
/// and CSV-writing machinery.
/// </summary>
public class EvaluationRunner : MonoBehaviour
{
    [Header("Evaluation Settings")]
    [Tooltip("Number of episodes to run before stopping Play mode.")]
    [SerializeField] private int _targetEpisodes = 100;

    [Tooltip("Output CSV path relative to the project root. Directory created if missing.")]
    [SerializeField] private string _outputPath = "results/baseline_heuristic.csv";

    [Tooltip("If true, flips every wired SortingAgent into automatic heuristic " +
             "mode at Start. Use for the heuristic baseline; leave false for DQN/PPO.")]
    [SerializeField] private bool _useHeuristicAutomatic = false;

    [Header("Wired References")]
    [Tooltip("EnvironmentManager whose OnEpisodeEnded drives the runner.")]
    [SerializeField] private EnvironmentManager _environmentManager;

    [Tooltip("All agents that should be configured for automatic heuristic mode. " +
             "Only used when UseHeuristicAutomatic = true.")]
    [SerializeField] private SortingAgent[] _agents;

    // Per-episode records, accumulated until target reached.
    private struct EpisodeRecord
    {
        public int Index;
        public int Correct;
        public int Incorrect;
        public int Missed;
        public float Accuracy;
        public float Throughput;
    }

    private readonly List<EpisodeRecord> _records = new List<EpisodeRecord>();
    private bool _stopped = false;

    private void Awake()
    {
        Debug.Assert(_environmentManager != null,
            "[EvaluationRunner] _environmentManager not assigned.", this);
    }

    private void Start()
    {
        if (_environmentManager != null)
        {
            _environmentManager.OnEpisodeEnded += HandleEpisodeEnded;
        }

        if (_useHeuristicAutomatic && _agents != null)
        {
            int n = 0;
            for (int i = 0; i < _agents.Length; i++)
            {
                if (_agents[i] == null) continue;
                _agents[i].UseAutomaticHeuristic = true;
                n++;
            }
            Debug.Log($"[EvaluationRunner] Automatic heuristic enabled on {n} agents. " +
                      $"Target episodes: {_targetEpisodes}. Output: {_outputPath}");
        }
        else
        {
            Debug.Log($"[EvaluationRunner] Started in pass-through mode (no heuristic flip). " +
                      $"Target episodes: {_targetEpisodes}. Output: {_outputPath}");
        }
    }

    private void OnDestroy()
    {
        if (_environmentManager != null)
        {
            _environmentManager.OnEpisodeEnded -= HandleEpisodeEnded;
        }
    }

    private void HandleEpisodeEnded()
    {
        if (_stopped) return;

        int correct = _environmentManager.CorrectSorts;
        int incorrect = _environmentManager.IncorrectSorts;
        int missed = _environmentManager.MissedPackages;
        int total = correct + incorrect + missed;
        float accuracy = total > 0 ? (float)correct / total : 0f;
        float throughput = correct;

        _records.Add(new EpisodeRecord
        {
            Index = _records.Count + 1,
            Correct = correct,
            Incorrect = incorrect,
            Missed = missed,
            Accuracy = accuracy,
            Throughput = throughput,
        });

        if (_records.Count % 10 == 0 || _records.Count == _targetEpisodes)
        {
            Debug.Log($"[EvaluationRunner] Episode {_records.Count}/{_targetEpisodes} " +
                      $"(latest: correct={correct} incorrect={incorrect} " +
                      $"missed={missed} accuracy={accuracy:F3})");
        }

        if (_records.Count >= _targetEpisodes)
        {
            FinaliseAndStop();
        }
    }

    private void FinaliseAndStop()
    {
        _stopped = true;
        WriteSummary();

#if UNITY_EDITOR
        UnityEditor.EditorApplication.isPlaying = false;
#else
        Application.Quit();
#endif
    }

    private void WriteSummary()
    {
        // Aggregate stats
        int n = _records.Count;
        if (n == 0)
        {
            Debug.LogWarning("[EvaluationRunner] No records to write.");
            return;
        }

        float meanAcc = 0f, meanThr = 0f;
        float meanMissed = 0f, meanIncorrect = 0f;
        for (int i = 0; i < n; i++)
        {
            meanAcc += _records[i].Accuracy;
            meanThr += _records[i].Throughput;
            meanMissed += _records[i].Missed;
            meanIncorrect += _records[i].Incorrect;
        }
        meanAcc /= n;
        meanThr /= n;
        meanMissed /= n;
        meanIncorrect /= n;

        float varAcc = 0f, varThr = 0f;
        for (int i = 0; i < n; i++)
        {
            float dAcc = _records[i].Accuracy - meanAcc;
            float dThr = _records[i].Throughput - meanThr;
            varAcc += dAcc * dAcc;
            varThr += dThr * dThr;
        }
        float stdAcc = n > 1 ? Mathf.Sqrt(varAcc / (n - 1)) : 0f;
        float stdThr = n > 1 ? Mathf.Sqrt(varThr / (n - 1)) : 0f;

        // Ensure output directory exists
        string fullPath = Path.GetFullPath(_outputPath);
        string dir = Path.GetDirectoryName(fullPath);
        if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
        {
            Directory.CreateDirectory(dir);
        }

        using (StreamWriter writer = new StreamWriter(fullPath))
        {
            writer.WriteLine($"# Evaluation summary");
            writer.WriteLine($"# episodes: {n}");
            writer.WriteLine($"# mean_accuracy: {meanAcc:F4}");
            writer.WriteLine($"# std_accuracy: {stdAcc:F4}");
            writer.WriteLine($"# mean_throughput: {meanThr:F4}");
            writer.WriteLine($"# std_throughput: {stdThr:F4}");
            writer.WriteLine($"# mean_missed: {meanMissed:F4}");
            writer.WriteLine($"# mean_incorrect: {meanIncorrect:F4}");
            writer.WriteLine();
            writer.WriteLine("episode_index,correct,incorrect,missed,total_resolved,accuracy,throughput");
            for (int i = 0; i < n; i++)
            {
                EpisodeRecord r = _records[i];
                int total = r.Correct + r.Incorrect + r.Missed;
                writer.WriteLine($"{r.Index},{r.Correct},{r.Incorrect},{r.Missed},{total},{r.Accuracy:F4},{r.Throughput:F2}");
            }
        }

        Debug.Log($"[EvaluationRunner] Summary written to {fullPath}");
        Debug.Log($"[EvaluationRunner] Mean accuracy: {meanAcc:F4} ± {stdAcc:F4}");
        Debug.Log($"[EvaluationRunner] Mean throughput: {meanThr:F4} ± {stdThr:F4}");
    }
}