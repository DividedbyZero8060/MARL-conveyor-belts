/// <summary>
/// Per-episode event counter contract shared by all reward distributor
/// variants. Implemented by RewardDistributor (cooperative, team reward)
/// and IndependentRewardDistributor (selfish, per-agent reward). Future
/// variants (e.g. a centralised-PPO single-agent distributor in Step 15c)
/// can implement this too.
///
/// DebugOverlay consumes this interface instead of a concrete type so the
/// overlay panel and custom TensorBoard metrics work regardless of which
/// distributor is currently active in the scene.
/// </summary>
public interface IEventCounter
{
    int CorrectSortEvents { get; }
    int IncorrectSortEvents { get; }
    int MissedPackageEvents { get; }

    /// <summary>
    /// Number of correct sorts attributed to the given branch this episode.
    /// Returns 0 for invalid branch indices.
    /// </summary>
    int GetCorrectSortsForBranch(int branchIndex);
}