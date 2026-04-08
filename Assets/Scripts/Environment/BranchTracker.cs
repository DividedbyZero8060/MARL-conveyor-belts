using System.Collections.Generic;

using UnityEngine;


    /// <summary>
    /// Tracks how many packages are currently on a single branch belt.
    ///
    /// Counting rule (workflow Step 08):
    ///   +1 on OnTriggerEnter when a Package enters the branch-entry trigger volume.
    ///   -1 when the paired DestinationZone fires OnCorrectSort or OnIncorrectSort
    ///      for a package we are currently tracking.
    ///   NO OnTriggerExit decrement — packages physically leave the small entry
    ///      trigger while still riding the branch belt, which would undercount.
    ///   All counters zeroed on EnvironmentManager.OnEpisodeReset.
    ///
    /// Exposes NormalisedCongestion = count / maxCapacity, clamped [0, 1],
    /// for the agent observation vector (ObsIndices.CongestionStart*).
    /// </summary>
    [RequireComponent(typeof(BoxCollider))]
    public class BranchTracker : MonoBehaviour
    {
        [Header("Branch Identity")]
        [Tooltip("Which branch this tracker belongs to (0, 1, or 2). Must match DiverterGate.BranchIndex.")]
        [SerializeField] private int _branchIndex;

        [Tooltip("DestinationZone at the end of this branch. Used to decrement on sort events.")]
        [SerializeField] private DestinationZone _pairedDestinationZone;

        [Header("Capacity")]
        [Tooltip("Maximum expected packages on this branch. Used to normalise congestion to [0,1].")]
        [SerializeField] private int _maxCapacity = 10;

        // Currently tracked package instances. HashSet for O(1) add/remove and duplicate-safety
        // in case a package re-enters the trigger due to physics jitter.
        private readonly HashSet<Package> _trackedPackages = new HashSet<Package>();

        /// <summary>Branch index this tracker belongs to (0..2).</summary>
        public int BranchIndex => _branchIndex;

        /// <summary>Raw count of packages currently on this branch.</summary>
        public int Count => _trackedPackages.Count;

        /// <summary>Congestion normalised to [0, 1] for the observation vector.</summary>
        public float NormalisedCongestion
        {
            get
            {
                if (_maxCapacity <= 0) return 0f;
                float n = (float)_trackedPackages.Count / _maxCapacity;
                return Mathf.Clamp01(n);
            }
        }

        private void Awake()
        {
            // Enforce trigger mode — workflow Step 08 requires trigger, not solid collider.
            BoxCollider box = GetComponent<BoxCollider>();
            if (!box.isTrigger)
            {
                Debug.LogWarning(
                    $"[BranchTracker] BoxCollider on '{name}' was not set to isTrigger. Forcing true.",
                    this);
                box.isTrigger = true;
            }

            Debug.Assert(_branchIndex >= 0 && _branchIndex <= 2,
                $"[BranchTracker] branchIndex must be in [0,2], got {_branchIndex}.", this);
            Debug.Assert(_pairedDestinationZone != null,
                $"[BranchTracker] '{name}' has no paired DestinationZone assigned.", this);
            Debug.Assert(_maxCapacity > 0,
                $"[BranchTracker] maxCapacity must be > 0, got {_maxCapacity}.", this);
        }

    private void Start()
    {
        if (_pairedDestinationZone != null)
        {
            _pairedDestinationZone.OnCorrectSort += HandlePackageLeftBranch;
            _pairedDestinationZone.OnIncorrectSort += HandlePackageLeftBranch;
        }

        if (EnvironmentManager.Instance != null)
        {
            EnvironmentManager.Instance.OnEpisodeReset += HandleEpisodeReset;
        }
        else
        {
            Debug.LogError($"[BranchTracker {_branchIndex}] EnvironmentManager.Instance is null in Start!", this);
        }
    }

    private void OnDestroy()
    {
        if (_pairedDestinationZone != null)
        {
            _pairedDestinationZone.OnCorrectSort -= HandlePackageLeftBranch;
            _pairedDestinationZone.OnIncorrectSort -= HandlePackageLeftBranch;
        }

        if (EnvironmentManager.Instance != null)
        {
            EnvironmentManager.Instance.OnEpisodeReset -= HandleEpisodeReset;
        }
    }

    private void OnTriggerEnter(Collider other)
        {
            // Increment when a Package enters this branch's entry volume.
            Package pkg = other.GetComponentInParent<Package>();
            if (pkg == null) return;

            // HashSet.Add returns false if already present — prevents double-count
            // from physics re-triggers at the trigger boundary.
            _trackedPackages.Add(pkg);
        }

        /// <summary>
        /// Called by paired DestinationZone sort events. Removes the package
        /// from the tracked set if present. Silently ignores packages we weren't
        /// tracking (e.g. packages that reached this zone from a different path).
        /// </summary>
        private void HandlePackageLeftBranch(Package pkg)
        {
            if (pkg == null) return;
            _trackedPackages.Remove(pkg);
        }

        private void HandleEpisodeReset()
        {
        
        _trackedPackages.Clear();
        }

#if UNITY_EDITOR
        private void OnDrawGizmosSelected()
        {
            BoxCollider box = GetComponent<BoxCollider>();
            if (box == null) return;
            Gizmos.color = new Color(0f, 1f, 1f, 0.25f);
            Gizmos.matrix = transform.localToWorldMatrix;
            Gizmos.DrawCube(box.center, box.size);
            Gizmos.color = Color.cyan;
            Gizmos.DrawWireCube(box.center, box.size);
        }

    
#endif
}

