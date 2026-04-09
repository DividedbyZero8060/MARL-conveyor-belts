using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Detects packages in front of the agent on the trunk belt and exposes them
/// as a fixed-size, distance-sorted slot array for the observation vector.
///
/// Slot layout (5 slots × 5 floats = 25 floats, matching ObsIndices):
///   [present, normalisedDistance, isDestA, isDestB, isDestC]
///
/// Rules (workflow Step 08):
///   - Slot 0 = closest package, Slot 4 = farthest.
///   - Empty slots are zero-padded.
///   - Hysteresis: two slots do NOT swap unless the distance difference exceeds
///     SlotSwapHysteresisMeters (0.15m). Prevents per-frame flicker when two
///     packages are nearly equidistant.
///   - Packages behind the agent (negative along-belt distance) are ignored.
///   - Distance is normalised by _maxDetectionRange to keep observation in [0,1].
/// </summary>
[RequireComponent(typeof(BoxCollider))]
public class PackageDetector : MonoBehaviour
{
    [Header("Detection")]
    [Tooltip("Maximum distance (metres) along the belt to detect packages. Also the normalisation divisor.")]
    [SerializeField] private float _maxDetectionRange = 15f;

    [Tooltip("Forward direction along the trunk belt (world space). Set from agent at init.")]
    [SerializeField] private Vector3 _beltForward = Vector3.forward;

    // Fixed-size slot state. null means empty.
    private readonly Package[] _slots = new Package[ObsIndices.PackageSlotCount];

    // Cached distance per slot to enforce hysteresis on the NEXT refresh.
    // Only meaningful where _slots[i] != null.
    private readonly float[] _slotDistances = new float[ObsIndices.PackageSlotCount];

    // Working buffers reused each frame to avoid GC.
    private readonly List<Package> _candidatesInRange = new List<Package>(32);
    private readonly List<float> _candidateDistances = new List<float>(32);

    /// <summary>
    /// Set the forward direction that defines "ahead" along the belt.
    /// Called by SortingAgent during Initialize.
    /// </summary>
    public void SetBeltForward(Vector3 worldForward)
    {
        _beltForward = worldForward.normalized;
    }

    /// <summary>
    /// Rebuild the slot list from the current set of trigger-overlapping packages.
    /// Call once per CollectObservations.
    /// </summary>
    /// <param name="overlapping">Packages currently inside this detector's trigger volume.</param>
    public void Refresh(IReadOnlyList<Package> overlapping)
    {
  
        // 1. Filter to packages AHEAD of the agent within detection range.
        _candidatesInRange.Clear();
        _candidateDistances.Clear();

        Vector3 origin = transform.position;
        for (int i = 0; i < overlapping.Count; i++)
        {
            Package pkg = overlapping[i];
            if (pkg == null) continue;

            Vector3 toPkg = pkg.transform.position - origin;
            float alongBelt = Vector3.Dot(toPkg, _beltForward);
            // Package is upstream (approaching) when alongBelt < 0. We want
            // distance until it reaches the agent, so flip sign. Packages
            // already past the agent (alongBelt > 0) are rejected.
            float distanceToAgent = -alongBelt;

           
            if (distanceToAgent < 0f) continue;                  // past agent
            if (distanceToAgent > _maxDetectionRange) continue;  // too far upstream
            _candidatesInRange.Add(pkg);
            _candidateDistances.Add(distanceToAgent);
        }

        // 2. Sort candidates by ascending along-belt distance.
        //    Small N (≤ ~15 typical), simple insertion sort keeps it allocation-free.
        int n = _candidatesInRange.Count;
        for (int i = 1; i < n; i++)
        {
            float d = _candidateDistances[i];
            Package p = _candidatesInRange[i];
            int j = i - 1;
            while (j >= 0 && _candidateDistances[j] > d)
            {
                _candidateDistances[j + 1] = _candidateDistances[j];
                _candidatesInRange[j + 1] = _candidatesInRange[j];
                j--;
            }
            _candidateDistances[j + 1] = d;
            _candidatesInRange[j + 1] = p;
        }

        // 3. Build the new slot assignment with hysteresis.
        //    For each slot index i, pick candidate[i] if present. But: if the
        //    previous occupant of slot i is still a candidate AND the incoming
        //    candidate is within the hysteresis threshold, KEEP the previous
        //    occupant to avoid jitter swaps.
        Package[] newSlots = new Package[ObsIndices.PackageSlotCount];
        float[] newDistances = new float[ObsIndices.PackageSlotCount];

        // Use a small bool array to mark which candidates are already placed.
        bool[] candidateUsed = new bool[n];

        for (int slotIdx = 0; slotIdx < ObsIndices.PackageSlotCount; slotIdx++)
        {
            Package prevOccupant = _slots[slotIdx];

            // Is the previous occupant still a valid (in-range) candidate?
            int prevCandidateIdx = -1;
            if (prevOccupant != null)
            {
                for (int c = 0; c < n; c++)
                {
                    if (!candidateUsed[c] && _candidatesInRange[c] == prevOccupant)
                    {
                        prevCandidateIdx = c;
                        break;
                    }
                }
            }

            // Find the first not-yet-used candidate in sorted order.
            int firstFreeIdx = -1;
            for (int c = 0; c < n; c++)
            {
                if (!candidateUsed[c]) { firstFreeIdx = c; break; }
            }

            if (firstFreeIdx == -1)
            {
                // No more candidates → empty slot.
                newSlots[slotIdx] = null;
                newDistances[slotIdx] = 0f;
                continue;
            }

            // Hysteresis decision: if the previous occupant is still in range
            // AND the natural-order candidate is a DIFFERENT package AND the
            // distance gap is below the hysteresis threshold, keep the previous
            // occupant in this slot.
            int chosenIdx;
            if (prevCandidateIdx != -1
                && prevCandidateIdx != firstFreeIdx
                && Mathf.Abs(_candidateDistances[prevCandidateIdx] - _candidateDistances[firstFreeIdx])
                   < ObsIndices.SlotSwapHysteresisMeters)
            {
                chosenIdx = prevCandidateIdx;
            }
            else
            {
                chosenIdx = firstFreeIdx;
            }

            newSlots[slotIdx] = _candidatesInRange[chosenIdx];
            newDistances[slotIdx] = _candidateDistances[chosenIdx];
            candidateUsed[chosenIdx] = true;
        }

        // Commit.
        for (int i = 0; i < ObsIndices.PackageSlotCount; i++)
        {
            _slots[i] = newSlots[i];
            _slotDistances[i] = newDistances[i];
        }
    }

    /// <summary>
    /// Write the 25-float slot block into the observation buffer starting at offset.
    /// Layout per slot: [present, normDist, isDestA, isDestB, isDestC].
    /// </summary>
    public void WriteObservations(float[] buffer, int offset)
    {
        Debug.Assert(buffer != null, "[PackageDetector] buffer is null.");
        Debug.Assert(offset + (ObsIndices.PackageSlotCount * ObsIndices.PackageSlotWidth) <= buffer.Length,
            "[PackageDetector] buffer too small for package slot block.");

        for (int slotIdx = 0; slotIdx < ObsIndices.PackageSlotCount; slotIdx++)
        {
            int baseIdx = offset + slotIdx * ObsIndices.PackageSlotWidth;
            Package pkg = _slots[slotIdx];

            if (pkg == null)
            {
                // Zero-pad empty slot.
                buffer[baseIdx + 0] = 0f;
                buffer[baseIdx + 1] = 0f;
                buffer[baseIdx + 2] = 0f;
                buffer[baseIdx + 3] = 0f;
                buffer[baseIdx + 4] = 0f;
                continue;
            }

            float normDist = Mathf.Clamp01(_slotDistances[slotIdx] / _maxDetectionRange);
            DestinationLabel label = pkg.DestinationLabel;

            buffer[baseIdx + 0] = 1f;                                      // present
            buffer[baseIdx + 1] = normDist;                                // distance
            buffer[baseIdx + 2] = (label == DestinationLabel.DestA) ? 1f : 0f;
            buffer[baseIdx + 3] = (label == DestinationLabel.DestB) ? 1f : 0f;
            buffer[baseIdx + 4] = (label == DestinationLabel.DestC) ? 1f : 0f;
        }
    }

    /// <summary>
    /// Clear all slot state. Call from SortingAgent.OnEpisodeBegin.
    /// </summary>
    public void ResetState()
    {
        for (int i = 0; i < ObsIndices.PackageSlotCount; i++)
        {
            _slots[i] = null;
            _slotDistances[i] = 0f;
        }
    }

#if UNITY_EDITOR
    private void OnDrawGizmosSelected()
    {
        BoxCollider box = GetComponent<BoxCollider>();
        if (box == null) return;
        Gizmos.color = new Color(1f, 1f, 0f, 0.2f);
        Gizmos.matrix = transform.localToWorldMatrix;
        Gizmos.DrawCube(box.center, box.size);
        Gizmos.color = Color.yellow;
        Gizmos.DrawWireCube(box.center, box.size);
    }
#endif
}