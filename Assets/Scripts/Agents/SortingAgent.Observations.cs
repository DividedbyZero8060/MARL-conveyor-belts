using Unity.MLAgents.Sensors;
using UnityEngine;

/// <summary>
/// SortingAgent observation vector assembly (Step 08, file 4c).
///
/// Full observability  (38 floats):
///   [0]      gate_state
///   [1]      cooldown
///   [2]      belt_speed
///   [3..6)   dest_mapping one-hot (THIS branch)
///   [6..31)  5 package slots × 5 floats
///   [31..35) peer features: 2 × gate_state + 2 × nearest_package_distance
///   [35..38) 3 × branch congestion
///
/// Partial observability (34 floats):
///   [0]      gate_state
///   [1]      cooldown
///   [2]      belt_speed
///   [3..6)   dest_mapping one-hot (THIS branch)
///   [6..31)  5 package slots × 5 floats
///   [31..34) 3 × branch congestion
///
/// Index constants live in ObsIndices.cs. Do NOT hardcode offsets here.
/// </summary>
public partial class SortingAgent
{
    // Scratch buffer sized to the max possible observation (full obs = 38).
    // Reused every call to avoid per-frame allocation.
    private readonly float[] _obsBuffer = new float[ObsIndices.FullObsSize];

    public override void CollectObservations(VectorSensor sensor)
    {
        bool partial = EnvironmentManager.Instance != null
                       && EnvironmentManager.Instance.PartialObservability;

        int obsSize = partial ? ObsIndices.PartialObsSize : ObsIndices.FullObsSize;

        // Zero the active slice of the buffer (cheap — 34 or 38 floats).
        for (int i = 0; i < obsSize; i++) _obsBuffer[i] = 0f;

        // --------------------------------------------------------------
        // [0] gate_state : 0 / 0.5 / 1
        // --------------------------------------------------------------
        _obsBuffer[ObsIndices.GateState] = NormalisedGateState;

        // --------------------------------------------------------------
        // [1] cooldown : [0, 1] normalised remaining
        // --------------------------------------------------------------
        _obsBuffer[ObsIndices.Cooldown] =
            _gate != null ? Mathf.Clamp01(_gate.NormalisedCooldownRemaining) : 0f;

        // --------------------------------------------------------------
        // [2] belt_speed : currentSpeed / maxSpeed, clamped [0, 1]
        // --------------------------------------------------------------
        if (BeltSpeedController.Instance != null)
        {
            float maxSpeed = BeltSpeedController.Instance.MaxSpeed;
            float speed = BeltSpeedController.Instance.CurrentSpeed;
            _obsBuffer[ObsIndices.BeltSpeed] =
                maxSpeed > 0f ? Mathf.Clamp01(speed / maxSpeed) : 0f;
        }

        // --------------------------------------------------------------
        // [3..6) destination mapping one-hot for THIS branch
        // --------------------------------------------------------------
        if (EnvironmentManager.Instance != null)
        {
            DestinationLabel myDest =
                EnvironmentManager.Instance.GetDestinationForBranch(_branchIndex);

            _obsBuffer[ObsIndices.DestMappingStart + 0] =
                (myDest == DestinationLabel.DestA) ? 1f : 0f;
            _obsBuffer[ObsIndices.DestMappingStart + 1] =
                (myDest == DestinationLabel.DestB) ? 1f : 0f;
            _obsBuffer[ObsIndices.DestMappingStart + 2] =
                (myDest == DestinationLabel.DestC) ? 1f : 0f;
        }

        // --------------------------------------------------------------
        // [6..31) 5 package slots × 5 floats = 25 floats
        // --------------------------------------------------------------
        _packageDetector.Refresh(_overlappingPackages);
        _packageDetector.WriteObservations(_obsBuffer, ObsIndices.PackageSlotsStart);

        // Update the cached nearest-package distance for peer reads.
        // Slot 0 index 1 is the normalised distance of the closest package.
        // If slot 0 is empty (present==0), fall back to 1.0 (nothing in range).
        float slot0Present = _obsBuffer[ObsIndices.PackageSlotsStart + 0];
        float slot0Dist = _obsBuffer[ObsIndices.PackageSlotsStart + 1];
        SetCachedNearestPackageDistance(slot0Present > 0.5f ? slot0Dist : 1f);

        // --------------------------------------------------------------
        // Peer features (FULL observability only)
        // [31] peer0 gate_state
        // [32] peer1 gate_state
        // [33] peer0 nearest_package_distance
        // [34] peer1 nearest_package_distance
        // --------------------------------------------------------------
        if (!partial)
        {
            int start = ObsIndices.OtherAgentsStartFull;
            SortingAgent peer0 = _peerAgents != null && _peerAgents.Length > 0 ? _peerAgents[0] : null;
            SortingAgent peer1 = _peerAgents != null && _peerAgents.Length > 1 ? _peerAgents[1] : null;

            _obsBuffer[start + 0] = peer0 != null ? peer0.NormalisedGateState : 0f;
            _obsBuffer[start + 1] = peer1 != null ? peer1.NormalisedGateState : 0f;
            _obsBuffer[start + 2] = peer0 != null ? peer0.NearestPackageDistance : 1f;
            _obsBuffer[start + 3] = peer1 != null ? peer1.NearestPackageDistance : 1f;
        }

        // --------------------------------------------------------------
        // Congestion (3 floats) : always present, at different indices per mode
        // Full    : indices 35..38
        // Partial : indices 31..34
        // --------------------------------------------------------------
        int congestionStart = partial
            ? ObsIndices.CongestionStartPartial
            : ObsIndices.CongestionStartFull;

        for (int b = 0; b < ObsIndices.CongestionWidth; b++)
        {
            BranchTracker tracker = _allBranchTrackers[b];
            _obsBuffer[congestionStart + b] =
                tracker != null ? Mathf.Clamp01(tracker.NormalisedCongestion) : 0f;
        }

        // --------------------------------------------------------------
        // Semantic asserts (editor only — fires on any malformed float)
        // --------------------------------------------------------------
#if UNITY_EDITOR
        // Gate state must be 0, 0.5, or 1
        float gs = _obsBuffer[ObsIndices.GateState];
        Debug.Assert(gs == 0f || gs == 0.5f || gs == 1f,
            $"[SortingAgent {_branchIndex}] gate_state out of spec: {gs}");

        // Cooldown and belt speed in [0, 1]
        Debug.Assert(_obsBuffer[ObsIndices.Cooldown] >= 0f && _obsBuffer[ObsIndices.Cooldown] <= 1f,
            $"[SortingAgent {_branchIndex}] cooldown out of [0,1]: {_obsBuffer[ObsIndices.Cooldown]}");
        Debug.Assert(_obsBuffer[ObsIndices.BeltSpeed] >= 0f && _obsBuffer[ObsIndices.BeltSpeed] <= 1f,
            $"[SortingAgent {_branchIndex}] belt_speed out of [0,1]: {_obsBuffer[ObsIndices.BeltSpeed]}");

        // Destination mapping must be a valid one-hot (sums to ~1)
        float destSum = _obsBuffer[ObsIndices.DestMappingStart + 0]
                      + _obsBuffer[ObsIndices.DestMappingStart + 1]
                      + _obsBuffer[ObsIndices.DestMappingStart + 2];
        Debug.Assert(Mathf.Abs(destSum - 1f) < 0.001f,
            $"[SortingAgent {_branchIndex}] dest mapping is not one-hot (sum={destSum})");

        // Each populated package slot: present=1 AND one-hot destination sums to 1
        for (int s = 0; s < ObsIndices.PackageSlotCount; s++)
        {
            int b0 = ObsIndices.PackageSlotsStart + s * ObsIndices.PackageSlotWidth;
            float present = _obsBuffer[b0 + 0];
            if (present < 0.5f) continue; // empty slot, skip
            float pDest = _obsBuffer[b0 + 2] + _obsBuffer[b0 + 3] + _obsBuffer[b0 + 4];
            Debug.Assert(Mathf.Abs(pDest - 1f) < 0.001f,
                $"[SortingAgent {_branchIndex}] package slot {s} dest not one-hot (sum={pDest})");
            float pDist = _obsBuffer[b0 + 1];
            Debug.Assert(pDist >= 0f && pDist <= 1f,
                $"[SortingAgent {_branchIndex}] package slot {s} distance out of [0,1]: {pDist}");
        }

        // Congestion floats in [0, 1]
        for (int b = 0; b < ObsIndices.CongestionWidth; b++)
        {
            float c = _obsBuffer[congestionStart + b];
            Debug.Assert(c >= 0f && c <= 1f,
                $"[SortingAgent {_branchIndex}] congestion[{b}] out of [0,1]: {c}");
        }
#endif

        // --------------------------------------------------------------
        // Commit to sensor
        // --------------------------------------------------------------
        for (int i = 0; i < obsSize; i++)
        {
            sensor.AddObservation(_obsBuffer[i]);
        }
    }
}