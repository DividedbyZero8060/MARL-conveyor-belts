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

        // Zero the active slice of the buffer.
        for (int i = 0; i < obsSize; i++) _obsBuffer[i] = 0f;

        // Build the peer arrays for this agent's context. Agent N's peers
        // are the other two agents in _peerAgents; we read their cached
        // NormalisedGateState and NearestPackageDistance (updated last frame).
        float[] peerGates = new float[2];
        float[] peerDistances = new float[2];
        SortingAgent peer0 = _peerAgents != null && _peerAgents.Length > 0 ? _peerAgents[0] : null;
        SortingAgent peer1 = _peerAgents != null && _peerAgents.Length > 1 ? _peerAgents[1] : null;
        peerGates[0] = peer0 != null ? peer0.NormalisedGateState : 0f;
        peerGates[1] = peer1 != null ? peer1.NormalisedGateState : 0f;
        peerDistances[0] = peer0 != null ? peer0.NearestPackageDistance : 1f;
        peerDistances[1] = peer1 != null ? peer1.NearestPackageDistance : 1f;

        // Assemble the context for this branch.
        ObservationBuilder.ObservationContext ctx = new ObservationBuilder.ObservationContext
        {
            BranchIndex = _branchIndex,
            Gate = _gate,
            PackageDetector = _packageDetector,
            OverlappingPackages = _overlappingPackages,
            AllBranchTrackers = _allBranchTrackers,
            PeerGateStates = peerGates,
            PeerNearestDistances = peerDistances,
            NearestPackageDistanceOut = 0f,
        };

        // Write the 38- or 34-float observation block starting at offset 0.
        ObservationBuilder.WriteBranchObservation(ref ctx, _obsBuffer, 0, partial);

        // Propagate the cached nearest-package distance so peers can read it.
        SetCachedNearestPackageDistance(ctx.NearestPackageDistanceOut);

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

        // Destination mapping must be a valid one-hot
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
            if (present < 0.5f) continue;
            float pDest = _obsBuffer[b0 + 2] + _obsBuffer[b0 + 3] + _obsBuffer[b0 + 4];
            Debug.Assert(Mathf.Abs(pDest - 1f) < 0.001f,
                $"[SortingAgent {_branchIndex}] package slot {s} dest not one-hot (sum={pDest})");
            float pDist = _obsBuffer[b0 + 1];
            Debug.Assert(pDist >= 0f && pDist <= 1f,
                $"[SortingAgent {_branchIndex}] package slot {s} distance out of [0,1]: {pDist}");
        }

        // Congestion floats in [0, 1]
        int congestionStart = partial
            ? ObsIndices.CongestionStartPartial
            : ObsIndices.CongestionStartFull;
        for (int b = 0; b < ObsIndices.CongestionWidth; b++)
        {
            float c = _obsBuffer[congestionStart + b];
            Debug.Assert(c >= 0f && c <= 1f,
                $"[SortingAgent {_branchIndex}] congestion[{b}] out of [0,1]: {c}");
        }
#endif

        // Commit to sensor
        for (int i = 0; i < obsSize; i++)
        {
            sensor.AddObservation(_obsBuffer[i]);
        }
    }
}