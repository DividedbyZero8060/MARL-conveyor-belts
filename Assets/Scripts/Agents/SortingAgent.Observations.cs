using Unity.MLAgents.Sensors;
using UnityEngine;

/// <summary>
/// SortingAgent observation vector assembly (Step 08 → extended Step 17).
///
/// Full observability  (38 floats, no comm):
///   [0]      gate_state
///   [1]      cooldown
///   [2]      belt_speed
///   [3..6)   dest_mapping one-hot (THIS branch)
///   [6..31)  5 package slots × 5 floats
///   [31..35) peer features: 2 × gate_state + 2 × nearest_package_distance
///   [35..38) 3 × branch congestion
///
/// Partial observability, no comm (34 floats):
///   [0..6)   same as above
///   [6..31)  5 package slots × 5 floats
///   [31..34) 3 × branch congestion
///
/// Partial observability, comm bandwidth=1 (36 floats):
///   [0..34)  same as partial no-comm
///   [34..36) peer messages: peer0_msg[0], peer1_msg[0]
///
/// Partial observability, comm bandwidth=3 (40 floats):
///   [0..34)  same as partial no-comm
///   [34..40) peer messages: peer0_msg[0..3), peer1_msg[0..3)
///
/// Comm variants require partial observability — full obs already exposes
/// peer gate state and distance, making messaging redundant.
///
/// Index constants live in ObsIndices.cs. Do NOT hardcode offsets here.
/// </summary>
public partial class SortingAgent
{
    // Scratch buffer sized to the max possible observation (40 floats = partial+comm3).
    // Reused every call to avoid per-frame allocation.
    private const int MaxObsBufferSize = 40;
    private readonly float[] _obsBuffer = new float[MaxObsBufferSize];

    public override void CollectObservations(VectorSensor sensor)
    {
        bool partial = EnvironmentManager.Instance != null
                       && EnvironmentManager.Instance.PartialObservability;

        // Base observation size: 34 (partial) or 38 (full).
        int baseObsSize = partial ? ObsIndices.PartialObsSize : ObsIndices.FullObsSize;

        // Comm message floats are appended only in partial-obs mode with bandwidth > 0.
        // Two peers × bandwidth floats each.
        int commFloats = (partial && _commBandwidth > 0) ? (2 * _commBandwidth) : 0;
        int obsSize = baseObsSize + commFloats;

        Debug.Assert(obsSize <= MaxObsBufferSize,
            $"[SortingAgent {_branchIndex}] obsSize {obsSize} exceeds buffer {MaxObsBufferSize}");

        // Zero the active slice of the buffer.
        for (int i = 0; i < obsSize; i++) _obsBuffer[i] = 0f;

        // Build the peer arrays for full-obs mode. In partial-obs mode these
        // are unused by ObservationBuilder but we still compute them cheaply.
        float[] peerGates = new float[2];
        float[] peerDistances = new float[2];
        SortingAgent peer0 = _peerAgents != null && _peerAgents.Length > 0 ? _peerAgents[0] : null;
        SortingAgent peer1 = _peerAgents != null && _peerAgents.Length > 1 ? _peerAgents[1] : null;
        peerGates[0] = peer0 != null ? peer0.NormalisedGateState : 0f;
        peerGates[1] = peer1 != null ? peer1.NormalisedGateState : 0f;
        peerDistances[0] = peer0 != null ? peer0.NearestPackageDistance : 1f;
        peerDistances[1] = peer1 != null ? peer1.NearestPackageDistance : 1f;

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

        // Write the 34- or 38-float base block starting at offset 0.
        ObservationBuilder.WriteBranchObservation(ref ctx, _obsBuffer, 0, partial);

        SetCachedNearestPackageDistance(ctx.NearestPackageDistanceOut);

        // Append peer messages (partial-obs comm variants only).
        // Layout matches Python critic_input.py's assumption:
        // messages start immediately after congestion at index 34.
        if (commFloats > 0 && CommChannel.Instance != null)
        {
            // Peer branch indices: the other two agents, in Inspector-assigned
            // _peerAgents order. We use the peers' branch indices (not their
            // array positions) so messages are correctly identified in the
            // buffer regardless of wiring order.
            int peer0Branch = peer0 != null ? peer0.BranchIndex : -1;
            int peer1Branch = peer1 != null ? peer1.BranchIndex : -1;

            float[] msg0 = peer0Branch >= 0 ? CommChannel.Instance.ReadMessage(peer0Branch) : System.Array.Empty<float>();
            float[] msg1 = peer1Branch >= 0 ? CommChannel.Instance.ReadMessage(peer1Branch) : System.Array.Empty<float>();

            int msgStart = ObsIndices.PartialObsSize;  // = 34
            for (int i = 0; i < _commBandwidth; i++)
            {
                _obsBuffer[msgStart + i] = (i < msg0.Length) ? Mathf.Clamp01(msg0[i]) : 0f;
                _obsBuffer[msgStart + _commBandwidth + i] = (i < msg1.Length) ? Mathf.Clamp01(msg1[i]) : 0f;
            }
        }

        // --------------------------------------------------------------
        // Semantic asserts (editor only — fires on any malformed float)
        // --------------------------------------------------------------
#if UNITY_EDITOR
        float gs = _obsBuffer[ObsIndices.GateState];
        Debug.Assert(gs == 0f || gs == 0.5f || gs == 1f,
            $"[SortingAgent {_branchIndex}] gate_state out of spec: {gs}");

        Debug.Assert(_obsBuffer[ObsIndices.Cooldown] >= 0f && _obsBuffer[ObsIndices.Cooldown] <= 1f,
            $"[SortingAgent {_branchIndex}] cooldown out of [0,1]: {_obsBuffer[ObsIndices.Cooldown]}");
        Debug.Assert(_obsBuffer[ObsIndices.BeltSpeed] >= 0f && _obsBuffer[ObsIndices.BeltSpeed] <= 1f,
            $"[SortingAgent {_branchIndex}] belt_speed out of [0,1]: {_obsBuffer[ObsIndices.BeltSpeed]}");

        float destSum = _obsBuffer[ObsIndices.DestMappingStart + 0]
                      + _obsBuffer[ObsIndices.DestMappingStart + 1]
                      + _obsBuffer[ObsIndices.DestMappingStart + 2];
        Debug.Assert(Mathf.Abs(destSum - 1f) < 0.001f,
            $"[SortingAgent {_branchIndex}] dest mapping is not one-hot (sum={destSum})");

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

        int congestionStart = partial
            ? ObsIndices.CongestionStartPartial
            : ObsIndices.CongestionStartFull;
        for (int b = 0; b < ObsIndices.CongestionWidth; b++)
        {
            float c = _obsBuffer[congestionStart + b];
            Debug.Assert(c >= 0f && c <= 1f,
                $"[SortingAgent {_branchIndex}] congestion[{b}] out of [0,1]: {c}");
        }

        // Message floats (if any) in [0, 1]
        if (commFloats > 0)
        {
            for (int i = 0; i < commFloats; i++)
            {
                float m = _obsBuffer[ObsIndices.PartialObsSize + i];
                Debug.Assert(m >= 0f && m <= 1f,
                    $"[SortingAgent {_branchIndex}] message[{i}] out of [0,1]: {m}");
            }
        }
#endif

        // Commit to sensor
        for (int i = 0; i < obsSize; i++)
        {
            sensor.AddObservation(_obsBuffer[i]);
        }
    }
}