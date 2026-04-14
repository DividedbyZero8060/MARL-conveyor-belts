using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Shared observation assembly logic used by both SortingAgent (decentralised)
/// and CentralisedAgent (Step 15c baseline). Produces a 34- or 38-float
/// observation block for a single branch's perspective, following the layout
/// documented in ObsIndices.cs.
///
/// The function is pure w.r.t. the supplied context — no hidden state. It
/// writes directly to a pre-allocated buffer at a caller-chosen offset so
/// CentralisedAgent can stitch three blocks together without intermediate
/// allocations.
///
/// Why this exists:
///   - SortingAgent.CollectObservations previously owned the entire 38-float
///     computation. Extracting it lets CentralisedAgent reuse the same logic
///     three times (once per branch) to build its 114-float concatenated
///     observation without reimplementing observation semantics.
///   - Ensures bit-for-bit equivalence: every 38-float block CentralisedAgent
///     writes is identical to what the corresponding SortingAgent would have
///     written from its perspective.
///
/// Place in: Assets/Scripts/Agents/
/// </summary>
public static class ObservationBuilder
{
    /// <summary>
    /// Per-branch context bundle. Construct one per branch, then call
    /// <see cref="WriteBranchObservation"/>. All fields are references or
    /// primitives, no allocation per call.
    ///
    /// BranchIndex identifies which branch (0/1/2) this context represents.
    /// Gate, PackageDetector, OverlappingPackages, AllBranchTrackers are the
    /// per-branch and scene-wide scene references. PeerGateStates and
    /// PeerNearestDistances hold the 2 other agents' data (used only in
    /// full-observability mode).
    /// </summary>
    public struct ObservationContext
    {
        public int BranchIndex;
        public DiverterGate Gate;
        public PackageDetector PackageDetector;
        public IReadOnlyList<Package> OverlappingPackages;
        public BranchTracker[] AllBranchTrackers;

        // Peer features (full-obs only). Length 2 expected. If either slot is
        // null, the corresponding peer contributes 0 to gate state and 1 to
        // nearest distance (matches SortingAgent's original fallback logic).
        public float[] PeerGateStates;        // length 2
        public float[] PeerNearestDistances;  // length 2

        // Cached nearest-package distance output. After WriteBranchObservation
        // runs, this is set to slot 0's normalised distance (or 1f if slot 0
        // is empty). SortingAgent uses this to update its own cached value
        // so peers can read it next frame. CentralisedAgent ignores it.
        public float NearestPackageDistanceOut;
    }

    /// <summary>
    /// Writes a 34- or 38-float observation block into <paramref name="buffer"/>
    /// starting at <paramref name="offset"/>, for the branch described by
    /// <paramref name="ctx"/>.
    ///
    /// The block layout matches ObsIndices.cs exactly. Peer features are only
    /// written in full-obs mode. The caller is responsible for zeroing the
    /// buffer range before calling (or accepting whatever was there — this
    /// function writes every element in [offset, offset + blockSize) so
    /// pre-zeroing is not strictly required).
    ///
    /// Returns the number of floats written (34 or 38).
    /// </summary>
    public static int WriteBranchObservation(
        ref ObservationContext ctx,
        float[] buffer,
        int offset,
        bool partialObservability)
    {
        int blockSize = partialObservability
            ? ObsIndices.PartialObsSize
            : ObsIndices.FullObsSize;

        Debug.Assert(buffer != null && buffer.Length >= offset + blockSize,
            $"[ObservationBuilder] buffer too small: length {buffer?.Length}, " +
            $"needs {offset + blockSize}");

        // [0] gate_state
        buffer[offset + ObsIndices.GateState] = GetNormalisedGateState(ctx.Gate);

        // [1] cooldown
        buffer[offset + ObsIndices.Cooldown] =
            ctx.Gate != null ? Mathf.Clamp01(ctx.Gate.NormalisedCooldownRemaining) : 0f;

        // [2] belt_speed
        if (BeltSpeedController.Instance != null)
        {
            float maxSpeed = BeltSpeedController.Instance.MaxSpeed;
            float speed = BeltSpeedController.Instance.CurrentSpeed;
            buffer[offset + ObsIndices.BeltSpeed] =
                maxSpeed > 0f ? Mathf.Clamp01(speed / maxSpeed) : 0f;
        }
        else
        {
            buffer[offset + ObsIndices.BeltSpeed] = 0f;
        }

        // [3..6) destination mapping one-hot for THIS branch
        buffer[offset + ObsIndices.DestMappingStart + 0] = 0f;
        buffer[offset + ObsIndices.DestMappingStart + 1] = 0f;
        buffer[offset + ObsIndices.DestMappingStart + 2] = 0f;
        if (EnvironmentManager.Instance != null)
        {
            DestinationLabel myDest =
                EnvironmentManager.Instance.GetDestinationForBranch(ctx.BranchIndex);
            switch (myDest)
            {
                case DestinationLabel.DestA:
                    buffer[offset + ObsIndices.DestMappingStart + 0] = 1f;
                    break;
                case DestinationLabel.DestB:
                    buffer[offset + ObsIndices.DestMappingStart + 1] = 1f;
                    break;
                case DestinationLabel.DestC:
                    buffer[offset + ObsIndices.DestMappingStart + 2] = 1f;
                    break;
            }
        }

        // [6..31) 5 package slots × 5 floats
        if (ctx.PackageDetector != null)
        {
            ctx.PackageDetector.Refresh(ctx.OverlappingPackages);
            ctx.PackageDetector.WriteObservations(buffer, offset + ObsIndices.PackageSlotsStart);
        }
        else
        {
            // Zero the 25-float package block.
            for (int i = 0; i < ObsIndices.PackageSlotCount * ObsIndices.PackageSlotWidth; i++)
            {
                buffer[offset + ObsIndices.PackageSlotsStart + i] = 0f;
            }
        }

        // Cache slot 0's distance for peer-feature reads next frame.
        float slot0Present = buffer[offset + ObsIndices.PackageSlotsStart + 0];
        float slot0Dist = buffer[offset + ObsIndices.PackageSlotsStart + 1];
        ctx.NearestPackageDistanceOut = slot0Present > 0.5f ? slot0Dist : 1f;

        // Peer features (FULL only): [31..35)
        //   [31] peer0 gate_state
        //   [32] peer1 gate_state
        //   [33] peer0 nearest_package_distance
        //   [34] peer1 nearest_package_distance
        if (!partialObservability)
        {
            int peerStart = offset + ObsIndices.OtherAgentsStartFull;
            float peer0Gate = (ctx.PeerGateStates != null && ctx.PeerGateStates.Length > 0)
                ? ctx.PeerGateStates[0] : 0f;
            float peer1Gate = (ctx.PeerGateStates != null && ctx.PeerGateStates.Length > 1)
                ? ctx.PeerGateStates[1] : 0f;
            float peer0Dist = (ctx.PeerNearestDistances != null && ctx.PeerNearestDistances.Length > 0)
                ? ctx.PeerNearestDistances[0] : 1f;
            float peer1Dist = (ctx.PeerNearestDistances != null && ctx.PeerNearestDistances.Length > 1)
                ? ctx.PeerNearestDistances[1] : 1f;

            buffer[peerStart + 0] = peer0Gate;
            buffer[peerStart + 1] = peer1Gate;
            buffer[peerStart + 2] = peer0Dist;
            buffer[peerStart + 3] = peer1Dist;
        }

        // Congestion (3 floats): indices depend on mode
        int congestionStart = offset + (partialObservability
            ? ObsIndices.CongestionStartPartial
            : ObsIndices.CongestionStartFull);

        if (ctx.AllBranchTrackers != null)
        {
            for (int b = 0; b < ObsIndices.CongestionWidth; b++)
            {
                BranchTracker tracker = (b < ctx.AllBranchTrackers.Length)
                    ? ctx.AllBranchTrackers[b] : null;
                buffer[congestionStart + b] =
                    tracker != null ? Mathf.Clamp01(tracker.NormalisedCongestion) : 0f;
            }
        }
        else
        {
            for (int b = 0; b < ObsIndices.CongestionWidth; b++)
            {
                buffer[congestionStart + b] = 0f;
            }
        }

        return blockSize;
    }

    /// <summary>
    /// Computes the 0/0.5/1 normalised gate state from a DiverterGate's
    /// current FSM state. Committed is treated as transit-equivalent (0.5)
    /// per the Step 14 Option C decision.
    /// </summary>
    public static float GetNormalisedGateState(DiverterGate gate)
    {
        if (gate == null) return 0f;
        switch (gate.CurrentState)
        {
            case GateState.Retracted: return 0f;
            case GateState.Deployed: return 1f;
            case GateState.Committed:
            case GateState.Deploying:
            case GateState.Retracting: return 0.5f;
            default: return 0f;
        }
    }
}