
    /// <summary>
    /// Canonical observation vector indices for SortingAgent.
    ///
    /// CollectObservations writes floats in this exact order:
    ///   [gate_state, cooldown, belt_speed, dest_mapping(3),
    ///    package_slots(25), other_agents(4 — full-obs only),
    ///    congestion(3), messages(0/2/6 — comm variants only)]
    ///
    /// Full observability  : 38 floats  (other_agents present, no messages)
    /// Partial observability: 34 floats (other_agents removed,  no messages)
    /// Partial + comm1     : 36 floats (+2 message floats AFTER congestion)
    /// Partial + comm3     : 40 floats (+6 message floats AFTER congestion)
    ///
    /// IMPORTANT: messages are appended AFTER congestion. Python critic slicing
    /// must use FIXED indices for congestion (IDX_CONGESTION_START_PARTIAL),
    /// NOT negative indexing like obs[-3:], because obs[-3:] would grab message
    /// floats in comm variants. See workflow Step 17 "Fixed slicing" section.
    /// </summary>
    public static class ObsIndices
    {
        // ---- Local features (always present) ----
        public const int GateState = 0;   // 1 float : 0=retracted, 0.5=transitioning, 1=deployed
        public const int Cooldown = 1;   // 1 float : normalised [0,1] cooldown remaining
        public const int BeltSpeed = 2;   // 1 float : currentSpeed / maxSpeed, in [0,1]

        public const int DestMappingStart = 3;   // 3 floats: one-hot {DestA,DestB,DestC} for THIS branch
        public const int DestMappingEnd = 6;   // exclusive

        public const int PackageSlotsStart = 6;  // 5 slots × 5 floats = 25
        public const int PackageSlotsEnd = 31; // exclusive
        public const int PackageSlotCount = 5;
        public const int PackageSlotWidth = 5;  // {present, distance, type_onehot(3)}

        // ---- Other-agent features (FULL observability only) ----
        // Layout: [otherGate0, otherGate1, otherNearestDist0, otherNearestDist1]
        public const int OtherAgentsStartFull = 31; // 4 floats, present only when partialObservability=false
        public const int OtherAgentsEndFull = 35; // exclusive

        // ---- Congestion (always present, but at different indices per mode) ----
        // Full obs:    indices 35..38   (after the 4 other-agent floats)
        // Partial obs: indices 31..34   (other-agent floats removed)
        public const int CongestionStartFull = 35;
        public const int CongestionEndFull = 38; // exclusive

        public const int CongestionStartPartial = 31;
        public const int CongestionEndPartial = 34; // exclusive

        public const int CongestionWidth = 3; // 3 branches

        // ---- Total sizes ----
        public const int FullObsSize = 38;
        public const int PartialObsSize = 34;

        // ---- Hysteresis / detector tuning (used by PackageDetector) ----
        public const float SlotSwapHysteresisMeters = 0.15f;
    }
