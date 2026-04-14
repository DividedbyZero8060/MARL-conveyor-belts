using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using UnityEngine;

/// <summary>
/// SortingAgent action handling (Step 09).
///
/// Handles both action-space variants from one codebase. The active mode is
/// detected at Initialize time by reading BehaviorParameters.ActionSpec:
///
///   Discrete [2]        → MA-POCA, PPO, DQN. Native framework masking via
///                         WriteDiscreteActionMask. Action 0 = do nothing,
///                         Action 1 = activate gate.
///
///   Continuous 1        → MADDPG. NO framework masking — a C# state guard
///                         in OnActionReceived prevents illegal activations
///                         during gate animation. Python masks the replay
///                         buffer separately.
///
/// Heuristic: Q activates Gate 0, W Gate 1, E Gate 2. Works in both modes.
/// Each agent only reacts to its own key (branchIndex → key).
/// </summary>
public partial class SortingAgent
{
    private enum ActionMode
    {
        Unknown,
        Discrete,    // Built-in trainers (MA-POCA, PPO, DQN)
        Continuous   // Custom trainers (MADDPG)
    }

    private ActionMode _actionMode = ActionMode.Unknown;
    private BehaviorParameters _behaviorParameters;

    /// <summary>
    /// When true, Heuristic() uses the rule-based automatic controller
    /// (perfect-information sort) instead of Q/W/E keyboard input. Set by
    /// EvaluationRunner before the heuristic baseline run, or manually in
    /// the Inspector for ad-hoc testing. The agent's BehaviorType must be
    /// Heuristic Only for this to take effect.
    /// </summary>
    public bool UseAutomaticHeuristic { get; set; } = false;

    /// <summary>
    /// Distance threshold (normalised, in [0,1]) below which the automatic
    /// heuristic considers a slot-0 package "close enough" to commit. The
    /// PackageDetector emits distances normalised by detection range (15m),
    /// so 0.05 corresponds to ~0.75m physical distance. With Option C
    /// pre-commit semantics, the agent only needs to fire early enough that
    /// the package will reach the commit zone soon — exact timing is
    /// physics-driven, so this threshold is forgiving.
    /// </summary>
    [Tooltip("Slot-0 distance threshold for automatic heuristic activation. " +
             "Normalised, [0,1]. With Option C, this can be lenient (~0.20).")]
    public float AutomaticHeuristicDistanceThreshold = 0.20f;

    /// <summary>
    /// Threshold above which a continuous gate action is treated as "activate".
    /// Applied ONLY in C#; the raw continuous value is what the Python critic sees.
    /// </summary>
    private const float ContinuousActivationThreshold = 0.5f;

    /// <summary>
    /// Called from SortingAgent.Initialize() to cache the action mode by
    /// inspecting the attached BehaviorParameters component.
    /// </summary>
    internal void DetectActionMode()
    {
        _behaviorParameters = GetComponent<BehaviorParameters>();
        Debug.Assert(_behaviorParameters != null,
            $"[SortingAgent {_branchIndex}] BehaviorParameters component missing.", this);

        if (_behaviorParameters == null)
        {
            _actionMode = ActionMode.Unknown;
            return;
        }

        ActionSpec spec = _behaviorParameters.BrainParameters.ActionSpec;

        if (spec.NumContinuousActions > 0)
        {
            _actionMode = ActionMode.Continuous;
            Debug.Assert(spec.NumContinuousActions >= 1,
                $"[SortingAgent {_branchIndex}] continuous mode requires >=1 continuous action, got {spec.NumContinuousActions}.");
        }
        else if (spec.BranchSizes != null && spec.BranchSizes.Length > 0)
        {
            _actionMode = ActionMode.Discrete;
            Debug.Assert(spec.BranchSizes[0] == 2,
                $"[SortingAgent {_branchIndex}] discrete mode expects branch 0 size 2, got {spec.BranchSizes[0]}.");
        }
        else
        {
            _actionMode = ActionMode.Unknown;
            Debug.LogError(
                $"[SortingAgent {_branchIndex}] BehaviorParameters has no actions configured.", this);
        }
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        _decisionCount++;
        switch (_actionMode)
        {
            case ActionMode.Discrete:
                {
                    // Framework has already applied WriteDiscreteActionMask below,
                    // so action 1 should only arrive when gate is Retracted. We
                    // still call TryActivateGate() which is idempotent — Activate()
                    // returns false if not actionable.
                    int gateAction = actions.DiscreteActions[0];
                    if (gateAction == 1)
                    {
                        _activationCount++;
                        TryActivateGate();
                    }
                    break;
                }

            case ActionMode.Continuous:
                {
                    // MADDPG: Python forces masked_action[0] = 0.0 when gate is
                    // not retracted, but we defend anyway in case of threshold
                    // drift or a raw inference path skipping the mask.
                    float gateAction = actions.ContinuousActions[0];
                    if (gateAction > ContinuousActivationThreshold
                        && _gate != null
                        && _gate.CurrentState == GateState.Retracted)
                    {
                        _activationCount++;
                        TryActivateGate();
                    }
                    break;
                }

            case ActionMode.Unknown:
            default:
                // DetectActionMode already logged an error. Fail silent here
                // so we don't spam the Console every decision step.
                break;
        }
    }

    /// <summary>
    /// Discrete action masking. Called by the ML-Agents framework every
    /// decision on discrete-mode agents. NOT called on continuous-mode agents.
    /// Masks action index 1 (activate) when the gate is not Retracted, so
    /// the policy cannot select an illegal activation during gate animation.
    /// </summary>
    public override void WriteDiscreteActionMask(IDiscreteActionMask actionMask)
    {
        // Safety: if called on a continuous agent somehow, do nothing.
        if (_actionMode != ActionMode.Discrete) return;

        if (_gate != null && _gate.CurrentState != GateState.Retracted)
        {
            // Modern API: SetActionEnabled(branch, actionIndex, isEnabled).
            // Do NOT use the old SetMask() method.
            actionMask.SetActionEnabled(0, 1, false);
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        bool wantActivate;

        if (UseAutomaticHeuristic)
        {
            wantActivate = ComputeAutomaticHeuristicActivation();
        }
        else
        {
            // Manual Q/W/E mode (existing behaviour).
            KeyCode myKey;
            switch (_branchIndex)
            {
                case 0: myKey = KeyCode.Q; break;
                case 1: myKey = KeyCode.W; break;
                case 2: myKey = KeyCode.E; break;
                default: myKey = KeyCode.None; break;
            }
            wantActivate = (myKey != KeyCode.None) && Input.GetKey(myKey);
        }

        if (_actionMode == ActionMode.Discrete)
        {
            ActionSegment<int> discreteOut = actionsOut.DiscreteActions;
            if (discreteOut.Length > 0)
            {
                discreteOut[0] = wantActivate ? 1 : 0;
            }
        }
        else if (_actionMode == ActionMode.Continuous)
        {
            ActionSegment<float> continuousOut = actionsOut.ContinuousActions;
            if (continuousOut.Length > 0)
            {
                continuousOut[0] = wantActivate ? 1f : 0f;
            }
        }
        // Unknown mode: leave buffer at default (zeros).
    }

    /// <summary>
    /// Rule-based perfect-information heuristic. Returns true if the agent
    /// should activate its gate this decision step.
    ///
    /// Logic:
    ///   1. If the gate is not actionable (animating or cooling down), do nothing.
    ///   2. Read slot 0 from the agent's observation slots — closest detected package.
    ///   3. If slot 0 is empty, do nothing.
    ///   4. If slot 0's destination matches THIS branch's currently assigned
    ///      destination AND slot 0's distance is below the threshold, activate.
    ///   5. Otherwise, do nothing.
    ///
    /// This uses perfect information — the heuristic looks up the current
    /// branch mapping via EnvironmentManager.GetDestinationForBranch and
    /// reads the closest package's destination label directly. Trained
    /// agents must learn the same mapping from observation only.
    /// </summary>
    private bool ComputeAutomaticHeuristicActivation()
    {
        if (_gate == null || !_gate.IsActionable) return false;

        // Find the closest in-range package by querying the detector's
        // overlapping list directly (same source the observation slots use).
        if (_overlappingPackages == null || _overlappingPackages.Count == 0)
            return false;

        // The detector's WriteObservations sorts by upstream distance, but
        // we don't have direct access to that ordering from here — instead,
        // recompute using the same metric (dot product against belt forward).
        Package closestPackage = null;
        float closestDistance = float.MaxValue;
        Vector3 agentPosition = transform.position;

        for (int i = 0; i < _overlappingPackages.Count; i++)
        {
            Package pkg = _overlappingPackages[i];
            if (pkg == null || !pkg.gameObject.activeInHierarchy) continue;
            float dist = Vector3.Distance(agentPosition, pkg.transform.position);
            if (dist < closestDistance)
            {
                closestDistance = dist;
                closestPackage = pkg;
            }
        }

        if (closestPackage == null) return false;

        // Normalise distance the same way the detector does: divide by the
        // detection range. PackageDetector uses 15m by convention.
        float normalisedDistance = Mathf.Clamp01(closestDistance / 15f);
        if (normalisedDistance > AutomaticHeuristicDistanceThreshold) return false;

        // Look up THIS branch's currently assigned destination.
        if (EnvironmentManager.Instance == null) return false;
        DestinationLabel myDestination = EnvironmentManager.Instance
            .GetDestinationForBranch(_branchIndex);

        // Compare with the closest package's destination.
        return closestPackage.DestinationLabel == myDestination;
    }

    /// <summary>
    /// Single-point gate activation. Ignores the return value — unsuccessful
    /// activations (during animation or cooldown) are expected and harmless;
    /// the policy learns to time them.
    /// </summary>
    private void TryActivateGate()
    {
        if (_gate == null) return;
        _gate.Activate();
    }
}